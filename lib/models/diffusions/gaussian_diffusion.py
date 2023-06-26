import math
import sys
import numpy as np
import torch
import torch.nn as nn
import mmcv

from copy import deepcopy
from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES, build_module
from mmgen.models.diffusions.utils import var_to_tensor, _get_noise_batch


@MODULES.register_module()
class GaussianDiffusion(nn.Module):

    def __init__(self,
                 denoising,
                 ddpm_loss=dict(
                     type='DDPMMSELoss',
                     log_cfgs=dict(
                         type='quartile', prefix_name='loss_mse', total_timesteps=1000)),
                 betas_cfg=dict(type='cosine'),
                 num_timesteps=1000,
                 num_classes=0,
                 sample_method='ddim',
                 timestep_sampler=dict(type='UniformTimeStepSampler'),
                 denoising_var_mode='FIXED_LARGE',
                 denoising_mean_mode='V',
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        # build denoising module in this function
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.sample_method = sample_method
        self._denoising_cfg = deepcopy(denoising)
        self.denoising = build_module(
            denoising,
            default_args=dict(
                num_classes=num_classes, num_timesteps=num_timesteps))

        # get output-related configs from denoising
        self.denoising_var_mode = denoising_var_mode
        self.denoising_mean_mode = denoising_mean_mode

        self.betas_cfg = deepcopy(betas_cfg)

        self.train_cfg = deepcopy(train_cfg) if train_cfg is not None else dict()
        self.test_cfg = deepcopy(test_cfg) if test_cfg is not None else dict()

        self.prepare_diffusion_vars()

        # build sampler
        self.sampler = build_module(
            timestep_sampler,
            default_args=dict(
                num_timesteps=num_timesteps,
                mean=self.sqrt_alphas_bar,
                std=self.sqrt_one_minus_alphas_bar,
                mode=self.denoising_mean_mode))
        self.ddpm_loss = build_module(ddpm_loss, default_args=dict(sampler=self.sampler))

    @staticmethod
    def linear_beta_schedule(diffusion_timesteps, beta_0=1e-4, beta_T=2e-2):
        r"""Linear schedule from Ho et al, extended to work for any number of
        diffusion steps.

        Args:
            diffusion_timesteps (int): The number of betas to produce.
            beta_0 (float, optional): `\beta` at timestep 0. Defaults to 1e-4.
            beta_T (float, optional): `\beta` at timestep `T` (the final
                diffusion timestep). Defaults to 2e-2.

        Returns:
            np.ndarray: Betas used in diffusion process.
        """
        scale = 1000 / diffusion_timesteps
        beta_0 = scale * beta_0
        beta_T = scale * beta_T
        return np.linspace(
            beta_0, beta_T, diffusion_timesteps, dtype=np.float64)

    @staticmethod
    def cosine_beta_schedule(diffusion_timesteps, max_beta=0.999, s=0.008):
        r"""Create a beta schedule that discretizes the given alpha_t_bar
        function, which defines the cumulative product of `(1-\beta)` over time
        from `t = [0, 1]`.

        Args:
            diffusion_timesteps (int): The number of betas to produce.
            max_beta (float, optional): The maximum beta to use; use values
                lower than 1 to prevent singularities. Defaults to 0.999.
            s (float, optional): Small offset to prevent `\beta` from being too
                small near `t = 0` Defaults to 0.008.

        Returns:
            np.ndarray: Betas used in diffusion process.
        """

        def f(t, T, s):
            return np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2

        betas = []
        for t in range(diffusion_timesteps):
            alpha_bar_t = f(t + 1, diffusion_timesteps, s)
            alpha_bar_t_1 = f(t, diffusion_timesteps, s)
            betas_t = 1 - alpha_bar_t / alpha_bar_t_1
            betas.append(min(betas_t, max_beta))
        return np.array(betas)

    def get_betas(self):
        """Get betas by defined schedule method in diffusion process."""
        self.betas_schedule = self.betas_cfg.pop('type')
        if self.betas_schedule == 'linear':
            return self.linear_beta_schedule(self.num_timesteps,
                                             **self.betas_cfg)
        elif self.betas_schedule == 'cosine':
            return self.cosine_beta_schedule(self.num_timesteps,
                                             **self.betas_cfg)
        elif self.betas_schedule == 'scaled_linear':
            return np.linspace(
                self.betas_cfg.get('beta_start', 0.0001) ** 0.5,
                self.betas_cfg.get('beta_end', 0.02) ** 0.5,
                self.num_timesteps,
                dtype=np.float64) ** 2
        else:
            raise AttributeError(f'Unknown method name {self.beta_schedule}'
                                 'for beta schedule.')

    def prepare_diffusion_vars(self):
        self.betas = self.get_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_bar = np.cumproduct(self.alphas, axis=0)
        self.alphas_bar_prev = np.append(1.0, self.alphas_bar[:-1])
        self.alphas_bar_next = np.append(self.alphas_bar[1:], 0.0)

        # calculations for diffusion q(x_t | x_0) and others
        self.sqrt_alphas_bar = np.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = np.sqrt(1.0 - self.alphas_bar)
        self.log_one_minus_alphas_bar = np.log(1.0 - self.alphas_bar)
        self.sqrt_recip_alplas_bar = np.sqrt(1.0 / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = np.sqrt(1.0 / self.alphas_bar - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.tilde_betas_t = self.betas * (1 - self.alphas_bar_prev) / (
            1 - self.alphas_bar)
        # clip log var for tilde_betas_0 = 0
        self.log_tilde_betas_t_clipped = np.log(
            np.append(self.tilde_betas_t[1], self.tilde_betas_t[1:]))
        self.tilde_mu_t_coef1 = np.sqrt(
            self.alphas_bar_prev) / (1 - self.alphas_bar) * self.betas
        self.tilde_mu_t_coef2 = np.sqrt(
            self.alphas) * (1 - self.alphas_bar_prev) / (1 - self.alphas_bar)

    def q_posterior_mean(self, x_0, x_t, t):
        device = get_module_device(self)
        tar_shape = x_0.shape
        tilde_mu_t_coef1 = var_to_tensor(self.tilde_mu_t_coef1, t, tar_shape,
                                         device)
        tilde_mu_t_coef2 = var_to_tensor(self.tilde_mu_t_coef2, t, tar_shape,
                                         device)
        posterior_mean = tilde_mu_t_coef1 * x_0 + tilde_mu_t_coef2 * x_t
        return posterior_mean

    def q_sample(self, x_0, t, noise=None):
        device = get_module_device(self)
        tar_shape = x_0.shape
        if noise is None:
            noise = _get_noise_batch(
                None, x_0.shape[-3:],
                num_timesteps=self.num_timesteps,
                num_batches=x_0.size(0),
                timesteps_noise=False
            ).to(x_0.device)  # (num_batches, num_channels, h, w)
        mean = var_to_tensor(self.sqrt_alphas_bar, t.cpu(), tar_shape, device)
        std = var_to_tensor(self.sqrt_one_minus_alphas_bar, t.cpu(), tar_shape, device)
        return x_0 * mean + noise * std, mean, std

    def pred_x_0(self, x_t, t, grad_guide_fn=None, concat_cond=None, cfg=dict(), update_denoising_output=False):
        clip_denoised = cfg.get('clip_denoised', True)
        clip_range = cfg.get('clip_range', [-1, 1])
        guidance_gain = cfg.get('guidance_gain', 1.0)
        grad_through_unet = cfg.get('grad_through_unet', True)
        snr_weight_power = cfg.get('snr_weight_power', 0.5)

        num_batches = x_t.size(0)
        if t.dim() == 0 or len(t) != num_batches:
            t = t.expand(num_batches)
        sqrt_alpha_bar_t = x_t.new_tensor(self.sqrt_alphas_bar)[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = x_t.new_tensor(self.sqrt_one_minus_alphas_bar)[t].reshape(-1, 1, 1, 1)

        if grad_guide_fn is not None and grad_through_unet:
            x_t.requires_grad = True
            grad_enabled_prev = torch.is_grad_enabled()
            torch.set_grad_enabled(True)

        denoising_output = self.denoising(x_t, t, concat_cond=concat_cond)

        if self.denoising_mean_mode.upper() == 'EPS':
            x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * denoising_output) / sqrt_alpha_bar_t
        elif self.denoising_mean_mode.upper() == 'START_X':
            x_0_pred = denoising_output
        elif self.denoising_mean_mode.upper() == 'V':
            x_0_pred = sqrt_alpha_bar_t * x_t - sqrt_one_minus_alpha_bar_t * denoising_output
        else:
            raise AttributeError('Unknown denoising mean output type '
                                 f'[{self.denoising_mean_mode}].')

        if grad_guide_fn is not None:
            if clip_denoised:
                x_0_pred = x_0_pred.clamp(*clip_range)
            if grad_through_unet:
                loss = grad_guide_fn(x_0_pred)
                grad = torch.autograd.grad(loss, x_t)[0]
            else:
                x_0_pred.requires_grad = True
                grad_enabled_prev = torch.is_grad_enabled()
                torch.set_grad_enabled(True)
                loss = grad_guide_fn(x_0_pred)
                grad = torch.autograd.grad(loss, x_0_pred)[0]
            torch.set_grad_enabled(grad_enabled_prev)
            x_0_pred.detach_()
            x_0_pred -= grad * (
                (sqrt_one_minus_alpha_bar_t ** (2 - snr_weight_power * 2))
                * (sqrt_alpha_bar_t ** (snr_weight_power * 2 - 1))
                * guidance_gain)

        if clip_denoised:
            x_0_pred = x_0_pred.clamp(*clip_range)

        if update_denoising_output and grad_guide_fn is not None:
            if self.denoising_mean_mode.upper() == 'EPS':
                denoising_output = (x_t - x_0_pred * sqrt_alpha_bar_t) / sqrt_one_minus_alpha_bar_t
            elif self.denoising_mean_mode.upper() == 'START_X':
                denoising_output = x_0_pred
            elif self.denoising_mean_mode.upper() == 'V':
                denoising_output = (sqrt_alpha_bar_t * x_t - x_0_pred) / sqrt_one_minus_alpha_bar_t

        return x_0_pred, denoising_output

    def p_sample_langevin(self,
                          x_t,
                          t,
                          noise=None,
                          cfg=dict(),
                          grad_guide_fn=None,
                          **kwargs):
        device = get_module_device(self)
        langevin_delta = cfg.get('langevin_delta', 0.1)
        sigma = self.sqrt_one_minus_alphas_bar[t]
        x_0_pred, _ = self.pred_x_0(x_t, t, grad_guide_fn=grad_guide_fn, cfg=cfg, **kwargs)
        eps_t_pred = (x_t - self.sqrt_alphas_bar[t] * x_0_pred) / sigma
        if noise is None:
            noise = _get_noise_batch(
                None, x_t.shape[-3:],
                num_timesteps=self.num_timesteps,
                num_batches=x_t.size(0),
                timesteps_noise=False
            ).to(device)  # (num_batches, num_channels, h, w)
        x_t = x_t - 0.5 * langevin_delta * sigma * eps_t_pred + math.sqrt(langevin_delta) * sigma * noise
        return x_t

    def p_sample_ddim(self,
                      x_t,
                      t,
                      t_prev,
                      noise=None,
                      cfg=dict(),
                      grad_guide_fn=None,
                      **kwargs):
        device = get_module_device(self)
        eta = cfg.get('eta', 0)  # equivalent to 'FIXED_SMALL' in DDPM if eta == 1

        alpha_bar_t_prev = self.alphas_bar[t_prev] if t_prev >= 0 else self.alphas_bar_prev[0]
        tilde_beta_t = self.tilde_betas_t[t]

        x_0_pred, _ = self.pred_x_0(x_t, t, grad_guide_fn=grad_guide_fn, cfg=cfg, **kwargs)
        eps_t_pred = (x_t - self.sqrt_alphas_bar[t] * x_0_pred) / self.sqrt_one_minus_alphas_bar[t]
        pred_sample_direction = np.sqrt(1 - alpha_bar_t_prev - tilde_beta_t * (eta ** 2)) * eps_t_pred
        x_prev = np.sqrt(alpha_bar_t_prev) * x_0_pred + pred_sample_direction

        if eta > 0:
            if noise is None:
                noise = _get_noise_batch(
                    None, x_t.shape[-3:],
                    num_timesteps=self.num_timesteps,
                    num_batches=x_t.size(0),
                    timesteps_noise=False
                ).to(device)  # (num_batches, num_channels, h, w)
            x_prev = x_prev + eta * np.sqrt(tilde_beta_t) * noise

        return x_prev, x_0_pred

    def ddim_sample(self, noise, show_pbar=False, concat_cond=None,
                    save_intermediates=False, **kwargs):
        device = get_module_device(self)
        x_t = noise
        num_timesteps = self.test_cfg.get('num_timesteps', self.num_timesteps)
        langevin_steps = self.test_cfg.get('langevin_steps', 0)
        langevin_t_range = self.test_cfg.get('langevin_t_range', [0, 1000])
        timesteps = torch.arange(
            start=self.num_timesteps - 1, end=-1, step=-(self.num_timesteps / num_timesteps)
        ).long().to(device)
        if show_pbar:
            pbar = mmcv.ProgressBar(len(timesteps))
        cond_step = 0
        x_0_x_t_list = [] if save_intermediates else None
        for step, t in enumerate(timesteps):
            if step + 1 < len(timesteps):
                t_prev = timesteps[step + 1]
            else:
                t_prev = -1
            x_t, x_0_pred = self.p_sample_ddim(
                x_t, t, t_prev, concat_cond=concat_cond[:, cond_step % concat_cond.size(1)] if concat_cond is not None else None,
                cfg=self.test_cfg, **kwargs)
            cond_step += 1
            if langevin_steps > 0 and langevin_t_range[0] < t_prev < langevin_t_range[1]:
                for _ in range(langevin_steps):
                    x_t = self.p_sample_langevin(
                        x_t, t_prev, concat_cond=concat_cond[:, cond_step % concat_cond.size(1)] if concat_cond is not None else None,
                        cfg=self.test_cfg, **kwargs)
                    cond_step += 1
            if x_0_x_t_list is not None:
                x_0_x_t_list.append(x_0_pred)
                x_0_x_t_list.append(x_t)
            if show_pbar:
                pbar.update()
        if show_pbar:
            sys.stdout.write('\n')
        return x_0_x_t_list if save_intermediates else x_t

    def p_sample_ddpm(self,
                      x_t,
                      t,
                      noise=None,
                      cfg=dict(),
                      grad_guide_fn=None,
                      **kwargs):
        device = get_module_device(self)

        target_shape = x_t.shape
        if self.denoising_var_mode.upper() == 'FIXED_LARGE':
            var_pred = var_to_tensor(np.append(self.tilde_betas_t[1], self.betas), t, target_shape, device)
        elif self.denoising_var_mode.upper() == 'FIXED_SMALL':
            var_pred = var_to_tensor(self.tilde_betas_t, t, target_shape, device)
        else:
            raise AttributeError('Unknown denoising var output type '
                                 f'[{self.denoising_var_mode}].')

        x_0_pred, _ = self.pred_x_0(x_t, t, grad_guide_fn=grad_guide_fn, cfg=cfg, **kwargs)
        mean_pred = self.q_posterior_mean(x_0_pred, x_t, t)

        if noise is None:
            noise = _get_noise_batch(
                None, x_t.shape[-3:],
                num_timesteps=self.num_timesteps,
                num_batches=x_t.size(0),
                timesteps_noise=False
            ).to(device)  # (num_batches, num_channels, h, w)

        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        x_prev = mean_pred + nonzero_mask * torch.sqrt(var_pred) * noise

        return x_prev, x_0_pred

    def ddpm_sample(self, noise, show_pbar=False, concat_cond=None, **kwargs):
        device = get_module_device(self)
        x_t = noise
        num_timesteps = self.test_cfg.get('num_timesteps', self.num_timesteps)
        timesteps = torch.arange(
            start=self.num_timesteps - 1, end=-1, step=-(self.num_timesteps / num_timesteps)
        ).long().to(device)
        if show_pbar:
            pbar = mmcv.ProgressBar(len(timesteps))
        cond_step = 0
        for t in timesteps:
            x_t, _ = self.p_sample_ddpm(
                x_t, t, concat_cond=concat_cond[:, cond_step % concat_cond.size(1)] if concat_cond is not None else None,
                cfg=self.test_cfg, **kwargs)
            cond_step += 1
            if show_pbar:
                pbar.update()
        if show_pbar:
            sys.stdout.write('\n')
        return x_t

    def sample_from_noise(self,
                          noise,
                          **kwargs):
        # get sample function by name
        sample_fn_name = f'{self.sample_method.lower()}_sample'
        if not hasattr(self, sample_fn_name):
            raise AttributeError(
                f'Cannot find sample method [{sample_fn_name}] correspond '
                f'to [{self.sample_method}].')
        sample_fn = getattr(self, sample_fn_name)

        outputs = sample_fn(
            noise=noise,
            **kwargs)
        return outputs

    def loss(self, denoising_output, x_0, noise, t, mean, std):
        if self.denoising_mean_mode.upper() == 'EPS':
            loss_kwargs = dict(eps_t_pred=denoising_output)
        elif self.denoising_mean_mode.upper() == 'START_X':
            loss_kwargs = dict(x_0_pred=denoising_output)
        elif self.denoising_mean_mode.upper() == 'V':
            loss_kwargs = dict(v_t_pred=denoising_output)
        else:
            raise AttributeError('Unknown denoising mean output type '
                                 f'[{self.denoising_mean_mode}].')
        loss_kwargs.update(
            x_0=x_0,
            noise=noise,
            timesteps=t)
        if 'v_t_pred' in loss_kwargs:
            loss_kwargs.update(v_t=mean * noise - std * x_0)
        return self.ddpm_loss(loss_kwargs)

    def forward_train(self, x_0, concat_cond=None, grad_guide_fn=None, cfg=dict(),
                      x_t_detach=False, **kwargs):
        device = get_module_device(self)

        assert x_0.dim() == 4
        num_batches, num_channels, h, w = x_0.size()

        t = self.sampler(num_batches).to(device)

        noise = _get_noise_batch(
            None, x_0.shape[-3:],
            num_timesteps=self.num_timesteps,
            num_batches=x_0.size(0),
            timesteps_noise=False
        ).to(device)  # (num_batches, num_channels, h, w)
        x_t, mean, std = self.q_sample(x_0, t, noise)
        if x_t_detach:
            x_t.detach_()

        _, denoising_output = self.pred_x_0(
            x_t, t, grad_guide_fn=grad_guide_fn, concat_cond=concat_cond,
            cfg=cfg, update_denoising_output=True)
        loss = self.loss(denoising_output, x_0, noise, t, mean, std)
        log_vars = self.ddpm_loss.log_vars
        log_vars.update(loss_ddpm_mse=float(loss))

        return loss, log_vars

    def forward_test(self, data, **kwargs):
        """Testing function for Diffusion Denosing Probability Models.

        Args:
            data (torch.Tensor | dict | None): Input data. This data will be
                passed to different methods.
        """
        assert data.dim() == 4
        return self.sample_from_noise(data, **kwargs)

    def forward(self, data, return_loss=False, **kwargs):
        if return_loss:
            return self.forward_train(data, **kwargs)

        return self.forward_test(data, **kwargs)

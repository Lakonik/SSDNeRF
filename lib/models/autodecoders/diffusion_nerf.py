import torch
import mmcv

from copy import deepcopy
from torch.nn.parallel.distributed import DistributedDataParallel
from mmgen.models.builder import MODELS, build_module
from mmgen.models.architectures.common import get_module_device

from ...core import eval_psnr, rgetattr, module_requires_grad
from .base_nerf import get_cam_rays
from .multiscene_nerf import MultiSceneNeRF


@MODELS.register_module()
class DiffusionNeRF(MultiSceneNeRF):

    def __init__(self,
                 *args,
                 diffusion=dict(type='GaussianDiffusion'),
                 diffusion_use_ema=True,
                 freeze_decoder=True,
                 image_cond=False,
                 code_permute=None,
                 code_reshape=None,
                 autocast_dtype=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        diffusion.update(train_cfg=self.train_cfg, test_cfg=self.test_cfg)
        self.diffusion = build_module(diffusion)
        self.diffusion_use_ema = diffusion_use_ema
        if self.diffusion_use_ema:
            self.diffusion_ema = deepcopy(self.diffusion)
        self.freeze_decoder = freeze_decoder
        if self.freeze_decoder:
            self.decoder.requires_grad_(False)
            if self.decoder_use_ema:
                self.decoder_ema.requires_grad_(False)
        self.image_cond = image_cond
        self.code_permute = code_permute
        self.code_reshape = code_reshape
        self.code_reshape_inv = [self.code_size[axis] for axis in self.code_permute] if code_permute is not None \
            else self.code_size
        self.code_permute_inv = [self.code_permute.index(axis) for axis in range(len(self.code_permute))] \
            if code_permute is not None else None

        self.autocast_dtype = autocast_dtype

        for key, value in self.test_cfg.get('override_cfg', dict()).items():
            self.train_cfg_backup[key] = rgetattr(self, key)

    def code_diff_pr(self, code):
        code_diff = code
        if self.code_permute is not None:
            code_diff = code_diff.permute([0] + [axis + 1 for axis in self.code_permute])  # add batch dimension
        if self.code_reshape is not None:
            code_diff = code_diff.reshape(code.size(0), *self.code_reshape)  # add batch dimension
        return code_diff

    def code_diff_pr_inv(self, code_diff):
        code = code_diff
        if self.code_reshape is not None:
            code = code.reshape(code.size(0), *self.code_reshape_inv)
        if self.code_permute_inv is not None:
            code = code.permute([0] + [axis + 1 for axis in self.code_permute_inv])
        return code

    def train_step(self, data, optimizer, running_status=None):
        diffusion = self.diffusion
        decoder = self.decoder_ema if self.freeze_decoder and self.decoder_use_ema else self.decoder

        num_scenes = len(data['scene_id'])
        extra_scene_step = self.train_cfg.get('extra_scene_step', 0)

        if 'optimizer' in self.train_cfg:
            code_list_, code_optimizers, density_grid, density_bitfield = self.load_cache(data)
            code = self.code_activation(torch.stack(code_list_, dim=0), update_stats=True)
        else:
            assert 'code' in data
            code, density_grid, density_bitfield = self.load_scene(
                data, load_density='decoder' in optimizer)
            code_optimizers = []

        for key in optimizer.keys():
            if key.startswith('diffusion'):
                optimizer[key].zero_grad()
        for code_optimizer in code_optimizers:
            code_optimizer.zero_grad()
        if 'decoder' in optimizer:
            optimizer['decoder'].zero_grad()

        concat_cond = None
        if 'cond_imgs' in data:
            cond_imgs = data['cond_imgs']  # (num_scenes, num_imgs, h, w, 3)
            cond_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            cond_poses = data['cond_poses']

            num_scenes, num_imgs, h, w, _ = cond_imgs.size()
            # (num_scenes, num_imgs, h, w, 3)
            cond_rays_o, cond_rays_d = get_cam_rays(cond_poses, cond_intrinsics, h, w)
            dt_gamma_scale = self.train_cfg.get('dt_gamma_scale', 0.0)
            # (num_scenes,)
            dt_gamma = dt_gamma_scale / cond_intrinsics[..., :2].mean(dim=(-2, -1))

            if self.image_cond:
                cond_inds = torch.randint(num_imgs, size=(num_scenes,))  # (num_scenes,)
                concat_cond = cond_imgs[range(num_scenes), cond_inds].permute(0, 3, 1, 2)  # (num_scenes, 3, h, w)
                diff_image_size = rgetattr(diffusion, 'denoising.image_size')
                assert diff_image_size[0] % concat_cond.size(-2) == 0
                assert diff_image_size[1] % concat_cond.size(-1) == 0
                concat_cond = concat_cond.tile((diff_image_size[0] // concat_cond.size(-2),
                                                diff_image_size[1] // concat_cond.size(-1)))

        x_t_detach = self.train_cfg.get('x_t_detach', False)

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
            loss_diffusion, log_vars = diffusion(
                self.code_diff_pr(code), concat_cond=concat_cond, return_loss=True,
                x_t_detach=x_t_detach, cfg=self.train_cfg)
        loss_diffusion.backward()
        for key in optimizer.keys():
            if key.startswith('diffusion'):
                optimizer[key].step()

        if extra_scene_step > 0:
            assert len(code_optimizers) > 0
            prior_grad = [code_.grad.data.clone() for code_ in code_list_]
            cfg = self.train_cfg.copy()
            cfg['n_inverse_steps'] = extra_scene_step
            code, _, _, loss_decoder, loss_dict_decoder, out_rgbs, target_rgbs = self.inverse_code(
                decoder, cond_imgs, cond_rays_o, cond_rays_d, dt_gamma=dt_gamma, cfg=cfg,
                code_=code_list_,
                density_grid=density_grid,
                density_bitfield=density_bitfield,
                code_optimizer=code_optimizers,
                prior_grad=prior_grad)
            for k, v in loss_dict_decoder.items():
                log_vars.update({k: float(v)})
        else:
            prior_grad = None

        if 'decoder' in optimizer or len(code_optimizers) > 0:
            if len(code_optimizers) > 0:
                code = self.code_activation(torch.stack(code_list_, dim=0))
                self.update_extra_state(
                    decoder, code, density_grid, density_bitfield,
                    0, density_thresh=self.train_cfg.get('density_thresh', 0.01))

            loss_decoder, log_vars_decoder, out_rgbs, target_rgbs = self.loss_decoder(
                decoder, code, density_bitfield, cond_rays_o, cond_rays_d,
                cond_imgs, dt_gamma, cfg=self.train_cfg)
            log_vars.update(log_vars_decoder)

            if prior_grad is not None:
                for code_, prior_grad_single in zip(code_list_, prior_grad):
                    code_.grad.copy_(prior_grad_single)
            loss_decoder.backward()

            if 'decoder' in optimizer:
                optimizer['decoder'].step()
            for code_optimizer in code_optimizers:
                code_optimizer.step()

        if len(code_optimizers) > 0:
            # ==== save cache ====
            self.save_cache(
                code_list_, code_optimizers,
                density_grid, density_bitfield, data['scene_id'], data['scene_name'])

            # ==== evaluate reconstruction ====
            with torch.no_grad():
                self.mean_ema_update(code)
                train_psnr = eval_psnr(out_rgbs, target_rgbs)
                code_rms = code.square().flatten(1).mean().sqrt()
                log_vars.update(train_psnr=float(train_psnr.mean()),
                                code_rms=float(code_rms.mean()))
                if 'test_imgs' in data and data['test_imgs'] is not None:
                    log_vars.update(self.eval_and_viz(
                        data, self.decoder, code, density_bitfield, cfg=self.train_cfg)[0])

        # ==== outputs ====
        if 'decoder' in optimizer or len(code_optimizers) > 0:
            log_vars.update(loss_decoder=float(loss_decoder))
        outputs_dict = dict(
            log_vars=log_vars, num_samples=num_scenes)

        return outputs_dict

    def val_uncond(self, data, show_pbar=False, **kwargs):
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder

        num_batches = len(data['scene_id'])
        noise = data.get('noise', None)
        if noise is None:
            noise = torch.randn(
                (num_batches, *self.code_size), device=get_module_device(self))

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
            code_out = diffusion(
                self.code_diff_pr(noise), return_loss=False,
                show_pbar=show_pbar, **kwargs)
        code_list = code_out if isinstance(code_out, list) else [code_out]
        density_grid_list = []
        density_bitfield_list = []
        for step_id, code in enumerate(code_list):
            code = self.code_diff_pr_inv(code)
            n_inverse_steps = self.test_cfg.get('n_inverse_steps', 0)
            if n_inverse_steps > 0 and step_id == (len(code_list) - 1):
                with module_requires_grad(diffusion, False), torch.enable_grad():
                    code_ = self.code_activation.inverse(code).requires_grad_(True)
                    code_optimizer = self.build_optimizer(code_, self.test_cfg)
                    code_scheduler = self.build_scheduler(code_optimizer, self.test_cfg)
                    if show_pbar:
                        pbar = mmcv.ProgressBar(n_inverse_steps)
                    for inverse_step_id in range(n_inverse_steps):
                        code_optimizer.zero_grad()
                        code = self.code_activation(code_)
                        loss, log_vars = diffusion(self.code_diff_pr(code), return_loss=True, cfg=self.test_cfg)
                        loss.backward()
                        code_optimizer.step()
                        if code_scheduler is not None:
                            code_scheduler.step()
                        if show_pbar:
                            pbar.update()
                code = self.code_activation(code_)
            code_list[step_id] = code
            density_grid, density_bitfield = self.get_density(decoder, code, cfg=self.test_cfg)
            density_grid_list.append(density_grid)
            density_bitfield_list.append(density_bitfield)
        if isinstance(code_out, list):
            return code_list, density_grid_list, density_bitfield_list
        else:
            return code_list[-1], density_grid_list[-1], density_bitfield_list[-1]

    def val_guide(self, data, **kwargs):
        device = get_module_device(self)
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder

        cond_imgs = data['cond_imgs']  # (num_scenes, num_imgs, h, w, 3)
        cond_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
        cond_poses = data['cond_poses']

        num_scenes, num_imgs, h, w, _ = cond_imgs.size()
        # (num_scenes, num_imgs, h, w, 3)
        cond_rays_o, cond_rays_d = get_cam_rays(cond_poses, cond_intrinsics, h, w)
        dt_gamma_scale = self.test_cfg.get('dt_gamma_scale', 0.0)
        # (num_scenes,)
        dt_gamma = dt_gamma_scale / cond_intrinsics[..., :2].mean(dim=(-2, -1))

        scene_arange = None
        if self.image_cond:
            concat_cond = cond_imgs.permute(0, 1, 4, 2, 3)  # (num_scenes, num_imgs, 3, h, w)
            if num_imgs > 1:
                cond_inds = torch.stack([torch.randperm(num_imgs, device=device) for _ in range(num_scenes)], dim=0)
                scene_arange = torch.arange(num_scenes, device=device)[:, None]
                concat_cond = concat_cond[scene_arange, cond_inds]  # (num_scenes, num_imgs, 3, h, w)
            diff_image_size = rgetattr(diffusion, 'denoising.image_size')
            assert diff_image_size[0] % concat_cond.size(-2) == 0
            assert diff_image_size[1] % concat_cond.size(-1) == 0
            concat_cond = concat_cond.tile((diff_image_size[0] // concat_cond.size(-2),
                                            diff_image_size[1] // concat_cond.size(-1)))
        else:
            concat_cond = None

        decoder_training_prev = decoder.training
        decoder.train(True)

        with module_requires_grad(diffusion, False), module_requires_grad(decoder, False):
            n_inverse_rays = self.test_cfg.get('n_inverse_rays', 4096)
            num_scene_pixels = num_imgs * h * w
            if num_scene_pixels > n_inverse_rays:
                minibatch_inds = [torch.randperm(num_scene_pixels, device=device) for _ in range(num_scenes)]
                minibatch_inds = torch.stack(minibatch_inds, dim=0).split(n_inverse_rays, dim=1)
                num_minibatch = len(minibatch_inds)
                if scene_arange is None:
                    scene_arange = torch.arange(num_scenes, device=device)[:, None]

            density_grid = torch.zeros((num_scenes, self.grid_size ** 3), device=device)
            density_bitfield = torch.zeros((num_scenes, self.grid_size ** 3 // 8), dtype=torch.uint8, device=device)
            inverse_step_id = torch.zeros((1, ), dtype=torch.long, device=device)

            def grad_guide_fn(x_0_pred):
                code_pred = self.code_diff_pr_inv(x_0_pred)
                self.update_extra_state(decoder, code_pred, density_grid, density_bitfield,
                                        0, density_thresh=self.test_cfg.get('density_thresh', 0.01))
                rays_o = cond_rays_o.reshape(num_scenes, num_scene_pixels, 3)
                rays_d = cond_rays_d.reshape(num_scenes, num_scene_pixels, 3)
                target_rgbs = cond_imgs.reshape(num_scenes, num_scene_pixels, 3)
                if num_scene_pixels > n_inverse_rays:
                    inds = minibatch_inds[inverse_step_id % num_minibatch]  # (num_scenes, n_inverse_rays)
                    rays_o = rays_o[scene_arange, inds]  # (num_scenes, n_inverse_rays, 3)
                    rays_d = rays_d[scene_arange, inds]  # (num_scenes, n_inverse_rays, 3)
                    target_rgbs = target_rgbs[scene_arange, inds]  # (num_scenes, n_inverse_rays, 3)
                _, loss, _ = self.loss(
                    decoder, code_pred, density_bitfield,
                    target_rgbs, rays_o, rays_d, dt_gamma, scale_num_ray=target_rgbs.size(1),
                    cfg=self.test_cfg)
                inverse_step_id[:] += 1
                return loss * num_scenes

            noise = data.get('noise', None)
            if noise is None:
                noise = torch.randn(
                    (num_scenes, *self.code_size), device=get_module_device(self))

            with torch.autocast(
                    device_type='cuda',
                    enabled=self.autocast_dtype is not None,
                    dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
                code = diffusion(
                    self.code_diff_pr(noise), return_loss=False,
                    grad_guide_fn=grad_guide_fn, concat_cond=concat_cond, **kwargs)

        decoder.train(decoder_training_prev)

        return self.code_diff_pr_inv(code), density_grid, density_bitfield

    def val_optim(self, data, code_=None,
                  density_grid=None, density_bitfield=None, show_pbar=False, **kwargs):
        device = get_module_device(self)
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion

        cond_imgs = data['cond_imgs']  # (num_scenes, num_imgs, h, w, 3)
        cond_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
        cond_poses = data['cond_poses']

        num_scenes, num_imgs, h, w, _ = cond_imgs.size()
        # (num_scenes, num_imgs, h, w, 3)
        cond_rays_o, cond_rays_d = get_cam_rays(cond_poses, cond_intrinsics, h, w)
        dt_gamma_scale = self.test_cfg.get('dt_gamma_scale', 0.0)
        # (num_scenes,)
        dt_gamma = dt_gamma_scale / cond_intrinsics[..., :2].mean(dim=(-2, -1))

        if self.image_cond:
            concat_cond = cond_imgs.permute(0, 1, 4, 2, 3)  # (num_scenes, num_imgs, 3, h, w)
            if num_imgs > 1:
                cond_inds = torch.stack([torch.randperm(num_imgs, device=device) for _ in range(num_scenes)], dim=0)
                scene_arange = torch.arange(num_scenes, device=device)[:, None]
                concat_cond = concat_cond[scene_arange, cond_inds]  # (num_scenes, num_imgs, 3, h, w)
            diff_image_size = rgetattr(diffusion, 'denoising.image_size')
            assert diff_image_size[0] % concat_cond.size(-2) == 0
            assert diff_image_size[1] % concat_cond.size(-1) == 0
            concat_cond = concat_cond.tile((diff_image_size[0] // concat_cond.size(-2),
                                            diff_image_size[1] // concat_cond.size(-1)))
        else:
            concat_cond = None

        decoder_training_prev = decoder.training
        decoder.train(True)

        extra_scene_step = self.test_cfg.get('extra_scene_step', 0)
        n_inverse_steps = self.test_cfg.get('n_inverse_steps', 100)
        assert n_inverse_steps > 0
        if show_pbar:
            pbar = mmcv.ProgressBar(n_inverse_steps)

        with module_requires_grad(diffusion, False), module_requires_grad(decoder, False), torch.enable_grad():
            if code_ is None:
                code_ = self.get_init_code_(num_scenes, cond_imgs.device)
            if density_grid is None:
                density_grid = self.get_init_density_grid(num_scenes, cond_imgs.device)
            if density_bitfield is None:
                density_bitfield = self.get_init_density_bitfield(num_scenes, cond_imgs.device)
            code_optimizer = self.build_optimizer(code_, self.test_cfg)
            code_scheduler = self.build_scheduler(code_optimizer, self.test_cfg)

            for inverse_step_id in range(n_inverse_steps):
                code_optimizer.zero_grad()
                code = self.code_activation(code_)
                with torch.autocast(
                        device_type='cuda',
                        enabled=self.autocast_dtype is not None,
                        dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
                    loss, log_vars = diffusion(
                        self.code_diff_pr(code), return_loss=True,
                        concat_cond=concat_cond[:, inverse_step_id % num_imgs] if concat_cond is not None else None,
                        x_t_detach=self.test_cfg.get('x_t_detach', False),
                        cfg=self.test_cfg, **kwargs)
                loss.backward()

                if extra_scene_step > 0:
                    prior_grad = code_.grad.data.clone()
                    cfg = self.test_cfg.copy()
                    cfg['n_inverse_steps'] = extra_scene_step + 1
                    self.inverse_code(
                        decoder, cond_imgs, cond_rays_o, cond_rays_d, dt_gamma=dt_gamma, cfg=cfg,
                        code_=code_,
                        density_grid=density_grid,
                        density_bitfield=density_bitfield,
                        code_optimizer=code_optimizer,
                        code_scheduler=code_scheduler,
                        prior_grad=prior_grad)
                else:  # avoid cloning the grad
                    code = self.code_activation(code_)
                    loss_decoder, log_vars_decoder, out_rgbs, target_rgbs = self.loss_decoder(
                        decoder, code, density_bitfield, cond_rays_o, cond_rays_d,
                        cond_imgs, dt_gamma, cfg=self.test_cfg)
                    loss_decoder.backward()
                    code_optimizer.step()
                    if code_scheduler is not None:
                        code_scheduler.step()

                if show_pbar:
                    pbar.update()

        decoder.train(decoder_training_prev)

        return self.code_activation(code_), density_grid, density_bitfield

    def val_step(self, data, viz_dir=None, viz_dir_guide=None, **kwargs):
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder

        with torch.no_grad():
            if 'code' in data:
                code, density_grid, density_bitfield = self.load_scene(
                    data, load_density=True)
            elif 'cond_imgs' in data:
                cond_mode = self.test_cfg.get('cond_mode', 'guide')
                if cond_mode == 'guide':
                    code, density_grid, density_bitfield = self.val_guide(data, **kwargs)
                elif cond_mode == 'optim':
                    code, density_grid, density_bitfield = self.val_optim(data, **kwargs)
                elif cond_mode == 'guide_optim':
                    code, density_grid, density_bitfield = self.val_guide(data, **kwargs)
                    if viz_dir_guide is not None and 'test_poses' in data:
                        self.eval_and_viz(
                            data, decoder, code, density_bitfield,
                            viz_dir=viz_dir_guide, cfg=self.test_cfg)
                    code, density_grid, density_bitfield = self.val_optim(
                        data,
                        code_=self.code_activation.inverse(code).requires_grad_(True),
                        density_grid=density_grid,
                        density_bitfield=density_bitfield,
                        **kwargs)
                else:
                    raise AttributeError
            else:
                code, density_grid, density_bitfield = self.val_uncond(data, **kwargs)

            # ==== evaluate reconstruction ====
            if 'test_poses' in data:
                log_vars, pred_imgs = self.eval_and_viz(
                    data, decoder, code, density_bitfield,
                    viz_dir=viz_dir, cfg=self.test_cfg)
            else:
                log_vars = dict()
                pred_imgs = None
                if viz_dir is None:
                    viz_dir = self.test_cfg.get('viz_dir', None)
                if viz_dir is not None:
                    if isinstance(decoder, DistributedDataParallel):
                        decoder = decoder.module
                    decoder.visualize(
                        code, data['scene_name'],
                        viz_dir, code_range=self.test_cfg.get('clip_range', [-1, 1]))

        # ==== save 3D code ====
        save_dir = self.test_cfg.get('save_dir', None)
        if save_dir is not None:
            self.save_scene(save_dir, code, density_grid, density_bitfield, data['scene_name'])
            save_mesh = self.test_cfg.get('save_mesh', False)
            if save_mesh:
                mesh_resolution = self.test_cfg.get('mesh_resolution', 256)
                mesh_threshold = self.test_cfg.get('mesh_threshold', 10)
                self.save_mesh(save_dir, decoder, code, data['scene_name'], mesh_resolution, mesh_threshold)

        # ==== outputs ====
        outputs_dict = dict(
            log_vars=log_vars,
            num_samples=len(data['scene_name']),
            pred_imgs=pred_imgs)

        return outputs_dict

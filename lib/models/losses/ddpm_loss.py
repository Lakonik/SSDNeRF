from functools import partial

import torch

from mmgen.models import MODULES
from mmgen.models.losses.ddpm_loss import DDPMLoss, mse_loss, reduce_loss

from ...core import reduce_mean


class DDPMLossMod(DDPMLoss):

    def __init__(self,
                 *args,
                 weight_scale=1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_scale = weight_scale

    def timestep_weight_rescale(self, loss, timesteps, weight):
        return loss * weight.to(timesteps.device)[timesteps] * self.weight_scale

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            assert isinstance(args[0], dict), (
                'You should offer a dictionary containing network outputs '
                'for building up computational graph of this loss module.')
            output_dict = args[0]
        elif 'output_dict' in kwargs:
            assert len(args) == 0, (
                'If the outputs dict is given in keyworded arguments, no'
                ' further non-keyworded arguments should be offered.')
            output_dict = kwargs.pop('outputs_dict')
        else:
            raise NotImplementedError(
                'Cannot parsing your arguments passed to this loss module.'
                ' Please check the usage of this module')

        # check keys in output_dict
        assert 'timesteps' in output_dict, (
            '\'timesteps\' is must for DDPM-based losses, but found'
            f'{output_dict.keys()} in \'output_dict\'')

        timesteps = output_dict['timesteps']
        loss = self._forward_loss(output_dict)

        loss_rescaled = self.rescale_fn(loss, timesteps)

        # update log_vars of this class
        self.collect_log(loss_rescaled, timesteps=timesteps)  # Mod: log after rescaling

        return reduce_loss(loss_rescaled, self.reduction)


@MODULES.register_module()
class DDPMMSELossMod(DDPMLossMod):
    _default_data_info = dict(pred='eps_t_pred', target='noise')

    def __init__(self,
                 rescale_mode=None,
                 rescale_cfg=None,
                 sampler=None,
                 weight=None,
                 weight_scale=1.0,
                 log_cfgs=None,
                 reduction='mean',
                 data_info=None,
                 loss_name='loss_ddpm_mse',
                 scale_norm=False,
                 momentum=0.001):
        super().__init__(rescale_mode=rescale_mode,
                         rescale_cfg=rescale_cfg,
                         log_cfgs=log_cfgs,
                         weight=weight,
                         weight_scale=weight_scale,
                         sampler=sampler,
                         reduction=reduction,
                         loss_name=loss_name)

        self.data_info = self._default_data_info \
            if data_info is None else data_info

        self.loss_fn = partial(mse_loss, reduction='flatmean')
        self.scale_norm = scale_norm
        self.freeze_norm = False
        if scale_norm:
            self.register_buffer('norm_factor', torch.ones(1, dtype=torch.float))
        self.momentum = momentum

    def forward(self, *args, **kwargs):
        loss = super().forward(*args, **kwargs)
        if self.scale_norm:
            if self.training and not self.freeze_norm:
                if len(args) == 1:
                    assert isinstance(args[0], dict), (
                        'You should offer a dictionary containing network outputs '
                        'for building up computational graph of this loss module.')
                    output_dict = args[0]
                elif 'output_dict' in kwargs:
                    assert len(args) == 0, (
                        'If the outputs dict is given in keyworded arguments, no'
                        ' further non-keyworded arguments should be offered.')
                    output_dict = kwargs.pop('outputs_dict')
                else:
                    raise NotImplementedError(
                        'Cannot parsing your arguments passed to this loss module.'
                        ' Please check the usage of this module')
                norm_factor = output_dict['x_0'].detach().square().mean()
                norm_factor = reduce_mean(norm_factor)
                self.norm_factor[:] = (1 - self.momentum) * self.norm_factor \
                                      + self.momentum * norm_factor
            loss = loss / self.norm_factor
        return loss

    def _forward_loss(self, outputs_dict):
        """Forward function for loss calculation.
        Args:
            outputs_dict (dict): Outputs of the model used to calculate losses.

        Returns:
            torch.Tensor: Calculated loss.
        """
        loss_input_dict = {
            k: outputs_dict[v]
            for k, v in self.data_info.items()
        }
        loss = self.loss_fn(**loss_input_dict) * 0.5
        return loss

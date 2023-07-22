import torch
import torch.nn.functional as F
import mmcv

from mmcv.runner import load_checkpoint
from mmgen.models import build_model
from mmgen.models.architectures.common import get_module_device

from lib.runner.hooks.ema_hook import get_ori_key


def init_model(
        config, checkpoint=None, device='cuda:0', cfg_options=None,
        ema_only=True, use_fp16=False):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    model = build_model(
        config.model, train_cfg=config.train_cfg, test_cfg=config.test_cfg)

    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')

    model._cfg = config  # save the config in the model for convenience

    if ema_only:
        module_keys = []
        for hook in config['custom_hooks']:
            if hook['type'] in ('ExponentialMovingAverageHookMod', 'ExponentialMovingAverageHook'):
                if isinstance(hook['module_keys'], str):
                    module_keys.append(hook['module_keys'])
                else:
                    module_keys.extend(hook['module_keys'])
        for key in module_keys:
            ori_key = get_ori_key(key)
            del model._modules[ori_key]

    if use_fp16:
        if hasattr(model, 'diffusion'):
            model.diffusion.half()
        if hasattr(model, 'diffusion_ema'):
            model.diffusion_ema.half()
        if hasattr(model, 'autocast_dtype'):
            model.autocast_dtype = 'float16'

    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def interp_diffusion_nerf_ddim(
        model, test_poses, test_intrinsics, viz_dir=None,
        num_samples=10, batchsize=10, type='linear', **kwargs):
    """
    Args:
        model (nn.Module): DiffusionNeRF model
        num_samples (int, optional): The total number of samples.
            Defaults to 16.
        batchsize (int, optional): The number of batch size for inference.
            Defaults to 4.
    """
    device = get_module_device(model)

    alpha = torch.linspace(0, 1, steps=num_samples)
    alpha = alpha.reshape([-1] + [1] * len(model.code_size))
    noise_ab = torch.randn((2,) + model.code_size)

    if type == 'spherical_linear':
        noise_ab_norm = F.normalize(noise_ab.flatten(1), dim=1)
        theta = torch.acos(noise_ab_norm.prod(dim=0).sum())
        noise = (torch.sin((1 - alpha) * theta) * noise_ab[0]
                 + torch.sin(alpha * theta) * noise_ab[1]) / torch.sin(theta)
    elif type == 'linear':
        noise = (1 - alpha) * noise_ab[0] + alpha * noise_ab[1]
    else:
        raise AttributeError

    noise_batches = noise.split(batchsize, dim=0)

    scene_id_cur = 0
    for noise_batch in noise_batches:
        bs = noise_batch.size(0)
        scene_id = range(scene_id_cur, scene_id_cur + bs)
        data = dict(
            noise=noise_batch.to(device),
            scene_id=scene_id,
            scene_name=['interp_{:02d}'.format(scene_id_single) for scene_id_single in scene_id],
            test_intrinsics=test_intrinsics[None].expand(bs, -1, -1).to(device),
            test_poses=test_poses[None].expand(bs, -1, -1, -1).to(device))
        model.val_step(data, viz_dir=viz_dir, show_pbar=True, **kwargs)
        scene_id_cur += bs

    return

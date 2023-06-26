import torch
import torch.nn.functional as F

from mmgen.models.architectures.common import get_module_device


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

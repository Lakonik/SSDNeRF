import numpy as np
import torch
import torch.nn.functional as F

import mcubes
from packaging import version as pver


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def get_ray_directions(h, w, intrinsics, norm=False, device=None):
    """
    Args:
        h (int)
        w (int)
        intrinsics: (*, 4), in [fx, fy, cx, cy]

    Returns:
        directions: (*, h, w, 3), the direction of the rays in camera coordinate
    """
    batch_size = intrinsics.shape[:-1]
    x = torch.linspace(0.5, w - 0.5, w, device=device)
    y = torch.linspace(0.5, h - 0.5, h, device=device)
    # (*, h, w, 2)
    directions_xy = torch.stack(
        [((x - intrinsics[..., 2:3]) / intrinsics[..., 0:1])[..., None, :].expand(*batch_size, h, w),
         ((y - intrinsics[..., 3:4]) / intrinsics[..., 1:2])[..., :, None].expand(*batch_size, h, w)], dim=-1)
    # (*, h, w, 3)
    directions = F.pad(directions_xy, [0, 1], mode='constant', value=1.0)
    if norm:
        directions = F.normalize(directions, dim=-1)
    return directions


def get_rays(directions, c2w, norm=False):
    """
    Args:
        directions: (*, h, w, 3) precomputed ray directions in camera coordinate
        c2w: (*, 3, 4) transformation matrix from camera coordinate to world coordinate
    Returns:
        rays_o: (*, h, w, 3), the origin of the rays in world coordinate
        rays_d: (*, h, w, 3), the normalized direction of the rays in world coordinate
    """
    rays_d = directions @ c2w[..., None, :3, :3].transpose(-1, -2)  # (*, h, w, 3)
    rays_o = c2w[..., None, None, :3, 3].expand(rays_d.shape)  # (*, h, w, 3)
    if norm:
        rays_d = F.normalize(rays_d, dim=-1)
    return rays_o, rays_d


def get_cam_rays(c2w, intrinsics, h, w):
    directions = get_ray_directions(
        h, w, intrinsics, norm=False, device=intrinsics.device)  # (num_scenes, num_imgs, h, w, 3)
    rays_o, rays_d = get_rays(directions, c2w, norm=True)
    return rays_o, rays_d


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys),
                                                  len(zs)).detach().cpu().numpy()  # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def _extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    # print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    # print(u.shape, u.max(), u.min(), np.percentile(u, 50))

    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def extract_geometry(decoder, code_single, resolution=256, threshold=10):

    def query_func(pts):
        with torch.no_grad():
            pts = pts.to(code_single.device)[None]
            sigma = decoder.point_density_decode(
                pts,
                code_single[None])[0].flatten()
            out_mask = (pts.squeeze(0) < decoder.aabb[:3]).any(dim=-1) | (pts.squeeze(0) > decoder.aabb[3:]).any(dim=-1)
            sigma.masked_fill_(out_mask, 0)
        return sigma

    vertices, triangles = _extract_geometry(
        decoder.aabb[:3] - 0.1, decoder.aabb[3:] + 0.1,
        resolution=resolution, threshold=threshold, query_func=query_func)
    return vertices, triangles

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


def get_ray_directions(h, w, focal, center, norm=False, device=None, with_radius=False):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    x = torch.linspace(0.5, w - 0.5, w, device=device)
    y = torch.linspace(0.5, h - 0.5, h, device=device)
    # (H, W, 2)
    directions_xy = torch.stack([((x - center[0]) / focal[0])[None, :].expand(h, w),
                                 ((y - center[1]) / focal[1])[:, None].expand(h, w)], dim=-1)
    # (H, W, 3)
    directions = F.pad(directions_xy, [0, 1], mode='constant', value=1.0)
    if with_radius:
        base_radius = ((focal[0] * focal[1]) ** -0.5) / 2
        radii = base_radius * (directions.norm(dim=-1) ** -1.5)
    else:
        radii = None
    if norm:
        directions = F.normalize(directions, dim=-1)
    return directions, radii


def get_rays(directions, c2w, norm=False):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (n, 3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (n, H, W, 3), the origin of the rays in world coordinate
        rays_d: (n, H, W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[..., None, :3, :3].transpose(-1, -2)  # (n, H, W, 3)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[..., None, None, :3, 3].expand(rays_d.shape)  # (H, W, 3)
    if norm:
        rays_d = F.normalize(rays_d, dim=-1)
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

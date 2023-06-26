import torch
import torch.nn as nn

from mmgen.models.builder import build_module

from lib.ops import (
    batch_near_far_from_aabb,
    march_rays_train, batch_composite_rays_train, march_rays, composite_rays)


class VolumeRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 min_near=0.2,
                 bg_radius=-1,
                 max_steps=256,
                 decoder_reg_loss=None,
                 ):
        super().__init__()

        self.bound = bound
        self.min_near = min_near
        self.bg_radius = bg_radius  # radius of the background sphere.
        self.max_steps = max_steps
        self.decoder_reg_loss = build_module(decoder_reg_loss) if decoder_reg_loss is not None else None

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        aabb = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        self.register_buffer('aabb', aabb)

    def point_decode(self, xyzs, dirs, code):
        raise NotImplementedError

    def point_density_decode(self, xyzs, code):
        raise NotImplementedError

    def loss(self):
        assert self.decoder_reg_loss is None
        return None

    def forward(self, rays_o, rays_d, code, density_bitfield, grid_size,
                dt_gamma=0, perturb=False, T_thresh=1e-4, return_loss=False):
        """
        Args:
            rays_o: Shape (num_scenes, num_rays_per_scene, 3)
            rays_d: Shape (num_scenes, num_rays_per_scene, 3)
            code: Shape (num_scenes, *code_size)
            density_bitfield: Shape (num_scenes, griz_size**3 // 8)
        """
        num_scenes = len(rays_o)
        assert num_scenes > 0
        if isinstance(grid_size, int):
            grid_size = [grid_size] * num_scenes
        if isinstance(dt_gamma, float):
            dt_gamma = [dt_gamma] * num_scenes

        nears, fars = batch_near_far_from_aabb(rays_o, rays_d, self.aabb, self.min_near)

        if self.training:
            xyzs = []
            dirs = []
            deltas = []
            rays = []
            for (rays_o_single, rays_d_single, density_bitfield_single,
                 nears_single, fars_single, grid_size_single, dt_gamma_single) in zip(
                    rays_o, rays_d, density_bitfield, nears, fars, grid_size, dt_gamma):
                xyzs_single, dirs_single, deltas_single, rays_single = march_rays_train(
                    rays_o_single, rays_d_single, self.bound, density_bitfield_single,
                    1, grid_size_single, nears_single, fars_single,
                    perturb=perturb, align=128, force_all_rays=True,
                    dt_gamma=dt_gamma_single.item(), max_steps=self.max_steps)
                xyzs.append(xyzs_single)
                dirs.append(dirs_single)
                deltas.append(deltas_single)
                rays.append(rays_single)
            sigmas, rgbs, num_points = self.point_decode(xyzs, dirs, code)
            weights_sum, depth, image = batch_composite_rays_train(sigmas, rgbs, deltas, rays, num_points, T_thresh)

        else:
            device = rays_o.device
            dtype = torch.float32

            weights_sum = []
            depth = []
            image = []

            for (rays_o_single, rays_d_single,
                 code_single, density_bitfield_single,
                 nears_single, fars_single,
                 grid_size_single, dt_gamma_single) in zip(
                    rays_o, rays_d, code, density_bitfield, nears, fars, grid_size, dt_gamma):
                num_rays_per_scene = rays_o_single.size(0)

                weights_sum_single = torch.zeros(num_rays_per_scene, dtype=dtype, device=device)
                depth_single = torch.zeros(num_rays_per_scene, dtype=dtype, device=device)
                image_single = torch.zeros(num_rays_per_scene, 3, dtype=dtype, device=device)

                num_rays_alive = num_rays_per_scene
                rays_alive = torch.arange(num_rays_alive, dtype=torch.int32, device=device)  # (num_rays_alive,)
                rays_t = nears_single.clone()  # (num_rays_alive,)

                step = 0
                while step < self.max_steps:
                    # count alive rays
                    num_rays_alive = rays_alive.size(0)
                    # exit loop
                    if num_rays_alive == 0:
                        break
                    # decide compact_steps
                    n_step = min(max(num_rays_per_scene // num_rays_alive, 1), 8)
                    xyzs, dirs, deltas = march_rays(
                        num_rays_alive, n_step, rays_alive, rays_t, rays_o_single, rays_d_single,
                        self.bound, density_bitfield_single, 1, grid_size_single, nears_single, fars_single,
                        align=128, perturb=perturb, dt_gamma=dt_gamma_single.item(), max_steps=self.max_steps)
                    sigmas, rgbs, _ = self.point_decode([xyzs], [dirs], code_single[None])
                    composite_rays(num_rays_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas,
                                   weights_sum_single, depth_single, image_single, T_thresh)
                    rays_alive = rays_alive[rays_alive >= 0]
                    step += n_step

                weights_sum.append(weights_sum_single)
                depth.append(depth_single)
                image.append(image_single)

        results = dict(
            weights_sum=weights_sum,
            depth=depth,
            image=image)

        if return_loss:
            results.update(decoder_reg_loss=self.loss())

        return results

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import mmcv
import trimesh

from copy import deepcopy
from glob import glob
from torch.nn.parallel.distributed import DistributedDataParallel
from mmcv.runner import load_checkpoint
from mmgen.models.builder import MODULES, build_module
from mmgen.models.architectures.common import get_module_device

from ...core import custom_meshgrid, eval_psnr, eval_ssim_skimage, reduce_mean, rgetattr, rsetattr, extract_geometry, \
    module_requires_grad
from lib.ops import morton3D, morton3D_invert, packbits

LPIPS_BS = 32


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


@MODULES.register_module()
class TanhCode(nn.Module):
    def __init__(self, scale=1.0):
        super(TanhCode, self).__init__()
        self.scale = scale

    def forward(self, code_, update_stats=False):
        return code_.tanh() if self.scale == 1 else code_.tanh() * self.scale

    def inverse(self, code):
        return code.atanh() if self.scale == 1 else (code / self.scale).atanh()


@MODULES.register_module()
class IdentityCode(nn.Module):
    @staticmethod
    def forward(code_, update_stats=False):
        return code_

    @staticmethod
    def inverse(code):
        return code


@MODULES.register_module()
class NormalizedTanhCode(nn.Module):
    def __init__(self, mean=0.0, std=1.0, clip_range=1, eps=1e-5, momentum=0.001):
        super(NormalizedTanhCode, self).__init__()
        self.mean = mean
        self.std = std
        self.clip_range = clip_range
        self.register_buffer('running_mean', torch.tensor([0.0]))
        self.register_buffer('running_var', torch.tensor([std ** 2]))
        self.momentum = momentum
        self.eps = eps

    def forward(self, code_, update_stats=False):
        if update_stats and self.training:
            with torch.no_grad():
                var, mean = torch.var_mean(code_)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * reduce_mean(mean))
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * reduce_mean(var))
        scale = (self.std / (self.running_var.sqrt() + self.eps)).to(code_.device)
        return (code_ * scale + (self.mean - self.running_mean.to(code_.device) * scale)
                ).div(self.clip_range).tanh().mul(self.clip_range)

    def inverse(self, code):
        scale = ((self.running_var.sqrt() + self.eps) / self.std).to(code.device)
        return code.div(self.clip_range).atanh().mul(self.clip_range) \
            * scale + (self.running_mean.to(code.device) - self.mean * scale)


class BaseNeRF(nn.Module):
    def __init__(self,
                 code_size=(3, 8, 64, 64),
                 code_activation=dict(
                     type='TanhCode',
                     scale=1),
                 grid_size=64,
                 decoder=dict(
                     type='TriPlaneDecoder'),
                 decoder_use_ema=False,
                 bg_color=1,
                 rc_loss=dict(
                     type='MSELoss'),
                 reg_loss=None,
                 update_extra_interval=16,
                 use_lpips_metric=True,
                 init_from_mean=False,
                 init_scale=1e-4,
                 mean_ema_momentum=0.001,
                 mean_scale=1.0,
                 train_cfg=dict(),
                 test_cfg=dict(),
                 pretrained=None):
        super().__init__()
        self.code_size = code_size
        self.code_activation = build_module(code_activation)
        self.grid_size = grid_size
        self.decoder = build_module(decoder)
        self.decoder_use_ema = decoder_use_ema
        if self.decoder_use_ema:
            self.decoder_ema = deepcopy(self.decoder)
        self.bg_color = bg_color
        self.rc_loss = build_module(rc_loss)
        self.reg_loss = build_module(reg_loss) if reg_loss is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.update_extra_interval = update_extra_interval
        self.lpips = [] if use_lpips_metric else None
        if init_from_mean:
            self.register_buffer('init_code', torch.zeros(code_size))
        else:
            self.init_code = None
        self.init_scale = init_scale
        self.mean_ema_momentum = mean_ema_momentum
        self.mean_scale = mean_scale
        if pretrained is not None and os.path.isfile(pretrained):
            load_checkpoint(self, pretrained, map_location='cpu')

        self.train_cfg_backup = dict()
        for key, value in self.test_cfg.get('override_cfg', dict()).items():
            self.train_cfg_backup[key] = rgetattr(self, key, None)

    def train(self, mode=True):
        if mode:
            for key, value in self.train_cfg_backup.items():
                rsetattr(self, key, value)
        else:
            for key, value in self.test_cfg.get('override_cfg', dict()).items():
                if self.training:
                    self.train_cfg_backup[key] = rgetattr(self, key)
                rsetattr(self, key, value)
        super().train(mode)
        return self

    def load_scene(self, data, load_density=False):
        device = get_module_device(self)
        code_list = []
        density_grid = []
        density_bitfield = []
        for code_state_single in data['code']:
            code_list.append(
                code_state_single['param']['code'] if 'code' in code_state_single['param']
                else self.code_activation(code_state_single['param']['code_']))
            if load_density:
                density_grid.append(code_state_single['param']['density_grid'])
                density_bitfield.append(code_state_single['param']['density_bitfield'])
        code = torch.stack(code_list, dim=0).to(device)
        density_grid = torch.stack(density_grid, dim=0).to(device) if load_density else None
        density_bitfield = torch.stack(density_bitfield, dim=0).to(device) if load_density else None
        return code, density_grid, density_bitfield

    @staticmethod
    def save_scene(save_dir, code, density_grid, density_bitfield, scene_name):
        os.makedirs(save_dir, exist_ok=True)
        for scene_id, scene_name_single in enumerate(scene_name):
            results = dict(
                scene_name=scene_name_single,
                param=dict(
                    code=code.data[scene_id].cpu(),
                    density_grid=density_grid.data[scene_id].cpu(),
                    density_bitfield=density_bitfield.data[scene_id].cpu()))
            torch.save(results, os.path.join(save_dir, scene_name_single) + '.pth')

    @staticmethod
    def save_mesh(save_dir, decoder, code, scene_name, mesh_resolution, mesh_threshold):
        os.makedirs(save_dir, exist_ok=True)
        for code_single, scene_name_single in zip(code, scene_name):
            vertices, triangles = extract_geometry(
                decoder,
                code_single,
                mesh_resolution,
                mesh_threshold)
            mesh = trimesh.Trimesh(vertices, triangles, process=False)
            mesh.export(os.path.join(save_dir, scene_name_single) + '.stl')

    def get_init_code_(self, num_scenes, device=None):
        code_ = torch.empty(
            self.code_size if num_scenes is None else (num_scenes, *self.code_size),
            device=device, requires_grad=True, dtype=torch.float32)
        if self.init_code is None:
            code_.data.uniform_(-self.init_scale, self.init_scale)
        else:
            code_.data[:] = self.code_activation.inverse(self.init_code * self.mean_scale)
        return code_

    def get_init_density_grid(self, num_scenes, device=None):
        return torch.zeros(
            self.grid_size ** 3 if num_scenes is None else (num_scenes, self.grid_size ** 3),
            device=device, dtype=torch.float16)

    def get_init_density_bitfield(self, num_scenes, device=None):
        return torch.zeros(
            self.grid_size ** 3 // 8 if num_scenes is None else (num_scenes, self.grid_size ** 3 // 8),
            device=device, dtype=torch.uint8)

    @staticmethod
    def build_optimizer(code_, cfg):
        optimizer_cfg = cfg['optimizer'].copy()
        optimizer_class = getattr(torch.optim, optimizer_cfg.pop('type'))
        if isinstance(code_, list):
            code_optimizer = [
                optimizer_class([code_single_], **optimizer_cfg)
                for code_single_ in code_]
        else:
            code_optimizer = optimizer_class([code_], **optimizer_cfg)
        return code_optimizer

    @staticmethod
    def build_scheduler(code_optimizer, cfg):
        if 'lr_scheduler' in cfg:
            scheduler_cfg = cfg['lr_scheduler'].copy()
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_cfg.pop('type'))
            if isinstance(code_optimizer, list):
                code_scheduler = [
                    scheduler_class(code_optimizer_single, **scheduler_cfg)
                    for code_optimizer_single in code_optimizer]
            else:
                code_scheduler = scheduler_class(code_optimizer, **scheduler_cfg)
        else:
            code_scheduler = None
        return code_scheduler

    def loss(self, decoder, code, density_bitfield, target_rgbs,
             rays_o, rays_d, dt_gamma=0.0, return_decoder_loss=False, scale_num_ray=1.0,
             rc_only=False, cfg=dict(), **kwargs):
        outputs = decoder(
            rays_o, rays_d, code, density_bitfield, self.grid_size,
            dt_gamma=dt_gamma, perturb=True, return_loss=return_decoder_loss)
        out_weights = outputs['weights_sum']
        out_rgbs = outputs['image'] + self.bg_color * (1 - out_weights.unsqueeze(-1))
        scale = 1 - math.exp(-cfg['loss_coef'] * scale_num_ray) if 'loss_coef' in cfg else 1
        rc_loss = self.rc_loss(out_rgbs, target_rgbs, **kwargs) * (scale * 3)
        loss = rc_loss
        loss_dict = dict(rc_loss=rc_loss)
        if not rc_only:
            if self.reg_loss is not None:
                reg_loss = self.reg_loss(code, **kwargs)
                loss = loss + reg_loss
                loss_dict.update(reg_loss=reg_loss)
            if return_decoder_loss and outputs['decoder_reg_loss'] is not None:
                decoder_reg_loss = outputs['decoder_reg_loss']
                loss = loss + decoder_reg_loss
                loss_dict.update(decoder_reg_loss=decoder_reg_loss)
        return out_rgbs, loss, loss_dict

    def loss_decoder(self, decoder, code, density_bitfield, cond_rays_o, cond_rays_d,
                     cond_imgs, dt_gamma=0.0, cfg=dict(), **kwargs):
        device = code.device
        decoder_training_prev = decoder.training
        decoder.train(True)
        num_scenes, num_imgs, h, w, _ = cond_rays_o.size()
        n_decoder_rays = cfg.get('n_decoder_rays', 4096)

        num_scene_pixels = num_imgs * h * w

        rays_o = cond_rays_o.reshape(num_scenes, num_scene_pixels, 3)
        rays_d = cond_rays_d.reshape(num_scenes, num_scene_pixels, 3)
        target_rgbs = cond_imgs.reshape(num_scenes, num_scene_pixels, 3)

        if num_scene_pixels > n_decoder_rays:
            inds = [torch.randperm(num_scene_pixels, device=device)[:n_decoder_rays] for _ in range(num_scenes)]
            inds = torch.stack(inds, dim=0)
            scene_arange = torch.arange(num_scenes, device=device)[:, None]
            rays_o = rays_o[scene_arange, inds]  # (num_scenes, n_decoder_rays, 3)
            rays_d = rays_d[scene_arange, inds]  # (num_scenes, n_decoder_rays, 3)
            target_rgbs = target_rgbs[scene_arange, inds]  # (num_scenes, n_decoder_rays, 3)

        out_rgbs, loss, loss_dict = self.loss(
            decoder, code, density_bitfield, target_rgbs,
            rays_o, rays_d, dt_gamma, return_decoder_loss=True, scale_num_ray=num_scene_pixels,
            cfg=cfg, **kwargs)
        log_vars = dict()
        for key, val in loss_dict.items():
            log_vars.update({key: float(val)})

        decoder.train(decoder_training_prev)

        return loss, log_vars, out_rgbs, target_rgbs

    def update_extra_state(self, decoder, code, density_grid, density_bitfield,
                           iter_density, density_thresh=0.01, decay=0.9, S=128):
        with torch.no_grad():
            device = get_module_device(self)
            num_scenes = density_grid.size(0)
            tmp_grid = torch.full_like(density_grid, -1)
            if isinstance(decoder, DistributedDataParallel):
                decoder = decoder.module

            # full update.
            if iter_density < 16:
                X = torch.arange(self.grid_size, dtype=torch.int32, device=device).split(S)
                Y = torch.arange(self.grid_size, dtype=torch.int32, device=device).split(S)
                Z = torch.arange(self.grid_size, dtype=torch.int32, device=device).split(S)

                for xs in X:
                    for ys in Y:
                        for zs in Z:
                            # construct points
                            xx, yy, zz = custom_meshgrid(xs, ys, zs)
                            coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                               dim=-1)  # [N, 3], in [0, 128)
                            indices = morton3D(coords).long()  # [N]
                            xyzs = (coords.float() - (self.grid_size - 1) / 2) * (2 * decoder.bound / self.grid_size)
                            # add noise
                            half_voxel_width = decoder.bound / self.grid_size
                            xyzs += torch.rand_like(xyzs) * (2 * half_voxel_width) - half_voxel_width
                            # query density
                            sigmas = decoder.point_density_decode(
                                xyzs[None].expand(num_scenes, -1, 3), code)[0].reshape(num_scenes, -1)  # (num_scenes, N)
                            # assign
                            tmp_grid[:, indices] = sigmas.clamp(
                                max=torch.finfo(tmp_grid.dtype).max).to(tmp_grid.dtype)

            # partial update (half the computation)
            else:
                N = self.grid_size ** 3 // 4  # H * H * H / 4
                # random sample some positions
                coords = torch.randint(0, self.grid_size, (N, 3), device=device)  # [N, 3], in [0, 128)
                indices = morton3D(coords).long()  # [N]
                # random sample occupied positions
                occ_indices_all = []
                for scene_id in range(num_scenes):
                    occ_indices = torch.nonzero(density_grid[scene_id] > 0).squeeze(-1)  # [Nz]
                    rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long,
                                              device=device)
                    occ_indices_all.append(occ_indices[rand_mask])  # [Nz] --> [N], allow for duplication
                occ_indices_all = torch.stack(occ_indices_all, dim=0)
                occ_coords_all = morton3D_invert(occ_indices_all.flatten()).reshape(num_scenes, N, 3)
                indices = torch.cat([indices[None].expand(num_scenes, N), occ_indices_all], dim=0)
                coords = torch.cat([coords[None].expand(num_scenes, N, 3), occ_coords_all], dim=0)
                # same below
                xyzs = (coords.float() - (self.grid_size - 1) / 2) * (2 * decoder.bound / self.grid_size)
                half_voxel_width = decoder.bound / self.grid_size
                xyzs += torch.rand_like(xyzs) * (2 * half_voxel_width) - half_voxel_width
                sigmas = decoder.point_density_decode(xyzs, code)[0].reshape(num_scenes, -1)  # (num_scenes, N + N)
                # assign
                tmp_grid[torch.arange(num_scenes, device=device)[:, None], indices] = sigmas.clamp(
                    max=torch.finfo(tmp_grid.dtype).max).to(tmp_grid.dtype)

            # ema update
            valid_mask = (density_grid >= 0) & (tmp_grid >= 0)
            density_grid[:] = torch.where(valid_mask, torch.maximum(density_grid * decay, tmp_grid), density_grid)
            # density_grid[valid_mask] = torch.maximum(density_grid[valid_mask] * decay, tmp_grid[valid_mask])
            mean_density = torch.mean(density_grid.clamp(min=0))  # -1 regions are viewed as 0 density.
            iter_density += 1

            # convert to bitfield
            density_thresh = min(mean_density, density_thresh)
            packbits(density_grid, density_thresh, density_bitfield)

        return

    def get_density(self, decoder, code, cfg=dict()):
        density_thresh = cfg.get('density_thresh', 0.01)
        density_step = cfg.get('density_step', 8)
        num_scenes = code.size(0)
        device = code.device
        density_grid = self.get_init_density_grid(num_scenes, device)
        density_bitfield = self.get_init_density_bitfield(num_scenes, device)
        for i in range(density_step):
            self.update_extra_state(decoder, code, density_grid, density_bitfield, i,
                                    density_thresh=density_thresh, decay=1.0)
        return density_grid, density_bitfield

    def inverse_code(self, decoder, cond_imgs, cond_rays_o, cond_rays_d, dt_gamma=0, cfg=dict(),
                     code_=None, density_grid=None, density_bitfield=None, iter_density=None,
                     code_optimizer=None, code_scheduler=None,
                     prior_grad=None, show_pbar=False):
        """
        Obtain scene codes via optimization-based inverse rendering.
        """
        device = get_module_device(self)
        decoder_training_prev = decoder.training
        decoder.train(True)

        with module_requires_grad(decoder, False):
            n_inverse_steps = cfg.get('n_inverse_steps', 1000)
            n_inverse_rays = cfg.get('n_inverse_rays', 4096)

            num_scenes, num_imgs, h, w, _ = cond_imgs.size()
            num_scene_pixels = num_imgs * h * w
            if num_scene_pixels > n_inverse_rays:
                minibatch_inds = [torch.randperm(num_scene_pixels, device=device) for _ in range(num_scenes)]
                minibatch_inds = torch.stack(minibatch_inds, dim=0).split(n_inverse_rays, dim=1)
                num_minibatch = len(minibatch_inds)
                scene_arange = torch.arange(num_scenes, device=device)[:, None]

            if code_ is None:
                code_ = self.get_init_code_(num_scenes, device=device)
            if density_grid is None:
                density_grid = self.get_init_density_grid(num_scenes, device)
            if density_bitfield is None:
                density_bitfield = self.get_init_density_bitfield(num_scenes, device)
            if iter_density is None:
                iter_density = 0

            if code_optimizer is None:
                assert code_scheduler is None
                code_optimizer = self.build_optimizer(code_, cfg)
            if code_scheduler is None:
                code_scheduler = self.build_scheduler(code_optimizer, cfg)

            assert n_inverse_steps > 0
            if show_pbar:
                pbar = mmcv.ProgressBar(n_inverse_steps)

            for inverse_step_id in range(n_inverse_steps):
                code = self.code_activation(
                    torch.stack(code_, dim=0) if isinstance(code_, list)
                    else code_)

                if inverse_step_id % self.update_extra_interval == 0:
                    self.update_extra_state(decoder, code, density_grid, density_bitfield,
                                            iter_density, density_thresh=cfg.get('density_thresh', 0.01))
                rays_o = cond_rays_o.reshape(num_scenes, num_scene_pixels, 3)
                rays_d = cond_rays_d.reshape(num_scenes, num_scene_pixels, 3)
                target_rgbs = cond_imgs.reshape(num_scenes, num_scene_pixels, 3)
                if num_scene_pixels > n_inverse_rays:
                    inds = minibatch_inds[inverse_step_id % num_minibatch]  # (num_scenes, n_inverse_rays)
                    rays_o = rays_o[scene_arange, inds]  # (num_scenes, n_inverse_rays, 3)
                    rays_d = rays_d[scene_arange, inds]  # (num_scenes, n_inverse_rays, 3)
                    target_rgbs = target_rgbs[scene_arange, inds]  # (num_scenes, n_inverse_rays, 3)

                out_rgbs, loss, loss_dict = self.loss(
                    decoder, code, density_bitfield,
                    target_rgbs, rays_o, rays_d, dt_gamma, scale_num_ray=num_scene_pixels,
                    cfg=cfg)

                if prior_grad is not None:
                    if isinstance(code_, list):
                        for code_single_, prior_grad_single in zip(code_, prior_grad):
                            code_single_.grad.copy_(prior_grad_single)
                    else:
                        code_.grad.copy_(prior_grad)
                else:
                    if isinstance(code_optimizer, list):
                        for code_optimizer_single in code_optimizer:
                            code_optimizer_single.zero_grad()
                    else:
                        code_optimizer.zero_grad()

                loss.backward()

                if isinstance(code_optimizer, list):
                    for code_optimizer_single in code_optimizer:
                        code_optimizer_single.step()
                else:
                    code_optimizer.step()

                if code_scheduler is not None:
                    if isinstance(code_scheduler, list):
                        for code_scheduler_single in code_scheduler:
                            code_scheduler_single.step()
                    else:
                        code_scheduler.step()

                if show_pbar:
                    pbar.update()

        decoder.train(decoder_training_prev)

        return code.detach(), density_grid, density_bitfield, \
               loss, loss_dict, out_rgbs, target_rgbs

    def render(self, decoder, code, density_bitfield, h, w, intrinsics, poses, cfg=dict()):
        decoder_training_prev = decoder.training
        decoder.train(False)

        dt_gamma_scale = cfg.get('dt_gamma_scale', 0.0)
        # (num_scenes,)
        dt_gamma = dt_gamma_scale * 2 / (intrinsics[..., 0] + intrinsics[..., 1]).mean(dim=-1)
        rays_o, rays_d = get_cam_rays(poses, intrinsics, h, w)
        num_scenes, num_imgs, h, w, _ = rays_o.size()

        rays_o = rays_o.reshape(num_scenes, num_imgs * h * w, 3)
        rays_d = rays_d.reshape(num_scenes, num_imgs * h * w, 3)
        max_render_rays = cfg.get('max_render_rays', -1)
        if 0 < max_render_rays < rays_o.size(1):
            rays_o = rays_o.split(max_render_rays, dim=1)
            rays_d = rays_d.split(max_render_rays, dim=1)
        else:
            rays_o = [rays_o]
            rays_d = [rays_d]

        out_image = []
        out_depth = []
        for rays_o_single, rays_d_single in zip(rays_o, rays_d):
            outputs = decoder(
                rays_o_single, rays_d_single,
                code, density_bitfield, self.grid_size,
                dt_gamma=dt_gamma, perturb=False)
            weights = torch.stack(outputs['weights_sum'], dim=0) if num_scenes > 1 else outputs['weights_sum'][0]
            rgbs = (torch.stack(outputs['image'], dim=0) if num_scenes > 1 else outputs['image'][0]) \
                   + self.bg_color * (1 - weights.unsqueeze(-1))
            depth = torch.stack(outputs['depth'], dim=0) if num_scenes > 1 else outputs['depth'][0]
            out_image.append(rgbs)
            out_depth.append(depth)
        out_image = torch.cat(out_image, dim=1) if len(out_image) > 1 else out_image[0]
        out_depth = torch.cat(out_depth, dim=1) if len(out_depth) > 1 else out_depth[0]
        out_image = out_image.reshape(num_scenes, num_imgs, h, w, 3)
        out_depth = out_depth.reshape(num_scenes, num_imgs, h, w)

        decoder.train(decoder_training_prev)
        return out_image, out_depth

    def eval_and_viz(self, data, decoder, code, density_bitfield, viz_dir=None, cfg=dict()):
        scene_name = data['scene_name']  # (num_scenes,)
        test_intrinsics = data['test_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
        test_poses = data['test_poses']
        num_scenes, num_imgs, _, _ = test_poses.size()

        if 'test_imgs' in data and not cfg.get('skip_eval', False):
            test_imgs = data['test_imgs']  # (num_scenes, num_imgs, h, w, 3)
            _, _, h, w, _ = test_imgs.size()
            test_img_paths = data['test_img_paths']  # (num_scenes, (num_imgs,))
            target_imgs = test_imgs.permute(0, 1, 4, 2, 3).reshape(num_scenes * num_imgs, 3, h, w)
        else:
            test_imgs = test_img_paths = target_imgs = None
            h, w = cfg['img_size']
        image, depth = self.render(
            decoder, code, density_bitfield, h, w, test_intrinsics, test_poses, cfg=cfg)
        pred_imgs = image.permute(0, 1, 4, 2, 3).reshape(
            num_scenes * num_imgs, 3, h, w).clamp(min=0, max=1)
        pred_imgs = torch.round(pred_imgs * 255) / 255

        if test_imgs is not None:
            test_psnr = eval_psnr(pred_imgs, target_imgs)
            test_ssim = eval_ssim_skimage(pred_imgs, target_imgs, data_range=1)
            log_vars = dict(test_psnr=float(test_psnr.mean()),
                            test_ssim=float(test_ssim.mean()))
            if self.lpips is not None:
                if len(self.lpips) == 0:
                    lpips_eval = lpips.LPIPS(net='vgg').to(pred_imgs.device)
                    lpips_eval.eval()
                    self.lpips.append(lpips_eval)
                test_lpips = []
                for pred_imgs_batch, target_imgs_batch in zip(
                        pred_imgs.split(LPIPS_BS, dim=0), target_imgs.split(LPIPS_BS, dim=0)):
                    test_lpips.append(self.lpips[0](pred_imgs_batch * 2 - 1, target_imgs_batch * 2 - 1).flatten())
                test_lpips = torch.cat(test_lpips, dim=0)
                log_vars.update(test_lpips=float(test_lpips.mean()))
            else:
                test_lpips = [math.nan for _ in range(num_scenes * num_imgs)]
        else:
            log_vars = dict()

        if viz_dir is None:
            viz_dir = cfg.get('viz_dir', None)
        if viz_dir is not None:
            os.makedirs(viz_dir, exist_ok=True)
            output_viz = torch.round(pred_imgs.permute(0, 2, 3, 1) * 255).to(
                torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 3)
            if test_imgs is not None:
                real_imgs_viz = (target_imgs.permute(0, 2, 3, 1) * 255).to(
                    torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 3)
                output_viz = np.concatenate([real_imgs_viz, output_viz], axis=-2)
            for scene_id, scene_name_single in enumerate(scene_name):
                for img_id in range(num_imgs):
                    if test_img_paths is not None:
                        base_name = 'scene_' + scene_name_single + '_' + os.path.splitext(
                            os.path.basename(test_img_paths[scene_id][img_id]))[0]
                        name = base_name + '_psnr{:02.1f}_ssim{:.2f}_lpips{:.3f}.png'.format(
                            test_psnr[scene_id * num_imgs + img_id],
                            test_ssim[scene_id * num_imgs + img_id],
                            test_lpips[scene_id * num_imgs + img_id])
                        existing_files = glob(os.path.join(viz_dir, base_name + '*.png'))
                        for file in existing_files:
                            os.remove(file)
                    else:
                        name = 'scene_' + scene_name_single + '_{:03d}.png'.format(img_id)
                    plt.imsave(
                        os.path.join(viz_dir, name),
                        output_viz[scene_id][img_id])
            if isinstance(decoder, DistributedDataParallel):
                decoder = decoder.module
            code_range = cfg.get('clip_range', [-1, 1])
            decoder.visualize(code, scene_name, viz_dir, code_range=code_range)
            if self.init_code is not None:
                decoder.visualize(self.init_code[None], ['000_mean'], viz_dir, code_range=code_range)

        return log_vars, pred_imgs.reshape(num_scenes, num_imgs, 3, h, w)

    def mean_ema_update(self, code):
        if self.init_code is None:
            return
        mean_code = reduce_mean(code.detach().mean(dim=0))
        self.init_code.mul_(1 - self.mean_ema_momentum).add_(
            mean_code.data, alpha=self.mean_ema_momentum)

    def train_step(self, data, optimizer, running_status=None):
        raise NotImplementedError

    def val_step(self, data, viz_dir=None, show_pbar=False, **kwargs):
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder

        if 'code' in data:
            code, density_grid, density_bitfield = self.load_scene(
                data, load_density=True)
            out_rgbs = target_rgbs = None
        else:
            cond_imgs = data['cond_imgs']  # (num_scenes, num_imgs, h, w, 3)
            cond_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            cond_poses = data['cond_poses']

            num_scenes, num_imgs, h, w, _ = cond_imgs.size()
            # (num_scenes, num_imgs, h, w, 3)
            cond_rays_o, cond_rays_d = get_cam_rays(cond_poses, cond_intrinsics, h, w)
            dt_gamma_scale = self.test_cfg.get('dt_gamma_scale', 0.0)
            # (num_scenes,)
            dt_gamma = dt_gamma_scale / cond_intrinsics[..., :2].mean(dim=(-2, -1))

            with torch.enable_grad():
                (code, density_grid, density_bitfield,
                 loss, loss_dict, out_rgbs, target_rgbs) = self.inverse_code(
                    decoder, cond_imgs, cond_rays_o, cond_rays_d,
                    dt_gamma=dt_gamma, cfg=self.test_cfg, show_pbar=show_pbar)

        # ==== evaluate reconstruction ====
        with torch.no_grad():
            if 'test_poses' in data:
                log_vars, pred_imgs = self.eval_and_viz(
                    data, decoder, code, density_bitfield,
                    viz_dir=viz_dir, cfg=self.test_cfg)
            else:
                log_vars = dict()
                pred_imgs = None
            if out_rgbs is not None and target_rgbs is not None:
                train_psnr = eval_psnr(out_rgbs, target_rgbs)
                log_vars.update(train_psnr=float(train_psnr.mean()))
            code_rms = code.square().flatten(1).mean().sqrt()
            log_vars.update(code_rms=float(code_rms.mean()))

        # ==== save 3D code ====
        save_dir = self.test_cfg.get('save_dir', None)
        if save_dir is not None:
            self.save_scene(save_dir, code, density_grid, density_bitfield, data['scene_name'])

        # ==== outputs ====
        outputs_dict = dict(
            log_vars=log_vars,
            num_samples=len(data['scene_name']),
            pred_imgs=pred_imgs)

        return outputs_dict

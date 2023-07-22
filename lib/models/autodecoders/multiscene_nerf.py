import os
import multiprocessing as mp
import numpy as np
import torch
import mmcv

from mmcv.runner import get_dist_info
from mmgen.models.builder import MODELS
from mmgen.models.architectures.common import get_module_device

from ...core import eval_psnr, optimizer_state_to, load_tensor_to_dict, \
    optimizer_state_copy, optimizer_set_state
from .base_nerf import BaseNeRF, get_cam_rays


def out_dict_to(d, device=None, code_dtype=torch.float32, optimizer_dtype=torch.float32):
    assert code_dtype.is_floating_point and optimizer_dtype.is_floating_point
    return dict(
        scene_id=d['scene_id'],
        scene_name=d['scene_name'],
        param=dict(
            code_=d['param']['code_'].clamp(
                min=torch.finfo(code_dtype).min, max=torch.finfo(code_dtype).max
            ).to(device=device, dtype=code_dtype),
            density_grid=d['param']['density_grid'].to(device=device),
            density_bitfield=d['param']['density_bitfield'].to(device=device)),
        optimizer=optimizer_state_to(d['optimizer'], device=device, dtype=optimizer_dtype))


@MODELS.register_module()
class MultiSceneNeRF(BaseNeRF):

    def __init__(self,
                 *args,
                 cache_size=0,  # cache in RAM, top priority
                 cache_16bit=False,
                 num_file_writers=0,  # cache in file system (for large dataset)
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.cache_size = cache_size
        self.cache_16bit = cache_16bit
        if cache_size > 0:
            rank, ws = get_dist_info()
            split_points = np.round(np.linspace(0, cache_size, num=ws + 1)).astype(np.int64)
            inds = np.arange(start=split_points[rank], stop=split_points[rank + 1])
            self.cache = {ind: None for ind in inds}
        else:
            self.cache = None
        self.cache_loaded = False

        self.num_file_writers = num_file_writers
        self.is_file_writers_initialized = False

    def init_file_writers(self, save_dir):
        if self.num_file_writers > 0:
            def file_writer(queue):
                while True:
                    obj = queue.get()
                    torch.save(obj, os.path.join(save_dir, obj['scene_name'] + '.pth'))

            self.file_queues = [mp.Queue(maxsize=1) for _ in range(self.num_file_writers)]
            for queue in self.file_queues:
                p = mp.Process(target=file_writer, args=(queue,))
                p.start()
        else:
            self.file_queues = None
        self.is_file_writers_initialized = True

    def load_cache(self, data):
        device = get_module_device(self)
        num_scenes = len(data['scene_id'])
        rank, ws = get_dist_info()

        if self.cache is not None:
            if not self.cache_loaded:
                cache_load_from = self.train_cfg.get('cache_load_from', None)
                loaded = False
                if cache_load_from is not None:
                    cache_files = os.listdir(cache_load_from)
                    cache_files.sort()
                    if len(cache_files) > 0:
                        assert len(cache_files) == self.cache_size
                        for ind in self.cache.keys():
                            self.cache[ind] = torch.load(
                                os.path.join(cache_load_from, cache_files[ind]), map_location='cpu')
                        loaded = True
                        if rank == 0:
                            mmcv.print_log('Loaded cache files from ' + cache_load_from + '.', 'mmgen')
                if not loaded:
                    if rank == 0:
                        mmcv.print_log('Initialize codes from scratch.', 'mmgen')
                self.cache_loaded = True
            cache_list = [self.cache[scene_id_single] for scene_id_single in data['scene_id']]
        elif 'code' in data:
            cache_list = data['code']
        else:
            cache_list = [None for _ in range(num_scenes)]
        code_list_ = []
        density_grid = []
        density_bitfield = []
        for scene_state_single in cache_list:
            if scene_state_single is None:
                code_list_.append(self.get_init_code_(None, device))
                density_grid.append(self.get_init_density_grid(None, device))
                density_bitfield.append(self.get_init_density_bitfield(None, device))
            else:
                code_list_.append((
                    scene_state_single['param']['code_'] if 'code_' in scene_state_single['param']
                    else self.code_activation.inverse(scene_state_single['param']['code'].to(dtype=torch.float32, device=device))
                ).to(dtype=torch.float32, device=device).requires_grad_(True))
                density_grid.append(scene_state_single['param']['density_grid'].to(device))
                density_bitfield.append(scene_state_single['param']['density_bitfield'].to(device))
        density_grid = torch.stack(density_grid, dim=0)
        density_bitfield = torch.stack(density_bitfield, dim=0)

        code_optimizers = self.build_optimizer(code_list_, self.train_cfg)
        for ind, scene_state_single in enumerate(cache_list):
            if scene_state_single is not None and 'optimizer' in scene_state_single:
                optimizer_set_state(code_optimizers[ind], scene_state_single['optimizer'])
        return code_list_, code_optimizers, density_grid, density_bitfield

    def save_cache(self, code_list_, code_optimizers,
                   density_grid, density_bitfield, scene_id, scene_name):
        if self.cache_16bit:
            code_dtype = torch.float16 if code_list_[0].dtype == torch.float32 else code_list_[0].dtype
            optimizer_dtype = torch.bfloat16
        else:
            code_dtype = code_list_[0].dtype
            optimizer_dtype = torch.float32
        if 'save_dir' in self.train_cfg:
            save_dir = self.train_cfg['save_dir']
            os.makedirs(save_dir, exist_ok=True)
            if not self.is_file_writers_initialized:
                self.init_file_writers(save_dir)
        else:
            save_dir = None
        for ind, code_single_ in enumerate(code_list_):
            scene_id_single = scene_id[ind]
            out = dict(
                scene_id=scene_id_single,
                scene_name=scene_name[ind],
                param=dict(
                    code_=code_single_.data,
                    density_grid=density_grid[ind],
                    density_bitfield=density_bitfield[ind]),
                optimizer=code_optimizers[ind].state_dict())
            if self.cache is not None:
                if self.cache[scene_id_single] is None:
                    self.cache[scene_id_single] = out_dict_to(
                        out, device='cpu', code_dtype=code_dtype, optimizer_dtype=optimizer_dtype)
                else:
                    if 'scene_id' not in self.cache[scene_id_single]:
                        self.cache[scene_id_single]['scene_id'] = out['scene_id']
                    if 'scene_name' not in self.cache[scene_id_single]:
                        self.cache[scene_id_single]['scene_name'] = out['scene_name']
                    if 'code' in self.cache[scene_id_single]['param']:
                        del self.cache[scene_id_single]['param']['code']
                    for key, val in out['param'].items():
                        load_tensor_to_dict(self.cache[scene_id_single]['param'], key, val,
                                            device='cpu', dtype=code_dtype)
                    if 'optimizer' in self.cache[scene_id_single]:
                        optimizer_state_copy(out['optimizer'], self.cache[scene_id_single]['optimizer'],
                                             device='cpu', dtype=optimizer_dtype)
                    else:
                        self.cache[scene_id_single]['optimizer'] = optimizer_state_to(
                            out['optimizer'], device='cpu', dtype=optimizer_dtype)
            if save_dir is not None:
                if self.file_queues is not None:
                    self.file_queues[ind // self.num_file_writers].put(
                        out_dict_to(out, device='cpu', code_dtype=code_dtype, optimizer_dtype=optimizer_dtype))
                else:
                    torch.save(
                        out_dict_to(out, device='cpu', code_dtype=code_dtype, optimizer_dtype=optimizer_dtype),
                        os.path.join(save_dir, scene_name + '.pth'))

    def train_step(self, data, optimizer, running_status=None):
        code_list_, code_optimizers, density_grid, density_bitfield = self.load_cache(data)

        # ==== optimize code ====
        cond_imgs = data['cond_imgs']  # (num_scenes, num_imgs, h, w, 3)
        cond_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
        cond_poses = data['cond_poses']

        num_scenes, num_imgs, h, w, _ = cond_imgs.size()
        # (num_scenes, num_imgs, h, w, 3)
        cond_rays_o, cond_rays_d = get_cam_rays(cond_poses, cond_intrinsics, h, w)
        dt_gamma_scale = self.train_cfg.get('dt_gamma_scale', 0.0)
        # (num_scenes,)
        dt_gamma = dt_gamma_scale / cond_intrinsics[..., :2].mean(dim=(-2, -1))

        extra_scene_step = self.train_cfg.get('extra_scene_step', 0)
        if extra_scene_step > 0:
            cfg = self.train_cfg.copy()
            cfg['n_inverse_steps'] = extra_scene_step
            self.inverse_code(
                self.decoder, cond_imgs, cond_rays_o, cond_rays_d, dt_gamma=dt_gamma, cfg=cfg,
                code_=code_list_,
                density_grid=density_grid,
                density_bitfield=density_bitfield,
                code_optimizer=code_optimizers)

        # ==== joint optimization ====
        for code_optimizer in code_optimizers:
            code_optimizer.zero_grad()
        optimizer['decoder'].zero_grad()

        code = self.code_activation(torch.stack(code_list_, dim=0), update_stats=True)

        self.update_extra_state(
            self.decoder, code, density_grid, density_bitfield,
            0, density_thresh=self.train_cfg.get('density_thresh', 0.01))

        loss, log_vars, out_rgbs, target_rgbs = self.loss_decoder(
            self.decoder, code, density_bitfield, cond_rays_o, cond_rays_d,
            cond_imgs, dt_gamma, cfg=self.train_cfg)
        loss.backward()
        log_vars.update(loss=float(loss))

        optimizer['decoder'].step()
        for code_optimizer in code_optimizers:
            code_optimizer.step()

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
                    data, self.decoder, code, density_bitfield, cfg=self.train_cfg))

        # ==== outputs ====
        outputs_dict = dict(
            log_vars=log_vars, num_samples=num_scenes)

        return outputs_dict

import os
import gc
import torch
import mmcv
import torch.distributed as dist

from copy import deepcopy
from torch.distributed import barrier
from torch.nn.parallel.distributed import DistributedDataParallel
from mmcv.runner import HOOKS, Hook, get_dist_info
from lib.datasets import build_dataloader


@HOOKS.register_module()
class SaveCacheHook(Hook):
    def __init__(self,
                 interval=-1,
                 by_epoch=False,
                 out_dir=None,
                 save_last=True,
                 viz_dir=None,
                 viz_step=32):
        self.interval = interval
        self.by_epoch = by_epoch
        self.out_dir = out_dir
        self.save_last = save_last
        self.viz_dir = viz_dir
        self.viz_step = viz_step
        rank, ws = get_dist_info()
        if rank == 0:
            os.makedirs(self.out_dir, exist_ok=True)
            if self.viz_dir is not None:
                os.makedirs(self.viz_dir, exist_ok=True)

    def _save(self, module):
        cache_list = module.cache
        for i, out in enumerate(cache_list.values()):
            if out is not None:
                torch.save(out, os.path.join(self.out_dir, out['scene_name'] + '.pth'))
                if i % self.viz_step == 0:
                    code = module.code_activation(out['param']['code_'])
                    decoder = module.decoder
                    if isinstance(decoder, DistributedDataParallel):
                        decoder = decoder.module
                    decoder.visualize(code[None], [out['scene_name']], self.viz_dir,
                                      code_range=module.test_cfg.get('clip_range', [-1, 1]))

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return

        if self.every_n_epochs(
                runner, self.interval) or (self.save_last
                                           and self.is_last_epoch(runner)):
            runner.logger.info(
                f'Saving cache files at {runner.epoch + 1} epochs')
            self._save(runner.model.module)

    def after_train_iter(self, runner):
        if self.by_epoch:
            return

        if self.every_n_iters(
                runner, self.interval) or (self.save_last
                                           and self.is_last_iter(runner)):
            runner.logger.info(
                f'Saving cache files at {runner.iter + 1} iterations')
            self._save(runner.model.module)


@HOOKS.register_module()
class ResetCacheHook(Hook):
    def __init__(self,
                 interval=-1,
                 by_epoch=False):
        self.interval = interval
        self.by_epoch = by_epoch

    @staticmethod
    def _reset(module):
        if module.cache is not None:
            for key, val in module.cache.items():
                module[key] = None
            gc.collect()
        else:
            raise NotImplementedError

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return

        if self.interval > 0 and runner.epoch > 0 and runner.epoch % self.interval == 0:
            runner.logger.info(
                f'Resetting cache files at {runner.epoch} epochs')
            self._reset(runner)

    def before_train_iter(self, runner):
        if self.by_epoch:
            return

        if self.interval > 0 and runner.iter > 0 and runner.iter % self.interval == 0:
            runner.logger.info(
                f'Resetting cache at {runner.iter} iterations')
            self._reset(runner)


@HOOKS.register_module()
class UpdateCacheHook(Hook):
    def __init__(self,
                 interval=-1,
                 by_epoch=False,
                 test_cfg=dict(),
                 viz_dir=None,
                 viz_step=32):
        self.interval = interval
        self.by_epoch = by_epoch
        self.test_cfg = test_cfg
        self.viz_dir = viz_dir
        self.viz_step = viz_step
        if self.viz_dir is not None:
            rank, ws = get_dist_info()
            if rank == 0:
                os.makedirs(self.viz_dir, exist_ok=True)

    def _update(self, runner):
        test_cfg_bak = deepcopy(runner.model.module.test_cfg)
        runner.model.module.test_cfg.update(self.test_cfg)
        assert runner.model.module.test_cfg.get('save_dir', None)
        runner.model.eval()

        dataloader = build_dataloader(
            runner.data_loader._dataloader.dataset,
            samples_per_gpu=runner.data_loader._dataloader.collate_fn.keywords['samples_per_gpu'],
            workers_per_gpu=runner.data_loader._dataloader.num_workers,
            num_gpus=runner.world_size,
            dist=dist.is_available() and dist.is_initialized(),
            shuffle=False,
            timeout=600)

        len_dataloader = len(dataloader)
        if runner.rank == 0:
            pbar = mmcv.ProgressBar(len_dataloader)
        for i, data in enumerate(dataloader):
            runner.model.val_step(
                data,
                show_pbar=runner.rank == 0,
                viz_dir=self.viz_dir if i % self.viz_step == 0 else None)
            if runner.rank == 0:
                pbar.update()

        barrier()
        runner.model.train()
        runner.model.module.test_cfg = test_cfg_bak
        runner.model.module.cache_loaded = False

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return

        if self.interval > 0 and runner.epoch > 0 and runner.epoch % self.interval == 0:
            runner.logger.info(
                f'Updating cache files at {runner.epoch} epochs')
            self._update(runner)

    def before_train_iter(self, runner):
        if self.by_epoch:
            return

        if self.interval > 0 and runner.iter > 0 and runner.iter % self.interval == 0:
            runner.logger.info(
                f'Updating cache at {runner.iter} iterations')
            self._update(runner)


@HOOKS.register_module()
class MeanCacheHook(Hook):
    def __init__(self,
                 step,
                 load_from=None,
                 by_epoch=True):
        self.by_epoch = by_epoch
        assert isinstance(step, list)
        self.step = step
        self.load_from = load_from

    def _update(self, runner):
        if runner.rank == 0:
            load_dir = self.load_from
            cache_files = os.listdir(load_dir)
            sum_code = None
            count = 0
            for cache_file in mmcv.track_iter_progress(cache_files):
                cache = torch.load(os.path.join(self.load_from, cache_file), map_location='cpu')
                if 'code_' in cache['param']:
                    code = runner.model.module.code_activation(cache['param']['code_'])
                elif 'code' in cache['param']:
                    code = cache['param']['code']
                else:
                    raise ValueError
                if sum_code is None:
                    sum_code = torch.zeros_like(code)
                sum_code += code
                count += 1
            mean_code = sum_code / count
            out_cache = dict(param=dict(
                code=mean_code,
                density_grid=runner.model.module.get_init_density_grid(None),
                density_bitfield=runner.model.module.get_init_density_bitfield(None)))
            out_dir = runner.model.module.train_cfg['cache_load_from']
            for cache_file in mmcv.track_iter_progress(cache_files):
                torch.save(out_cache, os.path.join(out_dir, cache_file))
        barrier()
        runner.model.module.cache_loaded = False

    def get_step_id(self, runner):
        progress = runner.epoch if self.by_epoch else runner.iter
        if progress in self.step:
            runner.logger.info(
                f'Resetting cache files to their mean value...')
            self._update(runner)

    def before_train_iter(self, runner):
        if not self.by_epoch:
            self.get_step_id(runner)

    def before_train_epoch(self, runner):
        if self.by_epoch:
            self.get_step_id(runner)

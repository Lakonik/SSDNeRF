import sys

import torch
from mmcv.runner import HOOKS, get_dist_info

from mmgen.core import GenerativeEvalHook
from lib.apis import evaluate_3d


@HOOKS.register_module()
class GenerativeEvalHook3D(GenerativeEvalHook):

    greater_keys = ['acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'is', 'test_ssim', 'test_psnr']
    less_keys = ['loss', 'fid', 'kid', 'test_lpips']
    _supported_best_metrics = ['fid', 'kid', 'is', 'test_ssim', 'test_psnr', 'test_lpips']

    def __init__(self,
                 *args,
                 data='',
                 viz_dir=None,
                 feed_batch_size=32,
                 viz_step=32,
                 **kwargs):
        super(GenerativeEvalHook3D, self).__init__(*args, **kwargs)
        self.data = data
        self.viz_dir = viz_dir
        self.feed_batch_size = feed_batch_size
        self.viz_step = viz_step

    @torch.no_grad()
    def after_train_iter(self, runner):
        interval = self.get_current_interval(runner)
        if not self.every_n_iters(runner, interval):
            return

        runner.model.eval()

        log_vars = evaluate_3d(
            runner.model, self.dataloader, self.metrics, self.feed_batch_size,
            self.viz_dir, self.viz_step, self.sample_kwargs)

        if len(runner.log_buffer.output) == 0:
            runner.log_buffer.clear()
        rank, ws = get_dist_info()
        # a dirty walkround to change the line at the end of pbar
        if rank == 0:
            sys.stdout.write('\n')
            for metric in self.metrics:
                with torch.no_grad():
                    metric.summary()
                for name, val in metric._result_dict.items():
                    runner.log_buffer.output[self.data + '_' + name] = val
                    # record best metric and save the best ckpt
                    if self.save_best_ckpt and name in self.best_metric:
                        self._save_best_ckpt(runner, val, name)
            for name, val in log_vars.items():
                print(self.data + '_' + name + ' = {}'.format(val))
                runner.log_buffer.output[self.data + '_' + name] = val
                # record best metric and save the best ckpt
                if self.save_best_ckpt and name in self.best_metric:
                    self._save_best_ckpt(runner, val, name)
            runner.log_buffer.ready = True

        runner.model.train()

        # clear all current states for next evaluation
        for metric in self.metrics:
            metric.clear()

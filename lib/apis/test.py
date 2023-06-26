from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import mmcv
from mmcv.runner import get_dist_info
from mmgen.core.evaluation.metrics import FID, IS
from mmgen.models.architectures.common import get_module_device


def evaluate_3d(model, dataloader, metrics=None,
                feed_batch_size=32, viz_dir=None, viz_step=1, sample_kwargs=dict()):
    batch_size = dataloader.batch_size
    rank, ws = get_dist_info()
    total_batch_size = batch_size * ws

    max_num_scenes = len(dataloader.dataset)
    if rank == 0:
        mmcv.print_log(
            f'Sample {max_num_scenes} fake scenes for evaluation', 'mmgen')
        pbar = mmcv.ProgressBar(max_num_scenes)

    log_vars = dict()
    batch_size_list = []
    # sampling fake images and directly send them to metrics
    for i, data in enumerate(dataloader):
        sample_kwargs_ = deepcopy(sample_kwargs)
        if viz_dir is not None and i % viz_step == 0:
            sample_kwargs_.update(viz_dir=viz_dir)
        outputs_dict = model.val_step(
            data, show_pbar=rank == 0,
            **sample_kwargs_)
        for k, v in outputs_dict['log_vars'].items():
            if k in log_vars:
                log_vars[k].append(outputs_dict['log_vars'][k])
            else:
                log_vars[k] = [outputs_dict['log_vars'][k]]
        batch_size_list.append(outputs_dict['num_samples'])

        if metrics is not None and len(metrics) > 0:
            pred_imgs = outputs_dict['pred_imgs'].reshape(
                -1, *outputs_dict['pred_imgs'].shape[2:]).split(feed_batch_size, dim=0)
            real_imgs = None
            for metric in metrics:
                if 'test_imgs' in data and not isinstance(metric, (FID, IS)) and real_imgs is None:
                    real_imgs = data['test_imgs'].permute(0, 1, 4, 2, 3)
                    real_imgs = real_imgs.reshape(-1, *real_imgs.shape[2:]).split(feed_batch_size, dim=0)
                for batch_id, batch_imgs in enumerate(pred_imgs):
                    # feed in fake images
                    metric.feed(batch_imgs * 2 - 1, 'fakes')
                    if not isinstance(metric, (FID, IS)) and real_imgs is not None:
                        metric.feed(real_imgs[batch_id] * 2 - 1, 'reals')

        if rank == 0:
            pbar.update(total_batch_size)

    if dist.is_initialized():
        device = get_module_device(model)
        batch_size_list = torch.tensor(batch_size_list, dtype=torch.float, device=device)
        batch_size_sum = torch.sum(batch_size_list)
        if dist.is_initialized():
            dist.all_reduce(batch_size_sum, op=dist.ReduceOp.SUM)
        for k, v in log_vars.items():
            weigted_values = torch.tensor(log_vars[k], dtype=torch.float, device=device) * batch_size_list
            weigted_values_sum = torch.sum(weigted_values)
            if dist.is_initialized():
                dist.all_reduce(weigted_values_sum, op=dist.ReduceOp.SUM)
            log_vars[k] = float(weigted_values_sum / batch_size_sum)
    else:
        for k, v in log_vars.items():
            log_vars[k] = np.average(log_vars[k], weights=batch_size_list)
    return log_vars

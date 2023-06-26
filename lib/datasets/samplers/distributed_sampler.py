# Modified from https://github.com/open-mmlab/mmgeneration

from __future__ import division

import random
import numpy as np
import torch

from mmcv.runner import get_dist_info
from mmgen.datasets import DistributedSampler as DistributedSampler_


class DistributedSampler(DistributedSampler_):

    def __init__(self,
                 *args,
                 split_data=False,
                 check_batch_disjoint=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.split_data = split_data
        self.init_split_inds()
        self.skip_iter = 0
        self.last_batch_inds = set()
        self.check_batch_disjoint = check_batch_disjoint

    def init_split_inds(self):
        rank, ws = get_dist_info()
        if self.split_data and ws > 1:
            assert self.num_replicas == ws
            split_points = np.round(np.linspace(0, len(self.dataset), num=ws + 1)).astype(np.int64)
            all_inds = [torch.arange(start=split_points[rank_cur], end=split_points[rank_cur + 1])
                        for rank_cur in range(ws)]
            self.inds = all_inds[rank]
            self.num_samples_per_replica = int(
                np.max([np.ceil(len(inds) * 1.0 / self.samples_per_gpu) for inds in all_inds]))
            self.num_samples = self.num_samples_per_replica * self.samples_per_gpu
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.inds = None

    def update_sampler(self, dataset, samples_per_gpu=None):
        super().update_sampler(dataset, samples_per_gpu)
        self.init_split_inds()

    def __iter__(self):
        if self.inds is None:
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = torch.arange(len(self.dataset)).tolist()
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size
            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
        else:
            # deterministically shuffle based on epoch
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                indices = self.inds[torch.randperm(len(self.inds), generator=g)].tolist()
            else:
                indices = self.inds.tolist()
            # add extra samples to make it evenly divisible
            indices += indices[:(self.num_samples - len(indices))]

        if self.check_batch_disjoint:
            if not set(indices[-2 * self.samples_per_gpu:-self.samples_per_gpu]).isdisjoint(
                    set(indices[-self.samples_per_gpu:])):
                raise RuntimeError('Batch Disjoint Check Failed! Dataset is too small.')
            if not set(indices[:self.samples_per_gpu]).isdisjoint(self.last_batch_inds):
                first_batch_pool = set(indices[2 * self.samples_per_gpu:]).difference(self.last_batch_inds)
                assert len(first_batch_pool) >= self.samples_per_gpu, \
                    'Batch Disjoint Check Failed! Dataset is too small.'
                # maybe non-deterministic
                indices[:self.samples_per_gpu] = random.choices(list(first_batch_pool), k=self.samples_per_gpu)
            self.last_batch_inds = set(indices[-self.samples_per_gpu:])

        indices = indices[self.skip_iter:]
        self.skip_iter = 0
        return iter(indices)

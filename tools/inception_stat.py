# Modified from https://github.com/open-mmlab/mmgeneration

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))

import argparse
import pickle
import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv import Config, print_log
from mmcv.parallel import is_module_wrapper

# yapf: disable
from mmgen.core.evaluation.metric_utils import extract_inception_features  # isort:skip  # noqa
from mmgen.datasets import (UnconditionalImageDataset, build_dataloader,  # isort:skip  # noqa
                            build_dataset)  # isort:skip  # noqa
from mmgen.models.architectures import InceptionV3  # isort:skip  # noqa
from mmgen.models.architectures.common import get_module_device
from lib.core import download_from_url

# yapf: enable

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-calculate inception data and save it in pkl file')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='batch size used in extracted features')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    for eval_cfg in cfg.evaluation:
        metrics = eval_cfg['metrics']
        if isinstance(metrics, dict):
            metrics = [metrics]
        fid_metric = None
        for metric in metrics:
            if metric['type'] in ['FID', 'FIDKID']:
                fid_metric = metric
        if fid_metric is None:
            continue

        data_cfg = cfg.data[eval_cfg.data]
        data_cfg['num_train_imgs'] = 0
        data_cfg['load_imgs'] = True
        if 'specific_observation_idcs' in data_cfg:
            del data_cfg['specific_observation_idcs']
        if 'max_num_scenes' in data_cfg:
            del data_cfg['max_num_scenes']
        dataset = build_dataset(data_cfg)
        data_loader = build_dataloader(dataset, 1, 4, dist=False)

        pkl_path = fid_metric['inception_pkl']
        pkl_dir = os.path.dirname(pkl_path)
        mmcv.mkdir_or_exist(pkl_dir)

        inception_style = fid_metric['inception_args']['type'].lower()

        # build inception network
        if inception_style == 'stylegan':
            inception_pth = fid_metric['inception_args']['inception_path']
            if not os.path.exists(inception_pth):
                pth_dir = os.path.dirname(inception_pth)
                mmcv.mkdir_or_exist(pth_dir)
                try:
                    path = download_from_url(
                        'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt',
                        dest_dir=pth_dir)
                    mmcv.print_log('Download Finished.')
                except Exception as e:
                    mmcv.print_log(f'Download Failed. {e} occurs.')
            inception = torch.jit.load(inception_pth).eval().cuda()
            inception = nn.DataParallel(inception)
            print_log('Adopt Inception network in StyleGAN', 'mmgen')
        else:
            inception = nn.DataParallel(
                InceptionV3([3], resize_input=True, normalize_input=False).cuda())
            inception.eval()

        feature_list = []

        pbar = mmcv.ProgressBar(len(data_loader))

        for scene in data_loader:
            all_imgs = scene['test_imgs'].squeeze(0).permute(0, 3, 1, 2) * 2 - 1
            img_batches = all_imgs.split(args.batch_size, dim=0)
            for img in img_batches:
                # the inception network is not wrapped with module wrapper.
                if not is_module_wrapper(inception):
                    # put the img to the module device
                    img = img.to(get_module_device(inception))

                if inception_style == 'stylegan':
                    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    feature = inception(img, return_features=True)
                else:
                    feature = inception(img)[0].view(img.shape[0], -1)
                feature_list.append(feature.to('cpu'))
            pbar.update()

        # Attention: the number of features may be different as you want.
        features = torch.cat(feature_list, 0).numpy()
        num_samples = features.shape[0]

        # to change the line after pbar
        sys.stdout.write('\n')

        print_log(f'Extract {num_samples} features', 'mmgen')

        mean = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False)

        with open(pkl_path, 'wb') as f:
            pickle.dump(
                {
                    'feats_np': features,
                    'mean': mean,
                    'cov': cov,
                    'size': num_samples,
                    'name': os.path.splitext(os.path.basename(pkl_path))[1]
                }, f)

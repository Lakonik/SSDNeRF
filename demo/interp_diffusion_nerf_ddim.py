import argparse
import os
import sys

from mmcv import Config, DictAction
from mmgen.datasets import build_dataset

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmgen.apis import set_random_seed  # isort:skip  # noqa
from lib.apis import init_model, interp_diffusion_nerf_ddim
# yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description='Interpolation demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    # parser.add_argument('--nerf', help='nerf checkpoint file')
    parser.add_argument(
        '--viz-dir',
        type=str,
        help='path to save unconditional samples')
    parser.add_argument(
        '--type',
        type=str,
        default='spherical_linear',
        choices=['spherical_linear', 'linear'],
        help='path to save unconditional samples')
    parser.add_argument(
        '--pose-ids',
        type=int,
        nargs='+',
        default=[64])
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA device id')
    # args for inference/sampling
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='The total number of samples')
    parser.add_argument(
        '--batchsize', type=int, default=10, help='Batch size in inference')
    parser.add_argument(
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    set_random_seed(args.seed, deterministic=args.deterministic)
    model = init_model(
        args.config, checkpoint=args.checkpoint, device=args.device)
    assert model.diffusion.sample_method == 'ddim'

    if args.viz_dir is None:
        args.viz_dir = './viz/interp_{}_'.format(args.seed) + args.type

    if args.sample_cfg is None:
        args.sample_cfg = dict()

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val_uncond)

    sample = dataset.parse_scene(0)
    test_poses = sample['test_poses']
    test_intrinsics = sample['test_intrinsics']

    if args.pose_ids is not None:
        test_poses = test_poses[args.pose_ids]
        test_intrinsics = test_intrinsics[args.pose_ids]

    interp_diffusion_nerf_ddim(
        model, test_poses, test_intrinsics, viz_dir=args.viz_dir,
        num_samples=args.num_samples, batchsize=args.batchsize, type=args.type,
        **args.sample_cfg)


if __name__ == '__main__':
    main()

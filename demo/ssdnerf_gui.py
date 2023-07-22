import os
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '8'

import argparse

from lib.core import SSDNeRFGUI


def parse_args():
    parser = argparse.ArgumentParser(description='SSDNeRF GUI')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use')
    parser.add_argument('--cameras', type=str, default='demo/camera_spiral_cars')
    parser.add_argument(
        '--fp16',
        action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.gpu_ids is not None:
        gpu_ids = args.gpu_ids
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_ids = [0]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_ids])
    if len(gpu_ids) != 1:
        raise NotImplementedError('multi-gpu inference is not yet supported')

    from lib.apis import init_model
    # build the model from a config file and a checkpoint file
    model = init_model(
        args.config, checkpoint=args.checkpoint, use_fp16=args.fp16)
    model.eval()
    nerf_gui = SSDNeRFGUI(model, cameras=args.cameras)
    nerf_gui.render()
    return


if __name__ == '__main__':
    main()

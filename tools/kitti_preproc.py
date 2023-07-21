# Some code is borrowed from https://github.com/tjiiv-cprg/EPro-PnP

import os
import os.path as osp
import argparse
import mmcv
import numpy as np
from scipy.linalg import solve_triangular


def yaw_to_rot_mat(yaw):
    """
    Args:
        yaw: (*)

    Returns:
        rot_mats: (*, 3, 3)
    """
    sin_yaw = np.sin(yaw)
    cos_yaw = np.cos(yaw)
    # [[ cos_yaw, 0, sin_yaw],
    #  [       0, 1,       0],
    #  [-sin_yaw, 0, cos_yaw]]
    rot_mats = np.zeros(yaw.shape + (3, 3), dtype=np.float32)
    rot_mats[..., 0, 0] = cos_yaw
    rot_mats[..., 2, 2] = cos_yaw
    rot_mats[..., 0, 2] = sin_yaw
    rot_mats[..., 2, 0] = -sin_yaw
    rot_mats[..., 1, 1] = 1
    return rot_mats


def open_label_file(path):
    with open(path) as f_label:
        label = [[float(v) if i != 0 and i != 2
                  else int(float(v)) if i == 2 else v
                  for i, v in enumerate(line_label.strip().split(' '))]
                 for line_label in f_label]
    return label


def open_calib_file(calib_file, cam=2):
    assert 0 <= cam <= 3
    with open(calib_file) as f_calib:
        f_calib = f_calib.readlines()[cam]
        proj_mat = np.array(
            [float(v) for v in f_calib.strip().split(' ')[1:]], dtype=np.float32
        ).reshape((3, 4))
    return proj_mat


rot_conversion = np.array(
    [[0, 1, 0],
     [0, 0, -1],
     [-1, 0, 0]],
    dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess the KITTI dataset')
    parser.add_argument(
        '--kitti-dir',
        default='data/kitti/training')
    parser.add_argument(
        '--out-dir',
        default='data/shapenet/cars_kitti')
    parser.add_argument(
        '--out-size',
        type=int,
        default=128)
    parser.add_argument(
        '--out-border',
        type=int,
        default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    kitti_dir = args.kitti_dir
    out_dir = args.out_dir
    out_size = args.out_size
    out_border = args.out_border

    image_dir = kitti_dir + '/image_2'
    seg_dir = kitti_dir + '/instance_2'
    label_dir = kitti_dir + '/label_2'
    calib_dir = kitti_dir + '/calib'

    resize_tgt = out_size - out_border * 2
    os.makedirs(out_dir, exist_ok=True)
    label_files = os.listdir(label_dir)
    label_files.sort()

    for label_file in mmcv.track_iter_progress(label_files):
        basename = osp.splitext(label_file)[0]
        label = open_label_file(osp.join(label_dir, label_file))
        cali_mat = open_calib_file(osp.join(calib_dir, label_file))
        cali_t_vec = cali_mat[:, 3:]
        cam_intrinsic = cali_mat[:, :3]
        cam_t_vec = solve_triangular(
            cam_intrinsic, cali_t_vec, lower=False).squeeze(-1)
        image = mmcv.imread(osp.join(image_dir, basename + '.png'), 'unchanged')
        seg = mmcv.imread(osp.join(seg_dir, basename + '.png'), 'unchanged')
        for i, instance in enumerate(label):
            if instance[1] == 0 and instance[2] == 0:
                mask = seg == i + 1000
                mask_inds = mask.nonzero()
                if len(mask_inds[0]) == 0:
                    continue
                y_min = mask_inds[0].min()
                y_max = mask_inds[0].max() + 1
                h = y_max - y_min
                x_min = mask_inds[1].min()
                x_max = mask_inds[1].max() + 1
                w = x_max - x_min
                img_crop_ori = image[y_min:y_max, x_min:x_max]
                mask_crop = mask[y_min:y_max, x_min:x_max]
                img_crop_ori[~mask_crop] = 255

                bbox_3d = np.array(instance[8:], dtype=np.float32)
                bbox_3d[[0, 1, 2]] = bbox_3d[[2, 0, 1]]  # to lhw
                diag = np.linalg.norm(bbox_3d[:3])
                bbox_3d[3:6] += cam_t_vec  # move to camera space
                bbox_3d[4] -= bbox_3d[1] / 2  # center offset
                bbox_3d[:6] /= diag
                rot_mat = yaw_to_rot_mat(bbox_3d[6]) @ rot_conversion
                c2w = np.concatenate(
                    [rot_mat.T, rot_mat.T @ (-bbox_3d[3:6])[:, None]], axis=1)
                c2w = np.concatenate(
                    [c2w, [[0, 0, 0, 1]]], axis=0)

                hw_max = max(h, w)
                pad_tgt = max(round(np.linalg.norm(bbox_3d[:3]) * cam_intrinsic[0, 0] / bbox_3d[5]), hw_max)
                scale = resize_tgt / pad_tgt
                if scale > 1:
                    continue
                pad_x_l = (pad_tgt - w) // 2
                pad_x_r = pad_tgt - w - pad_x_l
                pad_y_t = (pad_tgt - h) // 2
                pad_y_b = pad_tgt - h - pad_y_t
                img_crop = np.pad(img_crop_ori, ((pad_y_t, pad_y_b), (pad_x_l, pad_x_r), (0, 0)), constant_values=255)
                img_crop = mmcv.imresize(img_crop, (resize_tgt, resize_tgt))
                img_crop = np.pad(img_crop, ((out_border, out_border), (out_border, out_border), (0, 0)),
                                  constant_values=255)

                out_instance = basename + '_{:03d}'.format(i)
                out_instance_dir = osp.join(out_dir, out_instance)
                rgb_dir = osp.join(out_instance_dir, 'rgb')
                pose_dir = osp.join(out_instance_dir, 'pose')
                os.makedirs(rgb_dir, exist_ok=True)
                os.makedirs(pose_dir, exist_ok=True)
                mmcv.imwrite(img_crop, osp.join(rgb_dir, '000000.png'))
                mmcv.imwrite(img_crop_ori, osp.join(out_instance_dir, '000000.png'))
                np.savetxt(osp.join(pose_dir, '000000.txt'), c2w.reshape(1, -1))
                out_intrinsics = '{:.6f} {:.6f} {:.6f} 0.\n0. 0. 0.\n1.\n{} {}\n'.format(
                    cam_intrinsic[0, 0] * scale,
                    (cam_intrinsic[0, 2] - x_min + pad_x_l) * scale + out_border,
                    (cam_intrinsic[1, 2] - y_min + pad_y_t) * scale + out_border,
                    img_crop.shape[0], img_crop.shape[1])
                with open(osp.join(out_instance_dir, 'intrinsics.txt'), 'w') as f:
                    f.write(out_intrinsics)


if __name__ == '__main__':
    main()

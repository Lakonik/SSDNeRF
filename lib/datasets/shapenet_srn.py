import os
import random
import numpy as np
import torch
import mmcv
from torch.utils.data import Dataset

from mmcv.parallel import DataContainer as DC
from mmgen.datasets.builder import DATASETS


def load_intrinsics(path):
    with open(path, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
        grid_barycenter = list(map(float, file.readline().split()))
        scale = float(file.readline())
        height, width = map(int, file.readline().split())
    fx = fy = f
    return fx, fy, cx, cy, height, width


def load_pose(path):
    pose = np.loadtxt(path, dtype=np.float32, delimiter=' ').reshape(4, 4)
    return torch.from_numpy(pose)


@DATASETS.register_module()
class ShapeNetSRN(Dataset):
    def __init__(self,
                 data_prefix,
                 code_dir=None,
                 code_only=False,
                 load_imgs=True,
                 specific_observation_idcs=None,
                 num_test_imgs=0,
                 random_test_imgs=False,
                 scene_id_as_name=False,
                 cache_path=None,
                 test_pose_override=None,
                 num_train_imgs=-1,
                 load_cond_data=True,
                 load_test_data=True,
                 max_num_scenes=-1,  # for debug or testing
                 radius=0.5,
                 test_mode=False,
                 step=1,  # only for debug & visualization purpose
                 ):
        super(ShapeNetSRN, self).__init__()
        self.data_prefix = data_prefix
        self.code_dir = code_dir
        self.code_only = code_only
        self.load_imgs = load_imgs
        self.specific_observation_idcs = specific_observation_idcs
        self.num_test_imgs = num_test_imgs
        self.random_test_imgs = random_test_imgs
        self.scene_id_as_name = scene_id_as_name
        self.cache_path = cache_path
        self.test_pose_override = test_pose_override
        self.num_train_imgs = num_train_imgs
        self.load_cond_data = load_cond_data
        self.load_test_data = load_test_data
        self.max_num_scenes = max_num_scenes
        self.step = step

        self.radius = torch.tensor([radius], dtype=torch.float32).expand(3)
        self.center = torch.zeros_like(self.radius)

        self.load_scenes()

        if self.test_pose_override is not None:
            image_dir = os.path.join(self.test_pose_override, 'rgb')
            image_names = os.listdir(image_dir)
            image_names.sort()
            poses_list = []
            for image_name in image_names:
                pose_path = os.path.join(
                    self.test_pose_override, 'pose/' + os.path.splitext(image_name)[0] + '.txt')
                c2w = torch.FloatTensor(load_pose(pose_path))
                cam_to_ndc = torch.cat(
                    [c2w[:3, :3], (c2w[:3, 3:] - self.center[:, None]) / self.radius[:, None]], dim=-1)
                poses_list.append(
                    torch.cat([
                        cam_to_ndc,
                        cam_to_ndc.new_tensor([[0.0, 0.0, 0.0, 1.0]])
                    ], dim=-2))
            self.test_poses = torch.stack(poses_list, dim=0)  # (n, 4, 4)
            fx, fy, cx, cy, h, w = load_intrinsics(os.path.join(self.test_pose_override, 'intrinsics.txt'))
            intrinsics_single = torch.FloatTensor([fx, fy, cx, cy])
            self.test_intrinsics = intrinsics_single[None].expand(self.test_poses.size(0), -1)
        else:
            self.test_poses = self.test_intrinsics = None

    def load_scenes(self):
        if self.cache_path is not None and os.path.exists(self.cache_path):
            scenes = mmcv.load(self.cache_path)
        else:
            data_prefix_list = self.data_prefix if isinstance(self.data_prefix, list) else [self.data_prefix]
            scenes = []
            for data_prefix in data_prefix_list:
                sample_dir_list = os.listdir(data_prefix)
                # sample_dir_list.sort()
                for name in sample_dir_list:
                    sample_dir = os.path.join(data_prefix, name)
                    if os.path.isdir(sample_dir):
                        intrinsics = load_intrinsics(os.path.join(sample_dir, 'intrinsics.txt'))
                        image_dir = os.path.join(sample_dir, 'rgb')
                        image_names = os.listdir(image_dir)
                        image_names.sort()
                        image_paths = []
                        poses = []
                        for image_name in image_names:
                            image_paths.append(os.path.join(image_dir, image_name))
                            pose_path = os.path.join(
                                sample_dir, 'pose/' + os.path.splitext(image_name)[0] + '.txt')
                            poses.append(load_pose(pose_path))
                        scenes.append(dict(
                            intrinsics=intrinsics,
                            image_paths=image_paths,
                            poses=poses))
            scenes = sorted(scenes, key=lambda x: x['image_paths'][0].split('/')[-3])
            if self.cache_path is not None:
                mmcv.dump(scenes, self.cache_path)
        end = len(scenes)
        if self.max_num_scenes >= 0:
            end = min(end, self.max_num_scenes * self.step)
        self.scenes = scenes[:end:self.step]
        self.num_scenes = len(self.scenes)

    def parse_scene(self, scene_id):
        scene = self.scenes[scene_id]
        image_paths = scene['image_paths']
        scene_name = image_paths[0].split('/')[-3]
        results = dict(
            scene_id=DC(scene_id, cpu_only=True),
            scene_name=DC(
                '{:04d}'.format(scene_id) if self.scene_id_as_name else scene_name,
                cpu_only=True))

        if not self.code_only:
            fx, fy, cx, cy, h, w = scene['intrinsics']
            intrinsics_single = torch.FloatTensor([fx, fy, cx, cy])
            poses = scene['poses']

            def gather_imgs(img_ids):
                imgs_list = [] if self.load_imgs else None
                poses_list = []
                img_paths_list = []
                for img_id in img_ids:
                    pose = poses[img_id]
                    c2w = torch.FloatTensor(pose)
                    cam_to_ndc = torch.cat(
                        [c2w[:3, :3], (c2w[:3, 3:] - self.center[:, None]) / self.radius[:, None]], dim=-1)
                    poses_list.append(
                        torch.cat([
                            cam_to_ndc,
                            cam_to_ndc.new_tensor([[0.0, 0.0, 0.0, 1.0]])
                        ], dim=-2))
                    img_paths_list.append(image_paths[img_id])
                    if self.load_imgs:
                        img = mmcv.imread(image_paths[img_id], channel_order='rgb')
                        img = torch.from_numpy(img.astype(np.float32) / 255)  # (h, w, 3)
                        imgs_list.append(img)
                poses_list = torch.stack(poses_list, dim=0)  # (n, 4, 4)
                intrinsics = intrinsics_single[None].expand(len(img_ids), -1)
                if self.load_imgs:
                    imgs_list = torch.stack(imgs_list, dim=0)  # (n, h, w, 3)
                return imgs_list, poses_list, intrinsics, img_paths_list

            num_imgs = len(image_paths)
            if self.specific_observation_idcs is None:
                if self.num_train_imgs >= 0:
                    num_train_imgs = self.num_train_imgs
                else:
                    num_train_imgs = num_imgs - self.num_test_imgs
                if self.random_test_imgs:
                    cond_inds = random.sample(range(num_imgs), num_train_imgs)
                else:
                    cond_inds = np.round(np.linspace(0, num_imgs - 1, num_train_imgs)).astype(np.int64)
            else:
                cond_inds = self.specific_observation_idcs
            test_inds = list(range(num_imgs))
            for cond_ind in cond_inds:
                test_inds.remove(cond_ind)

            if self.load_cond_data and len(cond_inds) > 0:
                cond_imgs, cond_poses, cond_intrinsics, cond_img_paths = gather_imgs(cond_inds)
                results.update(
                    cond_poses=cond_poses,
                    cond_intrinsics=cond_intrinsics,
                    cond_img_paths=DC(cond_img_paths, cpu_only=True))
                if cond_imgs is not None:
                    results.update(cond_imgs=cond_imgs)

            if self.load_test_data and len(test_inds) > 0:
                test_imgs, test_poses, test_intrinsics, test_img_paths = gather_imgs(test_inds)
                results.update(
                    test_poses=test_poses,
                    test_intrinsics=test_intrinsics,
                    test_img_paths=DC(test_img_paths, cpu_only=True))
                if test_imgs is not None:
                    results.update(test_imgs=test_imgs)

        if self.code_dir is not None:
            code_file = os.path.join(self.code_dir, scene_name + '.pth')
            if os.path.exists(code_file):
                results.update(
                    code=DC(torch.load(code_file, map_location='cpu'), cpu_only=True))

        if self.test_pose_override is not None:
            results.update(test_poses=self.test_poses, test_intrinsics=self.test_intrinsics)

        return results

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, scene_id):
        return self.parse_scene(scene_id)

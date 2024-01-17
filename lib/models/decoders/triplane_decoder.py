import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F
import numpy as np

from mmcv.cnn import xavier_init, constant_init
from mmgen.models.builder import MODULES

from .base_volume_renderer import VolumeRenderer
from lib.ops import SHEncoder, TruncExp
import math


def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    elev = math.atan2(z,math.sqrt(XsqPlusYsq))     # theta
    az = math.atan2(y,x)                           # phi
    return elev, az

def fibonacci_sphere(samples=1000):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append(cart2sph(x,y,z))

    return points


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

rot_eta = lambda eta : torch.Tensor([
    [np.cos(eta),np.sin(eta),0,0],
    [-np.sin(eta), np.cos(eta),0,0],
    [0,0,1,0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    eta = 180
    c2w = rot_eta(eta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],
                                 [0,0,1,0],
                                 [0,1,0,0],
                                 [0,0,0,1]])) @ c2w
    return c2w.numpy()

class ImagePlanes(torch.nn.Module):
    def __init__(self, focal, poses, images, count=np.inf, device='cuda'):
        super(ImagePlanes, self).__init__()

        self.pose_matrices = []
        self.K_matrices = []
        self.images = []

        self.focal = focal
        for i in range(min(count, poses.shape[0])):
            M = poses[i]
            M = torch.from_numpy(M)
            M = M @ torch.Tensor([[-1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]]).to(M.device)
            M = torch.inverse(M)
            M = M[0:3]
            self.pose_matrices.append(M)

            image = images[i]
            # image = torch.from_numpy(image)
            self.images.append(image)#.permute(2, 0, 1))
            self.size = float(image.shape[0])
            K = torch.Tensor(
                [[1.0254, 0, 0.5],
                 [0, 1.0254, 0.5],
                 [0, 0, 1]])
            self.K_matrices.append(K)

        self.pose_matrices = torch.stack(self.pose_matrices).to(device)
        self.K_matrices = torch.stack(self.K_matrices).to(device)
        self.image_plane = torch.stack(self.images).to(device)

    def forward(self, points=None):
        if points.shape[0] == 1:
            points = points[0]

        points = torch.concat([points, torch.ones(points.shape[0], 1).to(points.device)], 1).to(points.device)
        points_in_camera_coords = self.pose_matrices @ points.T
        # camera-origin distance is equal to 1 in points_in_camera_coords
        ps = self.K_matrices @ points_in_camera_coords
        pixels = (ps / ps[:, None, 2])[:, 0:2, :]
        # pixels = pixels / self.size
        # print("Pixels")
        # p = pixels.flatten()
        # print(pixels.min(), torch.quantile(p, 0.05),torch.quantile(p, 0.5), pixels.max())
        pixels = torch.clamp(pixels, 0, 1)
        # print("Pixels")
        # print(pixels)
        pixels = pixels * 2.0 - 1.0
        pixels = pixels.permute(0, 2, 1)

        #print(pixels.shape)
        #print(self.image_plane.shape)

        num_points = pixels.shape[1]

        feats = []
        for img in range(self.image_plane.shape[0]):
            feat = torch.nn.functional.grid_sample(
                self.image_plane[img].unsqueeze(0),
                pixels[img].unsqueeze(0).unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False)
            feats.append(feat)

        feats = torch.stack(feats).squeeze(1)
        pixels = pixels.permute(1, 0, 2)
        pixels = pixels.flatten(1)

        feats = feats.permute(2, 3, 0, 1).squeeze(0)
        feats = feats.reshape(num_points, -1)
        # print(feats[0].shape) # torch.Size([262144, 96])
        # print(pixels.shape) # torch.Size([262144, 6])

        feats = torch.cat((feats, pixels), 1)
        return feats


@MODULES.register_module()
class TriPlaneDecoder(VolumeRenderer):

    activation_dict = {
        'relu': nn.ReLU,
        'silu': nn.SiLU,
        'softplus': nn.Softplus,
        'trunc_exp': TruncExp}

    def __init__(self,
                 *args,
                 interp_mode='bilinear',
                 base_layers=[3 * 6, 128],
                 density_layers=[128, 1],
                 color_layers=[128, 128, 3],
                 use_dir_enc=True,
                 dir_layers=None,
                 scene_base_size=None,
                 scene_rand_dims=(0, 1),
                 activation='silu',
                 sigma_activation='trunc_exp',
                 sigmoid_saturation=0.001,
                 code_dropout=0.0,
                 flip_z=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.interp_mode = interp_mode
        self.in_chn = base_layers[0]
        self.use_dir_enc = use_dir_enc
        if scene_base_size is None:
            self.scene_base = None
        else:
            rand_size = [1 for _ in scene_base_size]
            for dim in scene_rand_dims:
                rand_size[dim] = scene_base_size[dim]
            init_base = torch.randn(rand_size).expand(scene_base_size).clone()
            self.scene_base = nn.Parameter(init_base)
        self.dir_encoder = SHEncoder() if use_dir_enc else None
        self.sigmoid_saturation = sigmoid_saturation

        activation_layer = self.activation_dict[activation.lower()]

        base_net = []
        for i in range(len(base_layers) - 1):
            base_net.append(nn.Linear(base_layers[i], base_layers[i + 1]))
            if i != len(base_layers) - 2:
                base_net.append(activation_layer())
        self.base_net = nn.Sequential(*base_net)
        self.base_activation = activation_layer()

        density_net = []
        for i in range(len(density_layers) - 1):
            density_net.append(nn.Linear(density_layers[i], density_layers[i + 1]))
            if i != len(density_layers) - 2:
                density_net.append(activation_layer())
        density_net.append(self.activation_dict[sigma_activation.lower()]())
        self.density_net = nn.Sequential(*density_net)

        self.dir_net = None
        color_net = []
        if use_dir_enc:
            if dir_layers is not None:
                dir_net = []
                for i in range(len(dir_layers) - 1):
                    dir_net.append(nn.Linear(dir_layers[i], dir_layers[i + 1]))
                    if i != len(dir_layers) - 2:
                        dir_net.append(activation_layer())
                self.dir_net = nn.Sequential(*dir_net)
            else:
                color_layers[0] = color_layers[0] + 16  # sh_encoding
        for i in range(len(color_layers) - 1):
            color_net.append(nn.Linear(color_layers[i], color_layers[i + 1]))
            if i != len(color_layers) - 2:
                color_net.append(activation_layer())
        color_net.append(nn.Sigmoid())
        self.color_net = nn.Sequential(*color_net)

        self.code_dropout = nn.Dropout2d(code_dropout) if code_dropout > 0 else None
        self.flip_z = flip_z

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
        if self.dir_net is not None:
            constant_init(self.dir_net[-1], 0)

    # def xyz_transform(self, xyz):
    #     if self.flip_z:
    #         xyz = torch.cat([xyz[..., :2], -xyz[..., 2:]], dim=-1)
    #     xy = xyz[..., :2]
    #     xz = xyz[..., ::2]
    #     yz = xyz[..., 1:]
    #     if xyz.dim() == 2:
    #         out = torch.stack([xy, xz, yz], dim=0).unsqueeze(1)  # (3, 1, num_points, 2)
    #     elif xyz.dim() == 3:
    #         num_scenes, num_points, _ = xyz.size()
    #         out = torch.stack([xy, xz, yz], dim=1).reshape(num_scenes * 3, 1, num_points, 2)
    #     else:
    #         raise ValueError
    #     return out

    def point_decode(self, xyzs, dirs, code, density_only=False):
        """
        Args:
            xyzs: Shape (num_scenes, (num_points_per_scene, 3))
            dirs: Shape (num_scenes, (num_points_per_scene, 3))
            code: Shape (num_scenes, 3, n_channels, h, w)
        """
        num_scenes, _, n_channels, h, w = code.size()
        if self.code_dropout is not None:
            code = self.code_dropout(
                code.reshape(num_scenes * 3, n_channels, h, w)
            ).reshape(num_scenes, 3, n_channels, h, w)


        if self.scene_base is not None:
            code = code + self.scene_base


        # if isinstance(xyzs, torch.Tensor):
        #     assert xyzs.dim() == 3
        #     num_points = xyzs.size(-2)
        #     point_code = F.grid_sample(
        #         code.reshape(num_scenes * 3, -1, h, w),
        #         self.xyz_transform(xyzs),
        #         mode=self.interp_mode, padding_mode='border', align_corners=False
        #     ).reshape(num_scenes, 3, -1, num_points)
        #     point_code = point_code.permute(0, 3, 2, 1).reshape(
        #         num_scenes * num_points, -1)
        #     num_points = [num_points] * num_scenes

        # else:

        num_points = []
        point_code = []
        image_planes = []

        for code_single, xyzs_single in zip(code, xyzs):

            num_points_per_scene = xyzs_single.size(-2)
            # (3, code_chn, num_points_per_scene)
            # point_code_single = F.grid_sample(
            #     code_single,
            #     self.xyz_transform(xyzs_single),
            #     mode=self.interp_mode, padding_mode='border', align_corners=False
            # ).squeeze(-2)

            poses = [pose_spherical(theta, phi, -1.307) for phi, theta in fibonacci_sphere(16)]

            image_plane = ImagePlanes(focal=torch.Tensor([10.0]),
                                      poses=np.stack(poses),
                                      images=code_single.view(16, 3, code.shape[-2], code.shape[-1]))

            image_planes.append(image_plane)
            point_code_single = image_plane(xyzs_single) #### Czy rozmiary beda sie zgadzac???


            # print('!!!!--!!!!')
            # print(point_code_single.permute(2, 1, 0).shape)
            # # point_code_single = point_code_single.permute(2, 1, 0).reshape(
            # #     num_points_per_scene, -1)
            # print('!!!!')
            # print(point_code_single.shape)
            # print(xyzs[0].shape)
            # print(dirs[0].shape)
            num_points.append(num_points_per_scene)
            point_code.append(point_code_single)


        point_code = torch.cat(point_code, dim=0) if len(point_code) > 1 \
            else point_code[0]


        base_x = self.base_net(point_code)
        base_x_act = self.base_activation(base_x)
        sigmas = self.density_net(base_x_act).squeeze(-1)
        if density_only:
            rgbs = None
        else:
            if self.use_dir_enc:
                dirs = torch.cat(dirs, dim=0) if num_scenes > 1 else dirs[0]
                sh_enc = self.dir_encoder(dirs)
                if self.dir_net is not None:
                    color_in = self.base_activation(base_x + self.dir_net(sh_enc))
                else:
                    color_in = torch.cat([base_x_act, sh_enc], dim=-1)
            else:
                color_in = base_x_act
            rgbs = self.color_net(color_in)
            if self.sigmoid_saturation > 0:
                rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation


        return sigmas, rgbs, num_points

    def point_density_decode(self, xyzs, code, **kwargs):
        sigmas, _, num_points = self.point_decode(
            xyzs, None, code, density_only=True, **kwargs)
        return sigmas, num_points

    def visualize(self, code, scene_name, viz_dir, code_range=[-1, 1]):
        num_scenes, _, num_chn, h, w = code.size()
        code_viz = code.cpu().numpy()
        for code_viz_single, scene_name_single in zip(code_viz, scene_name):
            with open(os.path.join(viz_dir, 'scene_' + scene_name_single + '.pkl'), 'wb') as file:
                pickle.dump(code_viz_single, file)
        if not self.flip_z:
            code_viz = code_viz[..., ::-1, :]
        code_viz = code_viz.transpose(0, 1, 3, 2, 4).reshape(num_scenes, 3 * h, num_chn * w)
        for code_viz_single, scene_name_single in zip(code_viz, scene_name):
            plt.imsave(os.path.join(viz_dir, 'scene_' + scene_name_single + '.png'), code_viz_single,
                       vmin=code_range[0], vmax=code_range[1])

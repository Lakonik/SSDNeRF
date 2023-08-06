# modified from torch-ngp

import os
import random
import math
import numpy as np
import trimesh
import torch
import torch.nn.functional as F
import mmcv
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

from mmgen.models.architectures.common import get_module_device
from mmgen.apis import set_random_seed  # isort:skip  # noqa
from .utils import extract_geometry, surround_views
from lib.datasets.shapenet_srn import load_pose, load_intrinsics
from videoio import VideoWriter


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = fovy  # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [0.5, -0.5, 0.5, -0.5])
        self.up = np.array([0, 0, 1], dtype=np.float32)  # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return np.round(res, 3)

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]  # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])


class SSDNeRFGUI:
    def __init__(self, model, cameras=None, camera_id=64,
                 W=512, H=512, radius=2, fovy=60, max_spp=1,
                 debug=True):
        self.W = W
        self.H = H
        self.max_spp = max_spp
        self.cam = OrbitCamera(W, H, r=radius, fovy=fovy)
        self.debug = debug
        self.bg_color = torch.ones(3, dtype=torch.float32)  # default white bg
        self.step = 0  # training step

        self.model = model
        self.model_decoder = model.decoder_ema if model.decoder_use_ema else model.decoder
        self.model_diffusion = model.diffusion_ema if model.diffusion_use_ema else model.diffusion

        assert cameras is not None
        pose_dir = os.path.join(cameras, 'pose')
        poses = os.listdir(pose_dir)
        poses.sort()
        pose = poses[camera_id]
        c2w = torch.FloatTensor(load_pose(os.path.join(pose_dir, pose)))
        cam_to_ndc = torch.cat([c2w[:3, :3], c2w[:3, 3:] * 2], dim=-1)
        self.camera_pose = torch.cat([
            cam_to_ndc,
            cam_to_ndc.new_tensor([[0.0, 0.0, 0.0, 1.0]])
        ], dim=-2)  # (4, 4)
        fx, fy, cx, cy, h, w = load_intrinsics(os.path.join(cameras, 'intrinsics.txt'))
        self.video_sec = 4
        self.video_fps = 30
        self.video_res = 256
        self.camera_intrinsics = torch.FloatTensor([fx, fy, cx, cy])
        self.camera_intrinsics_hw = [h, w]

        self.render_buffer = np.zeros((self.H, self.W, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation
        self.spp = 1  # sample per pixel
        self.dt_gamma_scale = 0.0
        self.density_thresh = 0.1
        self.mode = 'image'  # choose from ['image', 'depth']

        self.mesh_resolution = 256
        self.mesh_threshold = 10
        self.scene_name = 'model_default'

        self.diffusion_seed = -1
        self.diffusion_steps = 20

        if self.model.init_code is None:
            self.code_buffer = torch.zeros(self.model.code_size, device=get_module_device(self.model))
        else:
            self.code_buffer = self.model.init_code.clone()
        _, self.density_bitfield = self.model.get_density(
            self.model_decoder, self.code_buffer[None],
            cfg=dict(density_thresh=self.density_thresh, density_step=16))

        self.dynamic_resolution = False
        self.downscale = 1

        dpg.create_context()
        self.register_dpg()
        self.test_step()

    def __del__(self):
        dpg.destroy_context()

    def prepare_buffer(self, outputs):
        if self.mode == 'image':
            return outputs['image']
        else:
            return np.expand_dims(outputs['depth'], -1).repeat(3, -1)

    def test_gui(self, pose, intrinsics, W, H, bg_color, spp, dt_gamma_scale, downscale):
        with torch.no_grad():
            self.model.bg_color = bg_color.to(self.code_buffer.device)
            image, depth = self.model.render(
                self.model_decoder,
                self.code_buffer[None],
                self.density_bitfield[None], H, W,
                self.code_buffer.new_tensor(intrinsics * downscale)[None, None],
                self.code_buffer.new_tensor(pose)[None, None],
                cfg=dict(dt_gamma_scale=dt_gamma_scale))
            results = dict(
                image=image[0, 0],
                depth=depth[0, 0])
            if downscale != 1:
                # TODO: have to permute twice with torch...
                results['image'] = F.interpolate(
                    results['image'].permute(2, 0, 1)[None], size=(H, W), mode='nearest'
                ).permute(0, 2, 3, 1).reshape(H, W, 3)
                results['depth'] = F.interpolate(results['depth'][None, None], size=(H, W), mode='nearest').reshape(H, W)
            results['image'] = results['image'].cpu().numpy()
            results['depth'] = results['depth'].cpu().numpy()
        return results

    def update_params(self):
        with torch.no_grad():
            self.density_bitfield = self.model.get_density(
                self.model_decoder, self.code_buffer[None],
                cfg=dict(density_thresh=self.density_thresh, density_step=16))[1].squeeze(0)

    def test_step(self):
        # TODO: seems we have to move data from GPU --> CPU --> GPU?

        if self.need_update or self.spp < self.max_spp:

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            outputs = self.test_gui(
                self.cam.pose, self.cam.intrinsics,
                self.W, self.H, self.bg_color, self.spp, self.dt_gamma_scale, self.downscale)

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1 / 4, math.sqrt(200 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            if self.need_update:
                self.render_buffer = np.ascontiguousarray(self.prepare_buffer(outputs))
                self.spp = 1
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + self.prepare_buffer(outputs)) / (self.spp + 1)
                self.spp += 1

            dpg.set_value('_log_infer_time', f'{t:.4f}ms ({int(1000 / t)} FPS)')
            dpg.set_value('_log_resolution', f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            dpg.set_value('_log_spp', self.spp)
            dpg.set_value('_log_scene_name', self.scene_name)
            dpg.set_value('_texture', self.render_buffer)

    def register_dpg(self):

        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag='_texture')

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag='_primary_window', width=self.W, height=self.H):

            # add the texture
            dpg.add_image('_texture')

        dpg.set_primary_window('_primary_window', True)

        # control window
        with dpg.window(label='Control', tag='_control_window', width=400, height=max(300, self.H), pos=[self.W, 0]):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time
            with dpg.group(horizontal=True):
                dpg.add_text('Infer time: ')
                dpg.add_text('no data', tag='_log_infer_time')

            with dpg.group(horizontal=True):
                dpg.add_text('SPP: ')
                dpg.add_text('1', tag='_log_spp')

            with dpg.collapsing_header(label='SSDNeRF', default_open=True):

                def callback_diffusion_generate(sender, app_data):
                    diffusion_seed = random.randint(0, 2**31) if self.diffusion_seed == -1 else self.diffusion_seed
                    set_random_seed(diffusion_seed, deterministic=True)
                    noise = torch.randn((1,) + self.model.code_size)
                    self.model_diffusion.test_cfg['num_timesteps'] = self.diffusion_steps
                    data = dict(
                        noise=noise.to(get_module_device(self.model)),
                        scene_id=[0],
                        scene_name=['seed_{}'.format(diffusion_seed)])
                    with torch.no_grad():
                        code, density_grid, density_bitfield = self.model.val_uncond(
                            data, show_pbar=True, save_intermediates=False)
                    self.code_buffer = code[0]
                    self.density_bitfield = density_bitfield[0]
                    self.scene_name = 'seed_{}'.format(diffusion_seed)
                    self.need_update = True

                def callback_set_diffusion_seed(sender, app_data):
                    self.diffusion_seed = app_data

                def callback_set_diffusion_steps(sender, app_data):
                    self.diffusion_steps = app_data

                with dpg.group(horizontal=True):
                    dpg.add_button(label='Generate', callback=callback_diffusion_generate)
                    dpg.add_input_int(
                        label='seed', width=130, min_value=-1, max_value=2**31 - 1, min_clamped=True, max_clamped=True,
                        default_value=self.diffusion_seed, callback=callback_set_diffusion_seed, tag='seed')
                    dpg.add_input_int(
                        label='steps', width=80, min_value=1, max_value=1000, min_clamped=True, max_clamped=True,
                        default_value=self.diffusion_steps, callback=callback_set_diffusion_steps)

                def callback_save_scene(sender, app_data):
                    path = app_data['file_path_name']
                    out = dict(
                        param=dict(
                            code=self.code_buffer.cpu(),
                            density_bitfield=self.density_bitfield.cpu()))
                    torch.save(out, path)

                with dpg.file_dialog(directory_selector=False, show=False, width=450, height=400,
                                     callback=callback_save_scene, tag='save_scene_dialog'):
                    dpg.add_file_extension('.pth')

                with dpg.group(horizontal=True):
                    dpg.add_button(label='Save scene', callback=lambda: dpg.show_item('save_scene_dialog'))

                # scene selector
                def callback_load_scene(sender, app_data):
                    self.scene_name = os.path.splitext(app_data['file_name'])[0]
                    scene = torch.load(app_data['file_path_name'], map_location='cpu')
                    self.code_buffer.copy_(
                        scene['param']['code'] if 'code' in scene['param']
                        else self.model.code_activation(scene['param']['code_']))
                    self.update_params()
                    print('Loaded scene: ' + self.scene_name)
                    self.need_update = True

                def callback_recover_seed(sender, app_data):
                    if self.scene_name.startswith('seed_'):
                        seed = int(self.scene_name[5:])
                        self.diffusion_seed = seed
                        dpg.set_value('seed', seed)
                        print('Recovered seed: ' + str(seed))
                    else:
                        print('Failed to recover seed: ' + self.scene_name)

                with dpg.file_dialog(directory_selector=False, show=False, width=450, height=400,
                                     callback=callback_load_scene, tag='scene_selector_dialog'):
                    dpg.add_file_extension('.pth')

                with dpg.group(horizontal=True):
                    dpg.add_button(label='Load scene', callback=lambda: dpg.show_item('scene_selector_dialog'))
                    dpg.add_text(tag='_log_scene_name')
                    dpg.add_button(label='Recover seed', callback=callback_recover_seed)

                # save geometry
                def callback_export_mesh(sender, app_data):
                    self.export_mesh(app_data['file_path_name'])

                def callback_save_code(sender, app_data):
                    dir_path = app_data['file_path_name']
                    assert os.path.isdir(dir_path), dir_path + ' is not a directory'
                    self.model_decoder.visualize(self.code_buffer[None], [self.scene_name], dir_path)

                def callback_set_mesh_resolution(sender, app_data):
                    self.mesh_resolution = app_data

                def callback_set_mesh_threshold(sender, app_data):
                    self.mesh_threshold = app_data

                def callback_set_video_resolution(sender, app_data):
                    self.video_res = app_data

                def callback_set_video_sec(sender, app_data):
                    self.video_sec = app_data

                def callback_export_video(sender, app_data):
                    path = app_data['file_path_name']
                    num_frames = int(round(self.video_fps * self.video_sec))
                    res_scale = self.video_res / self.camera_intrinsics_hw[0]
                    out_res = (
                        int(round(self.camera_intrinsics_hw[0] * res_scale)),
                        int(round(self.camera_intrinsics_hw[1]) * res_scale))
                    camera_poses = surround_views(self.camera_pose, num_frames=num_frames)
                    writer = VideoWriter(
                        path,
                        resolution=out_res,
                        lossless=False,
                        fps=self.video_fps)
                    bs = 4
                    device = self.code_buffer.device
                    with torch.no_grad():
                        prog = mmcv.ProgressBar(num_frames)
                        prog.start()
                        for pose_batch in camera_poses.split(bs, dim=0):
                            image_batch, depth = self.model.render(
                                self.model_decoder,
                                self.code_buffer[None],
                                self.density_bitfield[None], out_res[0], out_res[1],
                                (self.camera_intrinsics.to(device)[None] * res_scale).expand(pose_batch.size(0), -1)[None],
                                pose_batch.to(device)[None])
                            for image in np.round(image_batch[0].cpu().numpy() * 255).astype(np.uint8):
                                writer.write(image)
                            prog.update(bs)
                    writer.close()

                with dpg.file_dialog(directory_selector=False, show=False, width=450, height=400,
                                     callback=callback_export_mesh, tag='export_mesh_dialog'):
                    dpg.add_file_extension('.stl')
                    dpg.add_file_extension('.dict')
                    dpg.add_file_extension('.json')
                    dpg.add_file_extension('.glb')
                    dpg.add_file_extension('.obj')
                    dpg.add_file_extension('.gltf')
                    dpg.add_file_extension('.dict64')
                    dpg.add_file_extension('.msgpack')
                    dpg.add_file_extension('.stl_ascii')

                with dpg.file_dialog(directory_selector=False, show=False, width=450, height=400,
                                     callback=callback_save_code, tag='save_code_dialog'):
                    dpg.add_file_extension('.')

                with dpg.file_dialog(directory_selector=False, show=False, width=450, height=400,
                                     callback=callback_export_video, tag='export_video_dialog'):
                    dpg.add_file_extension('.mp4')

                dpg.add_button(label='Visualize code', callback=lambda: dpg.show_item('save_code_dialog'))

                with dpg.group(horizontal=True):
                    dpg.add_button(label='Export mesh', callback=lambda: dpg.show_item('export_mesh_dialog'))
                    dpg.add_input_int(
                        label='res', width=100, min_value=4, max_value=1024, min_clamped=True, max_clamped=True,
                        default_value=self.mesh_resolution, callback=callback_set_mesh_resolution)
                    dpg.add_input_float(
                        label='thr', width=100, min_value=0, max_value=1000, min_clamped=True, max_clamped=True,
                        format='%.2f', default_value=self.mesh_threshold, callback=callback_set_mesh_threshold)

                with dpg.group(horizontal=True):
                    dpg.add_button(label='Export video', callback=lambda: dpg.show_item('export_video_dialog'))
                    dpg.add_input_int(
                        label='res', width=100, min_value=4, max_value=1024, min_clamped=True, max_clamped=True,
                        default_value=self.video_res, callback=callback_set_video_resolution)
                    dpg.add_input_float(
                        label='len', width=100, min_value=0, max_value=10, min_clamped=True, max_clamped=True,
                        default_value=self.video_sec, callback=callback_set_video_sec, format='%.1f sec')

            with dpg.collapsing_header(label='Render options', default_open=True):

                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    def callback_set_dynamic_resolution(sender, app_data):
                        if self.dynamic_resolution:
                            self.dynamic_resolution = False
                            self.downscale = 1
                        else:
                            self.dynamic_resolution = True
                        self.need_update = True

                    dpg.add_checkbox(label='dynamic resolution', default_value=self.dynamic_resolution,
                                     callback=callback_set_dynamic_resolution)
                    dpg.add_text(f'{self.W}x{self.H}', tag='_log_resolution')

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(('image', 'depth'), label='mode', default_value=self.mode, callback=callback_change_mode)

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
                    self.need_update = True

                dpg.add_color_edit((255, 255, 255), label='Background Color', width=200, tag='_color_editor',
                                   no_alpha=True, callback=callback_change_bg)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(
                    label='FoV (vertical)', min_value=1, max_value=120, clamped=True,
                    format='%d deg', default_value=self.cam.fovy, callback=callback_set_fovy)

                # dt_gamma_scale slider
                def callback_set_dt_gamma_scale(sender, app_data):
                    self.dt_gamma_scale = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label='dt_gamma_scale', min_value=0, max_value=1.0, clamped=True,
                    format='%.2f', default_value=self.dt_gamma_scale, callback=callback_set_dt_gamma_scale)

                # max_steps slider
                def callback_set_max_steps(sender, app_data):
                    self.model_decoder.max_steps = app_data
                    self.need_update = True

                dpg.add_slider_int(
                    label='max steps', min_value=1, max_value=1024, clamped=True,
                    format='%d', default_value=self.model_decoder.max_steps, callback=callback_set_max_steps)

                # aabb slider
                def callback_set_aabb(sender, app_data, user_data):
                    # user_data is the dimension for aabb (xmin, ymin, zmin, xmax, ymax, zmax)
                    self.model_decoder.aabb[user_data] = app_data
                    self.need_update = True

                dpg.add_separator()
                dpg.add_text('Axis-aligned bounding box:')

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label='x', width=150, min_value=-self.model_decoder.bound, max_value=0, format='%.2f',
                                         default_value=-self.model_decoder.bound, callback=callback_set_aabb, user_data=0)
                    dpg.add_slider_float(label='', width=150, min_value=0, max_value=self.model_decoder.bound, format='%.2f',
                                         default_value=self.model_decoder.bound, callback=callback_set_aabb, user_data=3)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label='y', width=150, min_value=-self.model_decoder.bound, max_value=0, format='%.2f',
                                         default_value=-self.model_decoder.bound, callback=callback_set_aabb, user_data=1)
                    dpg.add_slider_float(label='', width=150, min_value=0, max_value=self.model_decoder.bound, format='%.2f',
                                         default_value=self.model_decoder.bound, callback=callback_set_aabb, user_data=4)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label='z', width=150, min_value=-self.model_decoder.bound, max_value=0, format='%.2f',
                                         default_value=-self.model_decoder.bound, callback=callback_set_aabb, user_data=2)
                    dpg.add_slider_float(label='', width=150, min_value=0, max_value=self.model_decoder.bound, format='%.2f',
                                         default_value=self.model_decoder.bound, callback=callback_set_aabb, user_data=5)

            # debug info
            if self.debug:
                with dpg.collapsing_header(label='Debug'):
                    # pose
                    dpg.add_separator()
                    dpg.add_text('Camera Pose:')
                    dpg.add_text(str(self.cam.pose), tag='_log_pose')

        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused('_primary_window'):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value('_log_pose', str(self.cam.pose))

        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused('_primary_window'):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value('_log_pose', str(self.cam.pose))

        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused('_primary_window'):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value('_log_pose', str(self.cam.pose))

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

        dpg.create_viewport(
            title='SSDNeRF GUI',
            width=self.W + 400, height=self.H + 50,
            resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme('_primary_window', theme_no_padding)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def render(self):

        while dpg.is_dearpygui_running():
            # update texture every frame
            self.test_step()
            dpg.render_dearpygui_frame()

    def export_mesh(self, save_path):
        print(f'==> Saving mesh to {save_path}')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vertices, triangles = extract_geometry(
            self.model_decoder,
            self.code_buffer,
            resolution=self.mesh_resolution,
            threshold=self.mesh_threshold)
        mesh = trimesh.Trimesh(vertices, triangles, process=False)  # important, process=True leads to seg fault...
        mesh.export(save_path)
        print(f'==> Finished saving mesh.')

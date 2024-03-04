import os
name = 'ssdnerf_mplane_cars_uncond_consistency'

DATA_PATH = '/net/tscratch/people/plgmarzol/SSDNeRF_data/shapenet'

cache_path = os.getcwd()
cache_path = os.path.dirname(cache_path)
cache_path = os.path.join(cache_path, "SSDNeRF_data", "shapenet")
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

model = dict(
    type='DiffusionNeRF',
    code_size=(3, 6, 128, 128),
    code_reshape=(18, 128, 128),
    code_activation=dict(
        type='TanhCode',
        scale=2),
    grid_size=64,
    diffusion=dict(
        type='GaussianDiffusion',
        num_timesteps=1000,
        betas_cfg=dict(type='linear'),
        denoising=dict(
            type='DenoisingUnetMod',
            image_size=128,  # size of triplanes (not images)
            in_channels=18,
            base_channels=128,
            channels_cfg=[1, 2, 2, 4, 4],
            resblocks_per_downsample=2,
            dropout=0.0,
            use_scale_shift_norm=True,
            downsample_conv=True,
            upsample_conv=True,
            num_heads=4,
            attention_res=[32, 16, 8]),
        timestep_sampler=dict(
            type='SNRWeightedTimeStepSampler',
            power=0.5),  # ω (SNR power)
        ddpm_loss=dict(
            type='DDPMMSELossMod',
            rescale_mode='timestep_weight',
            log_cfgs=dict(
                type='quartile', prefix_name='loss_mse', total_timesteps=1000),
            data_info=dict(pred='v_t_pred', target='v_t'),
            weight_scale=4.0,  # c_diff (diffusion weight constant)
            scale_norm=True)),
    decoder=dict(
        type='TriPlaneDecoder',
        interp_mode='bilinear',
        base_layers=[6 * 5, 64],
        density_layers=[64, 1],
        color_layers=[64, 3],
        use_dir_enc=True,
        dir_layers=[16, 64],
        activation='silu',
        sigma_activation='trunc_exp',
        sigmoid_saturation=0.001,
        max_steps=256),
    decoder_use_ema=True,
    freeze_decoder=False,
    bg_color=1,
    pixel_loss=dict(
        type='MSELoss',
        loss_weight=20.0),  # (0.5 * 2^14) * c_rend (rendering weight constant)
    reg_loss=dict(
        type='RegLoss',
        power=2,
        loss_weight=3e-3),
    cache_size=2458)  # number of training scenes

save_interval = 5000
eval_interval = 10000
code_dir = 'cache/' + name + '/code'
work_dir = 'work_dirs/' + name

train_cfg = dict(
    dt_gamma_scale=0.5,
    density_thresh=0.1,
    extra_scene_step=15,  # -1 + K_in (inner loop iterations)
    n_inverse_rays=2 ** 12,  # ray batch size
    n_consistency_rays=2 ** 12,  # ray batch size for consistency rendering
    n_decoder_rays=2 ** 12,  # ray batch size (used in the final inner iteration that updates the decoder)
    loss_coef=0.1 / (128 * 128),  # 0.1: the exponent in the λ_rend equation; 128 x 128: number of rays per view (image size)
    optimizer=dict(type='Adam', lr=5e-3, weight_decay=0.),
    cache_load_from=code_dir,
    viz_dir=None)
test_cfg = dict(
    img_size=(128, 128),  # size of rendered images
    num_timesteps=50,  # DDIM steps
    clip_range=[-2, 2],
    density_thresh=0.1,
    # max_render_rays=16 * 128 * 128,  # uncomment this line to use less rendering memory
    # uncomment the following lines to save NeRFs and meshes
    # save_dir=work_dir + '/save',
    # save_mesh=True,
    # mesh_resolution=256,
    # mesh_threshold=10,
)

optimizer = dict(
    diffusion=dict(type='Adam', lr=1e-4, weight_decay=0.),
    decoder=dict(type='Adam', lr=1e-3, weight_decay=0.))
dataset_type = 'ShapeNetSRN'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix= f'{DATA_PATH}/cars_train',
        cache_path= f'{cache_path}/cars_train_cache.pkl'),
    val_uncond=dict(
        type=dataset_type,
        data_prefix= f'{DATA_PATH}/cars_test',
        load_imgs=False,
        num_test_imgs=251,
        scene_id_as_name=True,
        cache_path= f'{cache_path}/cars_test_cache.pkl'),
    val_cond=dict(
        type=dataset_type,
        data_prefix= f'{DATA_PATH}/cars_test',
        specific_observation_idcs=[64],
        cache_path= f'{cache_path}/cars_test_cache.pkl'),
    train_dataloader=dict(split_data=True))
lr_config = dict(  # decay schedule of diffusion & decoder lr
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    gamma=0.5,
    step=[500000])

lr_config = dict(  # decay schedule of diffusion & decoder lr
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    gamma=0.5,
    step=[500000])

checkpoint_config = dict(interval=save_interval, by_epoch=False, max_keep_ckpts=2)

evaluation = [
    dict(
        type='GenerativeEvalHook3D',
        data='val_uncond',
        interval=eval_interval,
        feed_batch_size=32,
        viz_step=32,
        metrics=dict(
            type='FIDKID',
            num_images=704 * 251,
            inception_pkl='work_dirs/cache/cars_test_inception_stylegan.pkl',
            inception_args=dict(
                type='StyleGAN',
                inception_path='work_dirs/cache/inception-2015-12-05.pt'),
            bgr2rgb=False),
        viz_dir=work_dir + '/viz_uncond',
        save_best_ckpt=False)]

total_iters = 1000000  # K_out (outer loop iterations)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('diffusion_ema', 'decoder_ema'),
        interp_mode='lerp',
        interval=1,
        start_iter=0,
        momentum_policy='rampup',
        momentum_cfg=dict(
            ema_kimg=4, ema_rampup=0.05, batch_size=16, eps=1e-8),
        priority='VERY_HIGH'),
    dict(
        type='SaveCacheHook',
        interval=save_interval,
        by_epoch=False,
        out_dir=code_dir,
        viz_dir='cache/' + name + '/viz'),
    dict(
        type='ModelUpdaterHook',
        step=[2000, 100000, 500000],
        cfgs=[{'train_cfg.extra_scene_step': 3},  # decay schedule of K_in & triplane lr
              {'train_cfg.extra_scene_step': 1,
               'diffusion.ddpm_loss.freeze_norm': True},
              {'train_cfg.extra_scene_step': 1,
               'train_cfg.optimizer.lr': 2.5e-3,
               'diffusion.ddpm_loss.freeze_norm': True,
               'pixel_loss.loss_weight': 10.0,
               'reg_loss.loss_weight': 1.5e-3}],
        by_epoch=False)
]

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,
    pass_training_status=True)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', save_interval)]
use_ddp_wrapper = True
find_unused_parameters = False
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'

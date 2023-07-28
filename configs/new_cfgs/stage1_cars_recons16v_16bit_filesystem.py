name = 'stage1_cars_recons16v_16bit_filesystem'

model = dict(
    type='MultiSceneNeRF',
    code_size=(3, 6, 128, 128),
    code_activation=dict(
        type='NormalizedTanhCode', mean=0.0, std=0.5, clip_range=2),
    grid_size=64,
    decoder=dict(
        type='TriPlaneDecoder',
        interp_mode='bilinear',
        base_layers=[6 * 3, 64],
        density_layers=[64, 1],
        color_layers=[64, 3],
        use_dir_enc=True,
        dir_layers=[16, 64],
        activation='silu',
        sigma_activation='trunc_exp',
        sigmoid_saturation=0.001,
        max_steps=256),
    decoder_use_ema=True,
    bg_color=1,
    pixel_loss=dict(
        type='MSELoss',
        loss_weight=20.0),
    reg_loss=dict(
        type='TVLoss',
        power=1.5,
        loss_weight=1.0),
    num_file_writers=4,
    cache_16bit=True,
    init_from_mean=True)

save_interval = 5000
eval_interval = 20000
code_dir = 'cache/' + name + '/code'
code_bak_dir = 'cache/' + name + '/code_bak'
work_dir = 'work_dirs/' + name

train_cfg = dict(
    dt_gamma_scale=0.5,
    density_thresh=0.1,
    extra_scene_step=15,
    n_inverse_rays=2 ** 12,
    n_decoder_rays=2 ** 12,
    loss_coef=0.1 / (128 * 128),
    optimizer=dict(type='Adam', lr=0.04, weight_decay=0.),
    save_dir=code_dir,
    viz_dir=None)  # for displaying unscaled loss
test_cfg = dict(
    img_size=(128, 128),
    density_thresh=0.1,
    # max_render_rays=16 * 128 * 128,
    dt_gamma_scale=0.5,
    n_inverse_rays=2 ** 14,
    loss_coef=0.1 / (128 * 128),
    n_inverse_steps=400,
    optimizer=dict(type='Adam', lr=0.32, weight_decay=0.),
    lr_scheduler=dict(type='ExponentialLR', gamma=0.998)
)

optimizer = dict(
    decoder=dict(type='Adam', lr=1e-3, weight_decay=0.))
dataset_type = 'ShapeNetSRN'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        code_dir=code_dir,
        data_prefix='data/shapenet/cars_train',
        cache_path='data/shapenet/cars_train_cache.pkl'),
    val_cond=dict(
        type=dataset_type,
        data_prefix='data/shapenet/cars_test',
        num_test_imgs=251 - 16,
        cache_path='data/shapenet/cars_test_cache.pkl'),
    train_dataloader=dict(split_data=True, check_batch_disjoint=True))
lr_config = dict(
    policy='Fixed',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001)
checkpoint_config = dict(interval=save_interval, by_epoch=False, max_keep_ckpts=2)

evaluation = [
    dict(
        type='GenerativeEvalHook3D',
        data='val_cond',
        interval=eval_interval,
        feed_batch_size=32,
        viz_step=32,
        metrics=dict(
            type='FID',
            num_images=704 * (251 - 16),
            inception_pkl='work_dirs/cache/cars_test_inception_stylegan.pkl',
            inception_args=dict(
                type='StyleGAN',
                inception_path='work_dirs/cache/inception-2015-12-05.pt'),
            bgr2rgb=False),  # already is rgb
        viz_dir=work_dir + '/viz_cond',
        save_best_ckpt=False)]

total_iters = 400000
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
        module_keys=('decoder_ema', ),
        interp_mode='lerp',
        interval=1,
        start_iter=0,
        momentum_policy='rampup',
        momentum_cfg=dict(
            ema_kimg=4, ema_rampup=0.05, batch_size=16, eps=1e-8),
        priority='VERY_HIGH'),
    # Backup scene codes in case training is interrupted. Please manually replace the
    # files in `code_dir` with backups in `code_bak_dir` before resuming training.
    dict(
        type='DirCopyHook',
        interval=save_interval,
        in_dir=code_dir,
        out_dir=code_bak_dir),
    dict(
        type='ModelUpdaterHook',
        step=[2000],
        cfgs=[{'train_cfg.extra_scene_step': 3}],
        by_epoch=False)
]

# use dynamic runner
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

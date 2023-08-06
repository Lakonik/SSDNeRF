_base_ = ['../ssdnerf_cars_recons1v.py']


test_cfg = dict(
    n_inverse_steps=200,  # K_out (finetuning outer loop iterations)
    extra_scene_step=7,  # -1 + K_in (finetuning inner loop iterations)
    optimizer=dict(type='Adam', lr=0.08, weight_decay=0.),  # finetuning triplane lr
)

data = dict(
    val_cond=dict(
        specific_observation_idcs=None,
        num_test_imgs=251 - 32,
    ))

evaluation = [
    dict(
        type='GenerativeEvalHook3D',
        data='val_cond',
        feed_batch_size=32,
        viz_step=32,
        metrics=dict(
            type='FID',
            num_images=704 * (251 - 32),
            inception_pkl='work_dirs/cache/cars_test_inception_stylegan.pkl',
            inception_args=dict(
                type='StyleGAN',
                inception_path='work_dirs/cache/inception-2015-12-05.pt'),
            bgr2rgb=False),
        viz_dir='work_dirs/ssdnerf_cars_recons32v/viz_cond')]

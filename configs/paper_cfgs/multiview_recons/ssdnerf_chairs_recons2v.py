_base_ = ['../ssdnerf_chairs_recons1v.py']


test_cfg = dict(
    n_inverse_steps=50,  # K_out (finetuning outer loop iterations)
    optimizer=dict(type='Adam', lr=0.01, weight_decay=0.),  # finetuning triplane lr
)

data = dict(
    val_cond=dict(
        specific_observation_idcs=[64, 104]
    ))

evaluation = [
    dict(
        type='GenerativeEvalHook3D',
        data='val_cond',
        feed_batch_size=32,
        viz_step=32,
        metrics=dict(
            type='FID',
            num_images=1317 * 249,
            inception_pkl='work_dirs/cache/chairs_test_inception_stylegan.pkl',
            inception_args=dict(
                type='StyleGAN',
                inception_path='work_dirs/cache/inception-2015-12-05.pt'),
            bgr2rgb=False),
        viz_dir='work_dirs/ssdnerf_chairs_recons2v/viz_cond')]

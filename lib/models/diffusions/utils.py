import torch


def _get_noise_batch_3d(noise,
                        grid_shape,
                        num_timesteps=0,
                        num_batches=0,
                        timesteps_noise=False):
    """Get noise batch. Support get sequeue of noise along timesteps.

    We support the following use cases ('bz' denotes ```num_batches`` and 'n'
    denotes ``num_timesteps``):

    If timesteps_noise is True, we output noise which dimension is 5.
    - Input is [bz, c, d, h, w]: Expand to [n, bz, c, d, h, w]
    - Input is [n, c, d, h, w]: Expand to [n, bz, c, d, h, w]
    - Input is [n*bz, c, d, h, w]: View to [n, bz, c, d, h, w]
    - Dim of the input is 5: Return the input, ignore ``num_batches`` and
      ``num_timesteps``
    - Callable or None: Generate noise shape as [n, bz, c, d, h, w]
    - Otherwise: Raise error

    If timestep_noise is False, we output noise which dimension is 4 and
    ignore ``num_timesteps``.
    - Dim of the input is 4: Unsqueeze to [1, c, d, h, w], ignore ``num_batches``
    - Dim of the input is 5: Return input, ignore ``num_batches``
    - Callable or None: Generate noise shape as [bz, c, d, h, w]
    - Otherwise: Raise error

    It's to be noted that, we do not move the generated label to target device
    in this function because we can not get which device the noise should move
    to.

    Args:
        noise (torch.Tensor | callable | None): You can directly give a
            batch of noise through a ``torch.Tensor`` or offer a callable
            function to sample a batch of noise data. Otherwise, the
            ``None`` indicates to use the default noise sampler.
        grid_shape (torch.Size): Size of images in the diffusion process.
        num_timesteps (int, optional): Total timestpes of the diffusion and
            denoising process. Defaults to 0.
        num_batches (int, optional): The number of batch size. To be noted that
            this argument only work when the input ``noise`` is callable or
            ``None``. Defaults to 0.
        timesteps_noise (bool, optional): If True, returned noise will shape
            as [n, bz, c, d, h, w], otherwise shape as [bz, c, d, h, w].
            Defaults to False.
        device (str, optional): If not ``None``, move the generated noise to
            corresponding device.
    Returns:
        torch.Tensor: Generated noise with desired shape.
    """
    if isinstance(noise, torch.Tensor):
        # conduct sanity check for the last three dimension
        assert noise.shape[-4:] == grid_shape
        if timesteps_noise:
            if noise.ndim == 4:
                assert num_batches > 0 and num_timesteps > 0
                # noise shape as [n, c, d, h, w], expand to [n, bz, c, d, h, w]
                if noise.shape[0] == num_timesteps:
                    noise_batch = noise.view(num_timesteps, 1, *grid_shape)
                    noise_batch = noise_batch.expand(-1, num_batches, -1,
                                                     -1, -1, -1)
                # noise shape as [bz, c, d, h, w], expand to [n, bz, c, d, h, w]
                elif noise.shape[0] == num_batches:
                    noise_batch = noise.view(1, num_batches, *grid_shape)
                    noise_batch = noise_batch.expand(num_timesteps, -1,  -1,
                                                     -1, -1, -1)
                # noise shape as [n*bz, c, h, w], reshape to [n, bz, c, d, h, w]
                elif noise.shape[0] == num_timesteps * num_batches:
                    noise_batch = noise.view(num_timesteps, -1, *grid_shape)
                else:
                    raise ValueError(
                        'The timesteps noise should be in shape of '
                        '(n, c, d, h, w), (bz, c, d, h, w), (n*bz, c, d, h, w) or '
                        f'(n, bz, c, d, h, w). But receive {noise.shape}.')

            elif noise.ndim == 6:
                # direct return noise
                noise_batch = noise
            else:
                raise ValueError(
                    'The timesteps noise should be in shape of '
                    '(n, c, d, h, w), (bz, c, d, h, w), (n*bz, c, d, h, w) or '
                    f'(n, bz, c, d, h, w). But receive {noise.shape}.')
        else:
            if noise.ndim == 4:
                # reshape noise to [1, c, d, h, w]
                noise_batch = noise[None, ...]
            elif noise.ndim == 5:
                # do nothing
                noise_batch = noise
            else:
                raise ValueError(
                    'The noise should be in shape of (n, c, d, h, w) or'
                    f'(c, d, h, w), but got {noise.shape}')
    # receive a noise generator and sample noise.
    elif callable(noise):
        assert num_batches > 0
        noise_generator = noise
        if timesteps_noise:
            assert num_timesteps > 0
            # generate noise shape as [n, bz, c, d, h, w]
            noise_batch = noise_generator(
                (num_timesteps, num_batches, *grid_shape))
        else:
            # generate noise shape as [bz, c, d, h, w]
            noise_batch = noise_generator((num_batches, *grid_shape))
    # otherwise, we will adopt default noise sampler.
    else:
        assert num_batches > 0
        if timesteps_noise:
            assert num_timesteps > 0
            # generate noise shape as [n, bz, c, d, h, w]
            noise_batch = torch.randn(
                (num_timesteps, num_batches, *grid_shape))
        else:
            # generate noise shape as [bz, c, d, h, w]
            noise_batch = torch.randn((num_batches, *grid_shape))

    return noise_batch

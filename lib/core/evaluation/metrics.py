import pickle
import math
import numpy as np
import skimage
import torch
import mmcv

from torch.nn.functional import conv2d
from mmgen.core.registry import METRICS
from mmgen.core.evaluation.metrics import FID


def _get_gaussian_kernel1d(kernel_size: int, sigma: float, dtype=None, device=None):
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, device=device, dtype=dtype)
    pdf = torch.exp(-0.5 * (x / sigma).square())
    kernel1d = pdf / pdf.sum()
    return kernel1d


def _get_gaussian_kernel2d(kernel_size, sigma, dtype=None, device=None):
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0], dtype, device)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1], dtype, device)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


def filter_img_2d(img, kernel2d):
    batch_shape = img.shape[:-2]
    img = conv2d(img.reshape(batch_shape.numel(), 1, *img.shape[-2:]),
                 kernel2d[None, None])
    return img.reshape(*batch_shape, *img.shape[-2:])


def filter_img_2d_separate(img, kernel1d_x, kernel1d_y):
    batch_shape = img.shape[:-2]
    img = conv2d(conv2d(img.reshape(batch_shape.numel(), 1, *img.shape[-2:]),
                        kernel1d_x[None, None, None, :]),
                 kernel1d_y[None, None, :, None])
    return img.reshape(*batch_shape, *img.shape[-2:])


def gaussian_blur(img, kernel_size, sigma):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma), float(sigma)]
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=img.dtype, device=img.device)
    return filter_img_2d(img, kernel)


def eval_psnr(img1, img2, max_val=1.0, eps=1e-6):
    mse = (img1 - img2).square().flatten(1).mean(dim=-1)
    psnr = (10 * (2 * math.log10(max_val) - torch.log10(mse + eps)))
    return psnr


def eval_ssim_skimage(img1, img2, to_tensor=False, **kwargs):
    """Following pixelNeRF evaluation.

    Args:
        img1 (Tensor): Images with order "NCHW".
        img2 (Tensor): Images with order "NCHW".
    """
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    ssims = []
    for img1_, img2_ in zip(img1, img2):
        ssims.append(skimage.metrics.structural_similarity(
            img1_, img2_, channel_axis=0, **kwargs))
    return img1.new_tensor(ssims) if to_tensor else np.array(ssims, dtype=np.float32)


def eval_ssim(img1, img2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03,
              separate_filter=True):
    """Calculate SSIM (structural similarity) and contrast sensitivity using Gaussian kernel.

    Args:
        img1 (Tensor): Images with order "NCHW".
        img2 (Tensor): Images with order "NCHW".

    Returns:
        tuple: Pair containing the mean SSIM and contrast sensitivity between
        `img1` and `img2`.
    """
    assert img1.shape == img2.shape
    _, _, h, w = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, h, w)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        if separate_filter:
            window = _get_gaussian_kernel1d(size, sigma, dtype=img1.dtype, device=img1.device)
            mu1 = filter_img_2d_separate(img1, window, window)
            mu2 = filter_img_2d_separate(img2, window, window)
            sigma11 = filter_img_2d_separate(img1.square(), window, window)
            sigma22 = filter_img_2d_separate(img2.square(), window, window)
            sigma12 = filter_img_2d_separate(img1 * img2, window, window)
        else:
            window = _get_gaussian_kernel2d([size, size], [sigma, sigma], dtype=img1.dtype, device=img1.device)
            mu1 = filter_img_2d(img1, window)
            mu2 = filter_img_2d(img2, window)
            sigma11 = filter_img_2d(img1.square(), window)
            sigma22 = filter_img_2d(img2.square(), window)
            sigma12 = filter_img_2d(img1 * img2, window)
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1.square()
        sigma22 = img2.square()
        sigma12 = img1 * img2

    mu11 = mu1.square()
    mu22 = mu2.square()
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = torch.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)),
                      dim=(1, 2, 3))  # Return for each image individually.
    cs = torch.mean(v1 / v2, dim=(1, 2, 3))
    return ssim, cs


@METRICS.register_module()
class FIDKID(FID):
    name = 'FIDKID'

    def __init__(self,
                 num_images,
                 num_subsets=100,
                 max_subset_size=1000,
                 **kwargs):
        super().__init__(num_images, **kwargs)
        self.num_subsets = num_subsets
        self.max_subset_size = max_subset_size
        self.real_feats_np = None

    def prepare(self):
        if self.inception_pkl is not None and mmcv.is_filepath(
                self.inception_pkl):
            with open(self.inception_pkl, 'rb') as f:
                reference = pickle.load(f)
                self.real_mean = reference['mean']
                self.real_cov = reference['cov']
                self.real_feats_np = reference['feats_np']
                mmcv.print_log(
                    f'Load reference inception pkl from {self.inception_pkl}',
                    'mmgen')
            self.num_real_feeded = self.num_images

    @staticmethod
    def _calc_kid(real_feat, fake_feat, num_subsets, max_subset_size):
        """Refer to the implementation from:
        https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/kernel_inception_distance.py#L18  # noqa
        Args:
            real_feat (np.array): Features of the real samples.
            fake_feat (np.array): Features of the fake samples.
            num_subsets (int): Number of subsets to calculate KID.
            max_subset_size (int): The max size of each subset.
        Returns:
            float: The calculated kid metric.
        """
        n = real_feat.shape[1]
        m = min(min(real_feat.shape[0], fake_feat.shape[0]), max_subset_size)
        t = 0
        for _ in range(num_subsets):
            x = fake_feat[np.random.choice(
                fake_feat.shape[0], m, replace=False)]
            y = real_feat[np.random.choice(
                real_feat.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1)**3 + (y @ y.T / n + 1)**3
            b = (x @ y.T / n + 1)**3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m

        kid = t / num_subsets / m
        return float(kid)

    @torch.no_grad()
    def summary(self):
        if self.real_feats_np is None:
            feats = torch.cat(self.real_feats, dim=0)
            assert feats.shape[0] >= self.num_images
            feats = feats[:self.num_images]
            feats_np = feats.numpy()
            self.real_feats_np = feats_np
            self.real_mean = np.mean(feats_np, 0)
            self.real_cov = np.cov(feats_np, rowvar=False)

        fake_feats = torch.cat(self.fake_feats, dim=0)
        assert fake_feats.shape[0] >= self.num_images
        fake_feats = fake_feats[:self.num_images]
        fake_feats_np = fake_feats.numpy()
        fake_mean = np.mean(fake_feats_np, 0)
        fake_cov = np.cov(fake_feats_np, rowvar=False)

        fid, mean, cov = self._calc_fid(fake_mean, fake_cov, self.real_mean,
                                        self.real_cov)
        kid = self._calc_kid(self.real_feats_np, fake_feats_np, self.num_subsets,
                             self.max_subset_size) * 1000

        self._result_str = f'{fid:.4f} ({mean:.5f}/{cov:.5f}), {kid:.4f}'
        self._result_dict = dict(fid=fid, fid_mean=mean, fid_cov=cov, kid=kid)

        return fid, mean, cov, kid

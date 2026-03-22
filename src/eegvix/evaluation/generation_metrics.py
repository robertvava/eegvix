"""Generation quality metrics: FID, SSIM, LPIPS."""

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure


def compute_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    feature_dim: int = 2048,
) -> float:
    """Compute Frechet Inception Distance between real and generated images.

    Args:
        real_images: (n, 3, h, w) real images in [0, 255] uint8
        generated_images: (n, 3, h, w) generated images in [0, 255] uint8
        feature_dim: InceptionV3 feature dimension

    Returns:
        FID score (lower is better)
    """
    fid = FrechetInceptionDistance(feature=feature_dim, normalize=True)
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)
    return fid.compute().item()


def compute_ssim(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
) -> float:
    """Compute Structural Similarity Index between paired images.

    Args:
        real_images: (n, 3, h, w) in [0, 1]
        generated_images: (n, 3, h, w) in [0, 1]

    Returns:
        Mean SSIM (higher is better)
    """
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    return ssim(generated_images, real_images).item()


def compute_lpips(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    net: str = "alex",
) -> float:
    """Compute Learned Perceptual Image Patch Similarity.

    Args:
        real_images: (n, 3, h, w) in [0, 1]
        generated_images: (n, 3, h, w) in [0, 1]
        net: Backbone network ("alex", "vgg", "squeeze")

    Returns:
        Mean LPIPS distance (lower is better)
    """
    import lpips

    loss_fn = lpips.LPIPS(net=net)
    loss_fn.eval()

    # LPIPS expects [-1, 1] range
    real_scaled = real_images * 2 - 1
    gen_scaled = generated_images * 2 - 1

    with torch.no_grad():
        distances = loss_fn(gen_scaled, real_scaled)

    return distances.mean().item()

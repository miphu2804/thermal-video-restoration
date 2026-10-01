"""Full-reference image quality metrics (proposal 3.6): MSE, PSNR, SSIM.

All functions accept a single frame ``(H, W)`` or a sequence ``(T, H, W)`` on the
8-bit-equivalent scale (0-255). MSE and PSNR are computed over the whole array;
SSIM is computed per frame and averaged.
"""

import numpy as np
from skimage.metrics import structural_similarity

DATA_RANGE = 255.0


def _check_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected (H, W) or (T, H, W), got shape {x.shape}")
    return x, y


def mse(x: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error over the whole array.

    Args:
        x: Reference frame(s), shape ``(H, W)`` or ``(T, H, W)``.
        y: Test frame(s), same shape as ``x``.

    Returns:
        Mean squared error.

    Raises:
        ValueError: If shapes differ or are not 2D/3D.
    """
    x, y = _check_pair(x, y)
    return float(np.mean((x - y) ** 2))


def psnr(x: np.ndarray, y: np.ndarray, data_range: float = DATA_RANGE) -> float:
    """Peak signal-to-noise ratio in dB, from the MSE over the whole array.

    Args:
        x: Reference frame(s), shape ``(H, W)`` or ``(T, H, W)``.
        y: Test frame(s), same shape as ``x``.
        data_range: Peak value range of the signal.

    Returns:
        PSNR in dB; ``inf`` if ``x`` and ``y`` are identical.
    """
    err = mse(x, y)
    if err == 0:
        return float("inf")
    return float(10 * np.log10(data_range**2 / err))


def ssim(x: np.ndarray, y: np.ndarray, data_range: float = DATA_RANGE) -> float:
    """Structural similarity, averaged over frames for sequences.

    Args:
        x: Reference frame(s), shape ``(H, W)`` or ``(T, H, W)``.
        y: Test frame(s), same shape as ``x``.
        data_range: Peak value range of the signal.

    Returns:
        Mean SSIM (1.0 for identical inputs).
    """
    x, y = _check_pair(x, y)
    if x.ndim == 2:
        return float(structural_similarity(x, y, data_range=data_range))
    return float(
        np.mean(
            [structural_similarity(a, b, data_range=data_range) for a, b in zip(x, y)]
        )
    )

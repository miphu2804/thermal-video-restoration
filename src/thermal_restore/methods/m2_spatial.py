"""M2: spatial filtering with Gaussian, median and column (destriping) filters.

All filters work within each frame and never mix frames.
"""

import numpy as np
from scipy import ndimage

from thermal_restore.methods.registry import register


@register("gaussian")
def gaussian(frames: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smoothing in the spatial axes.

    Args:
        frames: Sequence of shape ``(T, H, W)``.
        sigma: Standard deviation of the Gaussian kernel, in pixels.

    Returns:
        Smoothed frames, float32, same shape.
    """
    frames = np.asarray(frames, dtype=np.float32)
    return ndimage.gaussian_filter(frames, sigma=(0, sigma, sigma))


@register("median")
def median(frames: np.ndarray, size: int = 3) -> np.ndarray:
    """Median filter in the spatial axes.

    Args:
        frames: Sequence of shape ``(T, H, W)``.
        size: Side length of the square window, in pixels.

    Returns:
        Filtered frames, float32, same shape.
    """
    frames = np.asarray(frames, dtype=np.float32)
    return ndimage.median_filter(frames, size=(1, size, size))


@register("column")
def column(frames: np.ndarray, size: int = 15) -> np.ndarray:
    """Remove column stripes by subtracting a per-column offset in each frame.

    The offset of each column is its mean minus a median-smoothed version of the
    column means along the horizontal axis. Estimating it per frame handles both
    fixed and temporal stripes (proposal 3.3, steps 3a and 3b).

    Args:
        frames: Sequence of shape ``(T, H, W)``.
        size: Odd window size, in columns, of the median used to smooth column
            means. Larger windows keep more real horizontal structure.

    Returns:
        Destriped frames, float32, same shape.

    Raises:
        ValueError: If ``size`` is not a positive odd integer.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError(f"size must be a positive odd integer, got {size}")
    frames = np.asarray(frames, dtype=np.float32)
    col_mean = frames.mean(axis=1)
    # Odd (point) reflection keeps linear trends at the borders, where an even
    # reflection would bias the median and leave false offsets.
    r = size // 2
    padded = np.pad(col_mean, ((0, 0), (r, r)), mode="reflect", reflect_type="odd")
    smooth = ndimage.median_filter(padded, size=(1, size))[:, r : r + col_mean.shape[1]]
    offset = col_mean - smooth
    return frames - offset[:, None, :]

"""Atomic degradation operators for thermal video sequences.

Every function is pure, operates on float32 arrays of shape ``(T, H, W)``,
and preserves the input shape and dtype.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def check_frames(frames: np.ndarray) -> np.ndarray:
    """Validate and convert input frames to float32 3D array.

    Args:
        frames: Sequence array expected to have shape ``(T, H, W)``.

    Returns:
        Frames as float32 array of shape ``(T, H, W)``.

    Raises:
        ValueError: If input array does not have exactly 3 dimensions.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError(
            f"Expected 3D array with shape (T, H, W), got shape {frames.shape} with {frames.ndim} dims"
        )
    return frames


def compress_contrast(frames: np.ndarray, factor: float) -> np.ndarray:
    """Compress dynamic range around the spatial mean pixel value (proposal 3.3, step 1).

    Scales deviations from the spatial mean by `factor`:
        out = mean + factor * (frames - mean)

    Args:
        frames: Sequence of shape ``(T, H, W)`` on the 0-255 scale.
        factor: Compression ratio in ``(0, 1]`` (e.g. 0.60, 0.40, 0.25).

    Returns:
        Contrast-compressed frames, float32, same shape.

    Raises:
        ValueError: If factor is not in ``(0, 1]``.
    """
    if factor <= 0.0 or factor > 1.0:
        raise ValueError(f"factor must be in (0, 1], got {factor}")
    frames = check_frames(frames)
    if factor == 1.0:
        return frames.copy()

    mean = frames.mean(axis=(-2, -1), keepdims=True)
    return (mean + factor * (frames - mean)).astype(np.float32)


def gaussian_blur(frames: np.ndarray, sigma: float) -> np.ndarray:
    """Apply spatial Gaussian blur frame-by-frame (proposal 3.3, step 2).

    Filters each frame independently across spatial dimensions (H, W) without
    mixing across the temporal dimension T.

    Args:
        frames: Sequence of shape ``(T, H, W)``.
        sigma: Standard deviation of the Gaussian kernel, in pixels.

    Returns:
        Blurred frames, float32, same shape.

    Raises:
        ValueError: If sigma is negative.
    """
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    frames = check_frames(frames)
    if sigma == 0.0:
        return frames.copy()

    return ndimage.gaussian_filter(frames, sigma=(0, sigma, sigma)).astype(np.float32)


def fixed_stripes(
    frames: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add fixed pattern noise (FPN) vertical stripes (proposal 3.3, step 3a).

    Samples an offset for each column from N(0, sigma^2) that remains identical
    across all rows and all frames in the sequence.

    Args:
        frames: Sequence of shape ``(T, H, W)``.
        sigma: Standard deviation of column stripe offsets.
        rng: Random number generator.

    Returns:
        Frames degraded with fixed vertical stripes, float32, same shape.

    Raises:
        ValueError: If sigma is negative.
    """
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    frames = check_frames(frames)
    if sigma == 0.0:
        return frames.copy()

    _, _, width = frames.shape
    col_offsets = rng.normal(loc=0.0, scale=sigma, size=(1, 1, width)).astype(
        np.float32
    )
    return (frames + col_offsets).astype(np.float32)


def temporal_stripes(
    frames: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add temporal vertical stripe offsets (proposal 3.3, step 3b).

    Samples a new offset for each column at each frame from N(0, sigma^2),
    modeling per-frame ADC/readout fluctuations.

    Args:
        frames: Sequence of shape ``(T, H, W)``.
        sigma: Standard deviation of temporal column stripe offsets.
        rng: Random number generator.

    Returns:
        Frames degraded with temporal vertical stripes, float32, same shape.

    Raises:
        ValueError: If sigma is negative.
    """
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    frames = check_frames(frames)
    if sigma == 0.0:
        return frames.copy()

    num_frames, _, width = frames.shape
    col_offsets = rng.normal(loc=0.0, scale=sigma, size=(num_frames, 1, width)).astype(
        np.float32
    )
    return (frames + col_offsets).astype(np.float32)


def gaussian_noise(
    frames: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add independent spatial-temporal Gaussian noise (proposal 3.3, step 4).

    Each pixel at each frame receives an independent draw from N(0, sigma^2).

    Args:
        frames: Sequence of shape ``(T, H, W)``.
        sigma: Standard deviation of Gaussian noise.
        rng: Random number generator.

    Returns:
        Frames degraded with additive Gaussian noise, float32, same shape.

    Raises:
        ValueError: If sigma is negative.
    """
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    frames = check_frames(frames)
    if sigma == 0.0:
        return frames.copy()

    noise = rng.normal(loc=0.0, scale=sigma, size=frames.shape).astype(np.float32)
    return (frames + noise).astype(np.float32)

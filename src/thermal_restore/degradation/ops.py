"""Atomic degradation operators for thermal video sequences.

Every function is pure, operates on float32 arrays of shape ``(T, H, W)``,
and preserves the input shape and dtype.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _check_frames(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError(
            f"Expected 3D array with shape (T, H, W), got shape {frames.shape} with {frames.ndim} dims"
        )
    return frames


def compress_contrast(frames: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 0.0 or factor > 1.0:
        raise ValueError(f"factor must be in (0, 1], got {factor}")
    frames = _check_frames(frames)
    if factor == 1.0:
        return frames.copy()

    mean = frames.mean(axis=(-2, -1), keepdims=True)
    return (mean + factor * (frames - mean)).astype(np.float32)


def gaussian_blur(frames: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    frames = _check_frames(frames)
    if sigma == 0.0:
        return frames.copy()

    return ndimage.gaussian_filter(frames, sigma=(0, sigma, sigma)).astype(np.float32)


def fixed_stripes(
    frames: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    frames = _check_frames(frames)
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
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    frames = _check_frames(frames)
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
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    frames = _check_frames(frames)
    if sigma == 0.0:
        return frames.copy()

    noise = rng.normal(loc=0.0, scale=sigma, size=frames.shape).astype(np.float32)
    return (frames + noise).astype(np.float32)

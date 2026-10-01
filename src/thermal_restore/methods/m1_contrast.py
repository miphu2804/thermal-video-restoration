"""M1: contrast enhancement with histogram equalization and CLAHE.

OpenCV needs uint8 input, so frames are clipped to 0-255 and rounded before
processing, then returned as float32.
"""

import cv2
import numpy as np

from thermal_restore.methods.registry import register


def _to_uint8(frames: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(frames), 0, 255).astype(np.uint8)


@register("hist_eq")
def hist_eq(frames: np.ndarray) -> np.ndarray:
    """Global histogram equalization, applied per frame.

    Args:
        frames: Sequence of shape ``(T, H, W)`` on the 0-255 scale.

    Returns:
        Equalized frames, float32, same shape.
    """
    out = [cv2.equalizeHist(f) for f in _to_uint8(frames)]
    return np.stack(out).astype(np.float32)


@register("clahe")
def clahe(
    frames: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization, applied per frame.

    Args:
        frames: Sequence of shape ``(T, H, W)`` on the 0-255 scale.
        clip_limit: Contrast limit for each tile.
        tile_grid_size: Number of tiles along each image axis.

    Returns:
        Enhanced frames, float32, same shape.
    """
    op = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
    )
    out = [op.apply(f) for f in _to_uint8(frames)]
    return np.stack(out).astype(np.float32)

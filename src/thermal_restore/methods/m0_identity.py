"""M0: no processing (baseline)."""

import numpy as np

from thermal_restore.methods.registry import register


@register("m0")
def identity(frames: np.ndarray) -> np.ndarray:
    """Return the frames unchanged (as a float32 copy).

    Args:
        frames: Sequence of shape ``(T, H, W)``.

    Returns:
        A float32 copy of the input frames.
    """
    return np.array(frames, dtype=np.float32)

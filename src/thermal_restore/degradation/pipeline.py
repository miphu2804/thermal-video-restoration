"""End-to-end degradation pipeline for thermal video sequences.

Composes atomic operations in the order defined by proposal section 3.3:
contrast compression -> Gaussian blur -> fixed pattern stripes ->
temporal stripes -> Gaussian noise.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from thermal_restore.degradation.ops import (
    _check_frames,
    compress_contrast,
    fixed_stripes,
    gaussian_blur,
    gaussian_noise,
    temporal_stripes,
)
from thermal_restore.degradation.presets import DEFAULT_STEPS, get_preset


def degrade(
    frames: np.ndarray,
    level: str | dict[str, float] = "medium",
    rng: np.random.Generator | int | None = None,
    steps: Sequence[str] | None = None,
    clip: bool = True,
) -> np.ndarray:
    frames = _check_frames(frames)
    params = get_preset(level)

    if steps is None:
        active_steps = list(DEFAULT_STEPS)
    else:
        active_steps = list(steps)
        for s in active_steps:
            if s not in DEFAULT_STEPS:
                raise ValueError(
                    f"Unknown degradation step '{s}'. Valid steps: {DEFAULT_STEPS}"
                )

    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))
    elif not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be np.random.Generator, int seed, or None, got {type(rng).__name__}"
        )

    out = frames.copy()

    for step_name in DEFAULT_STEPS:
        if step_name not in active_steps:
            continue

        if step_name == "contrast":
            out = compress_contrast(out, factor=float(params["contrast"]))
        elif step_name == "blur":
            out = gaussian_blur(out, sigma=float(params["blur"]))
        elif step_name == "fixed_stripes":
            out = fixed_stripes(out, sigma=float(params["fixed_stripes"]), rng=rng)
        elif step_name == "temporal_stripes":
            out = temporal_stripes(
                out, sigma=float(params["temporal_stripes"]), rng=rng
            )
        elif step_name == "noise":
            out = gaussian_noise(out, sigma=float(params["noise"]), rng=rng)

    if clip:
        out = np.clip(out, 0.0, 255.0)

    return out.astype(np.float32)

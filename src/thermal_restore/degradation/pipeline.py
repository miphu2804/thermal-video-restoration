"""End-to-end degradation pipeline for thermal video sequences.

Composes atomic operations in the order defined by proposal section 3.3:
contrast compression -> Gaussian blur -> fixed pattern stripes ->
temporal stripes -> Gaussian noise.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from thermal_restore.degradation.ops import (
    check_frames,
    compress_contrast,
    fixed_stripes,
    gaussian_blur,
    gaussian_noise,
    temporal_stripes,
)
from thermal_restore.degradation.presets import DEFAULT_STEPS, get_preset


def degrade(
    frames: np.ndarray,
    level: str | dict[str, float],
    rng: np.random.Generator | int,
    steps: Sequence[str] | None = None,
    clip: bool = True,
) -> np.ndarray:
    """Apply the proposal 3.3 degradation chain to a sequence.

    Args:
        frames: Clean sequence of shape ``(T, H, W)`` on the 0-255 scale.
        level: Preset name (``"light"``, ``"medium"``, ``"heavy"``) or dict of params.
        rng: Generator or integer seed for reproducible degradation.
        steps: Subset of ``DEFAULT_STEPS`` to apply; ``None`` applies all in order.
        clip: Clip the output to ``[0, 255]``.

    Returns:
        Degraded frames, float32, same shape.

    Raises:
        ValueError: If an unknown step name is specified.
        TypeError: If rng is not a Generator or integer seed.
    """
    frames = check_frames(frames)
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

    if isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))
    elif not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be np.random.Generator or int seed, got {type(rng).__name__}"
        )

    # Spawn independent RNG for each step so step noise is isolated (e.g. for E1)
    sub_rngs = dict(zip(DEFAULT_STEPS, rng.spawn(len(DEFAULT_STEPS))))

    out = frames.copy()

    for step_name in DEFAULT_STEPS:
        if step_name not in active_steps:
            continue

        if step_name == "contrast":
            out = compress_contrast(out, factor=float(params["contrast"]))
        elif step_name == "blur":
            out = gaussian_blur(out, sigma=float(params["blur"]))
        elif step_name == "fixed_stripes":
            out = fixed_stripes(
                out, sigma=float(params["fixed_stripes"]), rng=sub_rngs[step_name]
            )
        elif step_name == "temporal_stripes":
            out = temporal_stripes(
                out,
                sigma=float(params["temporal_stripes"]),
                rng=sub_rngs[step_name],
            )
        elif step_name == "noise":
            out = gaussian_noise(
                out, sigma=float(params["noise"]), rng=sub_rngs[step_name]
            )

    if clip:
        out = np.clip(out, 0.0, 255.0)

    return out.astype(np.float32)

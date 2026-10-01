"""Degradation parameter presets corresponding to proposal section 3.3.

Defines three severity levels: 'light', 'medium', and 'heavy'.
Parameters are specified on the 8-bit equivalent scale (0-255).
"""

from __future__ import annotations

# Proposal 3.3 table:
# Step 1: contrast: dynamic range compression ratio (60% / 40% / 25%)
# Step 2: blur: spatial Gaussian sigma in pixels (1 / 2 / 3)
# Step 3a: fixed_stripes: constant column stripe sigma (2 / 4 / 8)
# Step 3b: temporal_stripes: per-frame column stripe sigma (1 / 2 / 4)
# Step 4: noise: independent pixel Gaussian noise sigma (3 / 6 / 12)
LEVELS: dict[str, dict[str, float]] = {
    "light": {
        "contrast": 0.60,
        "blur": 1.0,
        "fixed_stripes": 2.0,
        "temporal_stripes": 1.0,
        "noise": 3.0,
    },
    "medium": {
        "contrast": 0.40,
        "blur": 2.0,
        "fixed_stripes": 4.0,
        "temporal_stripes": 2.0,
        "noise": 6.0,
    },
    "heavy": {
        "contrast": 0.25,
        "blur": 3.0,
        "fixed_stripes": 8.0,
        "temporal_stripes": 4.0,
        "noise": 12.0,
    },
}

DEFAULT_STEPS: tuple[str, ...] = (
    "contrast",
    "blur",
    "fixed_stripes",
    "temporal_stripes",
    "noise",
)


def get_preset(level: str | dict[str, float]) -> dict[str, float]:
    """Retrieve degradation parameters for a preset level or validate custom params.

    Args:
        level: Preset name (``"light"``, ``"medium"``, ``"heavy"``) or a dict of params
            containing all keys in ``DEFAULT_STEPS``.

    Returns:
        Dictionary mapping each degradation step name to its parameter value.

    Raises:
        KeyError: If level is an unknown preset string or a dict missing required keys.
        TypeError: If level is not a str or dict.
    """
    if isinstance(level, str):
        if level not in LEVELS:
            raise KeyError(
                f"Unknown degradation level '{level}'. Available levels: {list(LEVELS.keys())}"
            )
        return dict(LEVELS[level])
    if isinstance(level, dict):
        missing = set(DEFAULT_STEPS) - level.keys()
        if missing:
            raise KeyError(f"Missing degradation parameters: {sorted(missing)}")
        return dict(level)
    raise TypeError(f"level must be str or dict, got {type(level).__name__}")

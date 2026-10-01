"""Unit tests for thermal degradation module."""

from __future__ import annotations

import numpy as np
import pytest

from thermal_restore.degradation import (
    DEFAULT_STEPS,
    LEVELS,
    compress_contrast,
    degrade,
    fixed_stripes,
    gaussian_blur,
    gaussian_noise,
    get_preset,
    temporal_stripes,
)


@pytest.fixture
def sample_sequence() -> np.ndarray:
    """Create a synthetic 3D video sequence (T=5, H=32, W=40)."""
    rng = np.random.default_rng(123)
    # Range roughly 50 to 200 (within 0-255 scale)
    base = rng.uniform(50.0, 200.0, size=(1, 32, 40)).astype(np.float32)
    noise = rng.normal(0.0, 2.0, size=(5, 32, 40)).astype(np.float32)
    return base + noise


# ---------------------------------------------------------------------------
# Ops tests
# ---------------------------------------------------------------------------


def test_invalid_shape_rejected() -> None:
    rng = np.random.default_rng(0)
    arr_2d = np.ones((16, 16), dtype=np.float32)
    arr_4d = np.ones((2, 3, 16, 16), dtype=np.float32)

    for invalid in (arr_2d, arr_4d):
        with pytest.raises(ValueError, match="Expected 3D array"):
            compress_contrast(invalid, 0.5)
        with pytest.raises(ValueError, match="Expected 3D array"):
            gaussian_blur(invalid, 1.0)
        with pytest.raises(ValueError, match="Expected 3D array"):
            fixed_stripes(invalid, 2.0, rng)
        with pytest.raises(ValueError, match="Expected 3D array"):
            temporal_stripes(invalid, 2.0, rng)
        with pytest.raises(ValueError, match="Expected 3D array"):
            gaussian_noise(invalid, 2.0, rng)
        with pytest.raises(ValueError, match="Expected 3D array"):
            degrade(invalid, "light", rng=0)


def test_compress_contrast(sample_sequence: np.ndarray) -> None:
    # Factor = 1.0 should be an identity copy
    ident = compress_contrast(sample_sequence, factor=1.0)
    assert np.allclose(ident, sample_sequence)

    # Factor = 0.5 should reduce variance and preserve per-frame mean
    compressed = compress_contrast(sample_sequence, factor=0.5)
    assert compressed.shape == sample_sequence.shape
    assert compressed.dtype == np.float32
    for t in range(len(sample_sequence)):
        assert np.isclose(compressed[t].mean(), sample_sequence[t].mean(), atol=1e-4)
        assert compressed[t].std() < sample_sequence[t].std()

    # Invalid factors
    with pytest.raises(ValueError):
        compress_contrast(sample_sequence, factor=0.0)
    with pytest.raises(ValueError):
        compress_contrast(sample_sequence, factor=1.2)


def test_gaussian_blur(sample_sequence: np.ndarray) -> None:
    # Sigma = 0 should be an identity copy
    ident = gaussian_blur(sample_sequence, sigma=0.0)
    assert np.allclose(ident, sample_sequence)

    # Sigma > 0 should smooth each frame
    blurred = gaussian_blur(sample_sequence, sigma=2.0)
    assert blurred.shape == sample_sequence.shape
    assert blurred.dtype == np.float32

    # Blurred frames should have smaller high-frequency differences (Laplacian/gradient)
    grad_orig = np.diff(sample_sequence, axis=-1)
    grad_blur = np.diff(blurred, axis=-1)
    assert grad_blur.std() < grad_orig.std()

    with pytest.raises(ValueError):
        gaussian_blur(sample_sequence, sigma=-1.0)


def test_fixed_stripes_is_constant_across_time(sample_sequence: np.ndarray) -> None:
    rng = np.random.default_rng(42)
    striped = fixed_stripes(sample_sequence, sigma=5.0, rng=rng)

    assert striped.shape == sample_sequence.shape
    assert striped.dtype == np.float32

    diff = striped - sample_sequence  # shape (T, H, W)
    # The stripe offset must be strictly constant along axis 0 (time) and axis 1 (height)
    for t in range(1, len(sample_sequence)):
        assert np.allclose(diff[t], diff[0], atol=1e-5)
    # Along rows within the same frame, offsets must be constant
    for h in range(1, sample_sequence.shape[1]):
        assert np.allclose(diff[0, h, :], diff[0, 0, :], atol=1e-5)


def test_temporal_stripes_varies_across_time(sample_sequence: np.ndarray) -> None:
    rng = np.random.default_rng(42)
    striped = temporal_stripes(sample_sequence, sigma=5.0, rng=rng)

    assert striped.shape == sample_sequence.shape
    diff = striped - sample_sequence  # shape (T, H, W)

    # Within each frame, offset is vertical (constant along height)
    for t in range(len(sample_sequence)):
        for h in range(1, sample_sequence.shape[1]):
            assert np.allclose(diff[t, h, :], diff[t, 0, :], atol=1e-5)

    # Across time frames, offsets must NOT be identical
    assert not np.allclose(diff[0], diff[1])


def test_gaussian_noise(sample_sequence: np.ndarray) -> None:
    rng = np.random.default_rng(42)
    sigma = 4.0
    noisy = gaussian_noise(sample_sequence, sigma=sigma, rng=rng)

    assert noisy.shape == sample_sequence.shape
    assert noisy.dtype == np.float32

    noise = noisy - sample_sequence
    assert abs(noise.mean()) < 0.5
    assert abs(noise.std() - sigma) < 0.5


# ---------------------------------------------------------------------------
# Presets tests
# ---------------------------------------------------------------------------


def test_presets_match_proposal_3_3() -> None:
    assert DEFAULT_STEPS == (
        "contrast",
        "blur",
        "fixed_stripes",
        "temporal_stripes",
        "noise",
    )
    assert set(LEVELS.keys()) == {"light", "medium", "heavy"}

    # Check light level
    light = get_preset("light")
    assert light["contrast"] == 0.60
    assert light["blur"] == 1.0
    assert light["fixed_stripes"] == 2.0
    assert light["temporal_stripes"] == 1.0
    assert light["noise"] == 3.0

    # Check medium level
    medium = get_preset("medium")
    assert medium["contrast"] == 0.40
    assert medium["blur"] == 2.0
    assert medium["fixed_stripes"] == 4.0
    assert medium["temporal_stripes"] == 2.0
    assert medium["noise"] == 6.0

    # Check heavy level
    heavy = get_preset("heavy")
    assert heavy["contrast"] == 0.25
    assert heavy["blur"] == 3.0
    assert heavy["fixed_stripes"] == 8.0
    assert heavy["temporal_stripes"] == 4.0
    assert heavy["noise"] == 12.0

    with pytest.raises(KeyError):
        get_preset("extreme")


def test_presets_missing_keys_rejected() -> None:
    with pytest.raises(KeyError, match="Missing degradation parameters"):
        get_preset({"blur": 1.0})


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


def test_degrade_levels(sample_sequence: np.ndarray) -> None:
    for level_name in ("light", "medium", "heavy"):
        out = degrade(sample_sequence, level=level_name, rng=10)
        assert out.shape == sample_sequence.shape
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 255.0


def test_degrade_reproducibility(sample_sequence: np.ndarray) -> None:
    out1 = degrade(sample_sequence, level="medium", rng=99)
    out2 = degrade(sample_sequence, level="medium", rng=99)
    assert np.array_equal(out1, out2)

    # Different seeds yield different outputs
    out3 = degrade(sample_sequence, level="medium", rng=100)
    assert not np.array_equal(out1, out3)


def test_degrade_isolated_steps(sample_sequence: np.ndarray) -> None:
    # Only blur: should be deterministic and noise-free
    blurred1 = degrade(sample_sequence, level="medium", rng=1, steps=["blur"])
    blurred2 = degrade(sample_sequence, level="medium", rng=2, steps=["blur"])
    assert np.array_equal(blurred1, blurred2)

    # Only fixed stripes: difference across time frames must be identical
    striped = degrade(
        sample_sequence, level="medium", rng=1, steps=["fixed_stripes"], clip=False
    )
    diff = striped - sample_sequence
    assert np.allclose(diff[0], diff[1], atol=1e-5)

    # Invalid step name
    with pytest.raises(ValueError, match="Unknown degradation step"):
        degrade(sample_sequence, level="medium", rng=0, steps=["unknown_step"])


def test_degrade_step_isolation_rng_spawn(sample_sequence: np.ndarray) -> None:
    """Same seed yields identical noise component whether earlier steps are enabled or not."""
    # Run with only noise
    out_noise = degrade(
        sample_sequence, level="medium", rng=42, steps=["noise"], clip=False
    )
    noise_alone = out_noise - sample_sequence

    # Run with fixed_stripes and noise
    out_both = degrade(
        sample_sequence,
        level="medium",
        rng=42,
        steps=["fixed_stripes", "noise"],
        clip=False,
    )
    # Run with only fixed stripes with the exact same seed
    out_stripes = degrade(
        sample_sequence,
        level="medium",
        rng=42,
        steps=["fixed_stripes"],
        clip=False,
    )
    stripes_alone = out_stripes - sample_sequence

    # The noise component in out_both (out_both - sample_sequence - stripes_alone)
    # must exactly equal noise_alone because of rng.spawn()
    extracted_noise = out_both - sample_sequence - stripes_alone
    assert np.allclose(extracted_noise, noise_alone, atol=1e-5)


def test_degrade_custom_dict(sample_sequence: np.ndarray) -> None:
    custom = {
        "contrast": 0.8,
        "blur": 0.5,
        "fixed_stripes": 1.0,
        "temporal_stripes": 0.5,
        "noise": 1.0,
    }
    out = degrade(sample_sequence, level=custom, rng=42)
    assert out.shape == sample_sequence.shape
    assert out.dtype == np.float32

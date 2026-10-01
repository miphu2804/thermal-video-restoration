import numpy as np
import pytest

from thermal_restore.metrics import mse, psnr, ssim


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    return rng.uniform(0, 255, size=(3, 48, 64)).astype(np.float32)


def test_identical_inputs(frames):
    assert mse(frames, frames) == 0
    assert psnr(frames, frames) == float("inf")
    assert ssim(frames, frames) == pytest.approx(1.0)


def test_known_psnr(frames):
    shifted = frames + 1
    assert mse(frames, shifted) == pytest.approx(1.0)
    assert psnr(frames, shifted) == pytest.approx(20 * np.log10(255))


def test_noise_lowers_quality(frames):
    rng = np.random.default_rng(1)
    light = frames + rng.normal(0, 3, frames.shape)
    heavy = frames + rng.normal(0, 12, frames.shape)
    assert psnr(frames, light) > psnr(frames, heavy)
    assert ssim(frames, light) > ssim(frames, heavy)


def test_ssim_single_frame(frames):
    assert ssim(frames[0], frames[0]) == pytest.approx(1.0)


def test_shape_mismatch_raises(frames):
    with pytest.raises(ValueError, match="Shape mismatch"):
        mse(frames, frames[:2])

import numpy as np
import pytest

from thermal_restore.methods import available_methods, get_method, register
from thermal_restore.methods.registry import _REGISTRY
from thermal_restore.metrics import mse

NAMES = ["clahe", "column", "gaussian", "hist_eq", "m0", "median"]


@pytest.fixture
def clean():
    """Smooth synthetic sequence: horizontal gradient plus a bright blob."""
    t, h, w = 4, 32, 40
    yy, xx = np.mgrid[0:h, 0:w]
    frame = 60 + 100 * xx / w + 60 * np.exp(-((yy - 16) ** 2 + (xx - 20) ** 2) / 50)
    return np.repeat(frame[None], t, axis=0).astype(np.float32)


def test_available_methods():
    assert set(NAMES) <= set(available_methods())


def test_m0_returns_input(clean):
    assert np.array_equal(get_method("m0")(clean), clean)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="'nope'.*m0"):
        get_method("nope")


def test_duplicate_register_raises():
    name = "_test_dup"
    try:
        register(name)(lambda frames: frames)
        with pytest.raises(ValueError, match="already registered"):
            register(name)(lambda frames: frames)
    finally:
        _REGISTRY.pop(name, None)


@pytest.mark.parametrize("name", NAMES)
def test_output_contract(name, clean):
    out = get_method(name)(clean)
    assert out.shape == clean.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("name", ["gaussian", "median"])
def test_denoisers_reduce_noise(name, clean):
    rng = np.random.default_rng(0)
    noisy = (clean + rng.normal(0, 10, clean.shape)).astype(np.float32)
    assert mse(clean, get_method(name)(noisy)) < mse(clean, noisy)


def test_params_are_passed(clean):
    rng = np.random.default_rng(0)
    noisy = (clean + rng.normal(0, 10, clean.shape)).astype(np.float32)
    weak = get_method("gaussian", sigma=0.5)(noisy)
    strong = get_method("gaussian", sigma=2.0)(noisy)
    assert weak.std() > strong.std()


def test_column_removes_stripes(clean):
    rng = np.random.default_rng(0)
    stripes = rng.normal(0, 8, clean.shape[2]).astype(np.float32)
    striped = clean + stripes[None, None, :]
    restored = get_method("column")(striped)
    assert mse(clean, restored) < 0.5 * mse(clean, striped)


@pytest.mark.parametrize("name", ["hist_eq", "clahe"])
def test_contrast_methods_widen_range(name, clean):
    low = (clean - clean.mean()) * 0.25 + clean.mean()
    out = get_method(name)(low)
    assert np.ptp(out) > np.ptp(low)


def test_invalid_param_raises():
    with pytest.raises(ValueError, match="Invalid parameters.*gaussian"):
        get_method("gaussian", sigmaa=2)


def test_m0_invalid_param_raises():
    with pytest.raises(ValueError):
        get_method("m0", foo=1)


def test_non_3d_input_raises():
    arr_2d = np.zeros((32, 40), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\(T, H, W\)"):
        get_method("gaussian")(arr_2d)


@pytest.mark.parametrize("size", [0, 2, -3])
def test_column_invalid_size_raises(size, clean):
    with pytest.raises(ValueError, match="odd"):
        get_method("column", size=size)(clean)


def test_column_size_one_noop(clean):
    assert np.array_equal(get_method("column", size=1)(clean), clean)

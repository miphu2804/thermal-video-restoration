import numpy as np
import pytest

from thermal_restore.experiments import range_remapping_stages
from thermal_restore.methods import get_method


@pytest.fixture
def sample_frames():
    yy, xx = np.mgrid[0:32, 0:40]
    frame = 40 + 2 * xx + 10 * np.exp(-((yy - 16) ** 2 + (xx - 20) ** 2) / 40)
    return np.stack([frame, frame + 3]).astype(np.float32)


@pytest.mark.parametrize(
    "name",
    [
        "linear",
        "fshs",
        "log",
        "mean",
        "geometric",
        "harmonic",
        "contraharmonic",
        "fourier_lowpass",
        "fourier_notch",
        "wavelet_soft",
    ],
)
def test_preprocessing_output_contract(name, sample_frames):
    params = {
        "fshs": {"low_percentile": 1.0, "high_percentile": 99.0},
        "mean": {"size": 3},
        "geometric": {"size": 3},
        "harmonic": {"size": 3},
        "contraharmonic": {"size": 3},
        "fourier_lowpass": {"cutoff": 0.15},
        "fourier_notch": {"bandwidth": 0.005, "preserve": 0.01},
        "wavelet_soft": {"level": 1},
    }.get(name, {})
    output = get_method(name, **params)(sample_frames)
    assert output.shape == sample_frames.shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert output.min() >= 0
    assert output.max() <= 255


def test_linear_point_operation(sample_frames):
    output = get_method("linear", gain=0.5, offset=10)(sample_frames)
    np.testing.assert_allclose(output, np.clip(sample_frames * 0.5 + 10, 0, 255))


def test_full_scale_stretch_expands_range(sample_frames):
    output = get_method("fshs")(sample_frames)
    assert np.isclose(output.min(), 0)
    assert np.isclose(output.max(), 255)


def test_robust_full_scale_stretch(sample_frames):
    output = get_method("fshs", low_percentile=1.0, high_percentile=99.0)(sample_frames)
    assert output.shape == sample_frames.shape
    assert np.isfinite(output).all()


@pytest.mark.parametrize("name", ["mean", "geometric", "harmonic", "contraharmonic"])
def test_spatial_filters_reject_even_window(name, sample_frames):
    with pytest.raises(ValueError, match="odd"):
        get_method(name, size=2)(sample_frames)


def test_fourier_parameters_are_validated(sample_frames):
    with pytest.raises(ValueError, match="cutoff"):
        get_method("fourier_lowpass", cutoff=0)(sample_frames)
    with pytest.raises(ValueError, match="axis"):
        get_method("fourier_notch", axis="z")(sample_frames)


def test_full_scale_stretch_honours_output_range(sample_frames):
    output = get_method("fshs", out_low=100.0, out_high=126.5)(sample_frames)
    assert np.isclose(output.min(), 100.0)
    assert np.isclose(output.max(), 126.5)
    logged = get_method("log", out_low=100.0, out_high=126.5)(sample_frames)
    assert logged.min() >= 100.0 and logged.max() <= 126.5
    with pytest.raises(ValueError, match="output range"):
        get_method("fshs", out_low=200.0, out_high=100.0)(sample_frames)


def test_range_remapping_stages_flags_only_full_range_stretches():
    assert range_remapping_stages(
        [{"name": "median", "params": {"size": 3}}, {"name": "clahe"}]
    ) == ["clahe"]
    assert range_remapping_stages([{"name": "fshs"}, {"name": "log"}]) == [
        "fshs",
        "log",
    ]
    assert not range_remapping_stages(
        [{"name": "fshs", "params": {"out_low": 100.0, "out_high": 126.5}}]
    )
    assert not range_remapping_stages([{"name": "wavelet_soft"}])


def test_fourier_notch_removes_column_stripes_and_keeps_scene():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:128, 0:128]
    scene = 110 + 10 * np.exp(-((yy - 40) ** 2 + (xx - 70) ** 2) / 50)
    scene = (scene + rng.normal(0, 2, scene.shape))[None].astype(np.float32)
    stripes = np.tile(rng.normal(0, 3, 128), (128, 1))[None].astype(np.float32)
    notch = get_method("fourier_notch")
    residual = notch(scene + stripes) - notch(scene)
    assert residual.std() < 0.15 * stripes.std()
    assert np.abs(notch(scene) - scene).std() < 0.2
    rows = get_method("fourier_notch", axis="y")(
        np.transpose(scene + stripes, (0, 2, 1))
    )
    np.testing.assert_allclose(
        rows, np.transpose(notch(scene + stripes), (0, 2, 1)), atol=1e-3
    )


def test_wavelet_noise_estimate_ignores_scene_edges():
    yy, xx = np.mgrid[0:64, 0:64]
    edges = (100 + 30 * ((xx // 8 + yy // 8) % 2))[None].astype(np.float32)
    output = get_method("wavelet_soft")(edges)
    np.testing.assert_allclose(output, edges, atol=1e-3)


def test_wavelet_handles_zero_noise_estimate():
    flat = np.zeros((1, 32, 32), dtype=np.float32)
    np.testing.assert_array_equal(get_method("wavelet_soft")(flat), flat)
    quantized = np.full((1, 32, 32), 80, dtype=np.float32)
    quantized[0, 8:24, 8:24] = 120
    output = get_method("wavelet_soft")(quantized)
    assert np.isfinite(output).all()

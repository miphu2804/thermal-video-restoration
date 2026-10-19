from pathlib import Path

import numpy as np
import pytest

from thermal_restore.data.experiment_config import (
    load_restoration_config,
    source_sequence,
)
from thermal_restore.degradation import degrade
from thermal_restore.detection.live import render_comparison
from thermal_restore.experiments import QualityTotals, build_restoration, degrade_frame


def test_streamed_fixed_pattern_is_constant_per_sequence():
    source = np.full((16, 20), 120, dtype=np.float32)
    first = degrade_frame(
        source, sequence_key="s", frame_key="1", seed=3, steps=["fixed_stripes"]
    )
    second = degrade_frame(
        source, sequence_key="s", frame_key="2", seed=3, steps=["fixed_stripes"]
    )
    np.testing.assert_array_equal(first, second)
    other = degrade_frame(
        source, sequence_key="other", frame_key="1", seed=3, steps=["fixed_stripes"]
    )
    assert not np.array_equal(first, other)


def test_streamed_frame_noise_is_reproducible_and_order_independent():
    source = np.full((16, 20), 120, dtype=np.float32)
    expected = degrade_frame(source, sequence_key="s", frame_key="1", seed=4)
    different = degrade_frame(source, sequence_key="s", frame_key="2", seed=4)
    actual = degrade_frame(source, sequence_key="s", frame_key="1", seed=4)
    np.testing.assert_array_equal(expected, actual)
    assert not np.array_equal(expected, different)
    assert not np.array_equal(
        expected, degrade_frame(source, sequence_key="s", frame_key="1", seed=5)
    )
    assert expected.dtype == np.float32 and expected.shape == source.shape
    assert expected.min() >= 0 and expected.max() <= 255
    np.testing.assert_array_equal(source, np.full((16, 20), 120, dtype=np.float32))


def test_streamed_fixed_override_respects_pipeline_order_and_shape():
    frames = np.full((2, 10, 12), 120, dtype=np.float32)
    offsets = np.arange(12, dtype=np.float32).reshape(1, 1, 12)
    out = degrade(frames, "medium", 2, steps=["fixed_stripes"], fixed_offsets=offsets)
    np.testing.assert_array_equal(out, frames + offsets)
    np.testing.assert_array_equal(
        degrade(frames, "medium", 2, steps=[], fixed_offsets=offsets), frames
    )
    with pytest.raises(ValueError, match="fixed_offsets"):
        degrade(frames, "medium", 2, fixed_offsets=np.zeros((1, 12)))
    with pytest.raises(ValueError, match="fixed_offsets"):
        degrade(frames, "medium", 2, fixed_offsets=offsets * np.nan)


def test_restoration_identity_and_chain_preserve_inputs():
    source = np.arange(256, dtype=np.float32).reshape(16, 16)
    original = source.copy()
    np.testing.assert_array_equal(build_restoration([{"name": "m0"}])(source), source)
    chain = build_restoration(
        [{"name": "median", "params": {"size": 3}}, {"name": "clahe"}]
    )
    out = chain(source)
    assert out.shape == source.shape and out.dtype == np.float32
    np.testing.assert_array_equal(source, original)
    with pytest.raises(ValueError):
        build_restoration([])
    with pytest.raises(ValueError, match="Unknown method"):
        build_restoration([{"name": "missing"}])
    with pytest.raises(ValueError, match="Invalid parameters"):
        build_restoration([{"name": "median", "params": {"misspelled": 4}}])


def test_image_quality_totals_use_pixel_weighted_error_and_safe_infinity():
    totals = QualityTotals()
    reference = np.zeros((8, 8), dtype=np.float32)
    totals.update(reference, reference)
    assert totals.result()["psnr_infinite"]
    assert totals.result()["psnr_db"] is None
    totals.update(
        np.zeros((16, 16), dtype=np.float32), np.ones((16, 16), dtype=np.float32)
    )
    result = totals.result()
    assert result["mse"] == pytest.approx(256 / 320)
    assert result["psnr_db"] == pytest.approx(10 * np.log10(255**2 / result["mse"]))
    assert result["frames"] == 2 and result["pixels"] == 320
    with pytest.raises(ValueError, match="No image"):
        QualityTotals().result()


def test_shared_display_range_preserves_visible_contrast_loss():
    source = np.tile(np.linspace(100, 130, 100, dtype=np.float32), (100, 1))
    degraded = 115 + 0.25 * (source - 115)
    image = render_comparison(
        [source, degraded, degraded],
        [[], [], []],
        {},
        labels=["A", "B", "C"],
        panel_width=100,
    )
    assert image.shape == (134, 300, 3)
    assert image[34:, 100:200, 0].std() < image[34:, :100, 0].std() * 0.4
    np.testing.assert_array_equal(image[34:, 100:200], image[34:, 200:300])


def test_config_validation_and_flir_sequence_identity(tmp_path):
    config = load_restoration_config(Path("configs/abc_baselines.json"))
    assert "m0" in config["pipelines"]
    assert source_sequence(Path("video-a-frame-123-x.tiff")) == "video-a"
    path = tmp_path / "bad.json"
    path.write_text(
        '{"version":1,"default_pipeline":"oops","pipelines":{"m0":[{"name":"m0"}]}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="default_pipeline"):
        load_restoration_config(path)

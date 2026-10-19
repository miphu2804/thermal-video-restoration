from pathlib import Path

import numpy as np
import pytest
import tifffile

from thermal_restore.data import stream
from thermal_restore.data.stream import (
    capture_to_thermal,
    iter_tiff_frames,
    select_tiff_paths,
)
from thermal_restore.detection import Detection
from thermal_restore.detection.live import (
    ppm_bytes,
    render_detections,
    render_preprocessing_comparison,
)
from thermal_restore.detection.torchvision_backend import (
    normalization_for_dtype,
    normalize_detector_frame,
)


def test_tiff_selection_preserves_sequence_and_numeric_order(tmp_path):
    for name in (
        "video-b-frame-10-x.tiff",
        "video-a-frame-2-x.tiff",
        "video-a-frame-1-x.tiff",
    ):
        (tmp_path / name).touch()
    with pytest.raises(ValueError, match="Select --sequence"):
        select_tiff_paths(tmp_path)
    selected = select_tiff_paths(tmp_path, "video-a")
    assert [path.name for path in selected] == [
        "video-a-frame-1-x.tiff",
        "video-a-frame-2-x.tiff",
    ]
    with pytest.raises(ValueError, match="Unknown sequence"):
        select_tiff_paths(tmp_path, "missing")


def test_generic_tiff_selection_natural_sort_and_empty_source(tmp_path):
    with pytest.raises(ValueError, match="No TIFF"):
        select_tiff_paths(tmp_path)
    for name in ("frame10.TIFF", "frame2.tif", "ignored.png"):
        (tmp_path / name).touch()
    assert [path.name for path in select_tiff_paths(tmp_path)] == [
        "frame2.tif",
        "frame10.TIFF",
    ]
    assert select_tiff_paths(tmp_path / "frame2.tif") == [tmp_path / "frame2.tif"]


def test_lazy_raw_tiff_load_keeps_scale_and_frame_identity(tmp_path):
    path = tmp_path / "video-a-frame-12-x.tiff"
    tifffile.imwrite(path, np.array([[0, 16383]], dtype=np.uint16))
    packet = next(iter_tiff_frames([path]))
    assert packet.index == 12
    assert packet.dtype == np.uint16
    assert packet.pixels.dtype == np.float32
    np.testing.assert_allclose(packet.pixels, [[0, 255]])


def test_uint8_stream_policy_matches_evaluation(tmp_path):
    path = tmp_path / "frame.tiff"
    tifffile.imwrite(path, np.array([[25, 200]], dtype=np.uint8))
    with pytest.raises(ValueError, match="uint16"):
        next(iter_tiff_frames([path]))
    packet = next(iter_tiff_frames([path], allow_uint8=True))
    mean, std = normalization_for_dtype(
        packet.dtype, mean=113.2, std=5.6, allow_uint8=True
    )
    assert mean != 113.2 and std != 5.6
    np.testing.assert_allclose(
        normalize_detector_frame(packet.pixels, mean=mean, std=std),
        (packet.pixels - mean) / std,
    )


def test_capture_requires_explicit_uint8_and_converts_gray_not_radiometry():
    gray = np.array([[0, 127, 255]], dtype=np.uint8)
    with pytest.raises(ValueError, match="allow-uint8"):
        capture_to_thermal(gray, allow_uint8=False)
    np.testing.assert_array_equal(capture_to_thermal(gray, allow_uint8=True), gray)
    bgr = np.repeat(gray[:, :, None], 3, axis=2)
    np.testing.assert_array_equal(capture_to_thermal(bgr, allow_uint8=True), gray)
    with pytest.raises(ValueError, match="uint8"):
        capture_to_thermal(gray.astype(np.uint16), allow_uint8=True)
    with pytest.raises(ValueError, match="shape"):
        capture_to_thermal(np.zeros((2, 2, 2), dtype=np.uint8), allow_uint8=True)


def test_capture_released_on_generator_close(monkeypatch):
    class FakeCapture:
        released = False

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((10, 10), dtype=np.uint8)

        def release(self):
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(stream.cv2, "VideoCapture", lambda source: capture)
    frames = stream.iter_capture_frames(Path("test.mp4"), allow_uint8=True)
    assert next(frames).index == 0
    assert next(frames).index == 1
    frames.close()
    assert capture.released


def test_capture_released_if_source_cannot_open(monkeypatch):
    class ClosedCapture:
        released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    capture = ClosedCapture()
    monkeypatch.setattr(stream.cv2, "VideoCapture", lambda source: capture)
    with pytest.raises(ValueError, match="Cannot open"):
        next(stream.iter_capture_frames(0, allow_uint8=True))
    assert capture.released


def test_display_does_not_modify_detector_input_and_clips_boxes():
    pixels = np.linspace(110, 120, 4096, dtype=np.float32).reshape(64, 64)
    original = pixels.copy()
    detections = [
        Detection(12, 2, (-5, 20, 25, 40), 0.9),
        Detection(12, 3, (100, 100, 10, 10), 0.8),
    ]
    result = render_detections(pixels, detections, {2: "bike"}, status="CUDA")
    assert result.shape == (64, 64, 3) and result.dtype == np.uint8
    np.testing.assert_array_equal(pixels, original)
    assert not np.array_equal(result[:, :, 0], result[:, :, 1])
    with pytest.raises(ValueError, match="finite"):
        render_detections(np.array([[np.nan]]), [], {})


def test_agc_display_retains_intensities_and_ppm_channel_order():
    gray = np.array([[0, 80, 255]], dtype=np.float32)
    image = render_detections(gray, [], {}, stretch=False)
    np.testing.assert_array_equal(image[:, :, 0], gray)
    bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
    assert ppm_bytes(bgr) == b"P6\n1 1\n255\n" + bytes([30, 20, 10])
    with pytest.raises(ValueError, match="uint8 BGR"):
        ppm_bytes(gray)


def test_preprocessing_comparison_renders_two_aligned_panels():
    frames = [
        np.arange(32 * 48, dtype=np.float32).reshape(32, 48),
        np.arange(32 * 48, dtype=np.float32).reshape(32, 48) + 1,
    ]
    image = render_preprocessing_comparison(
        frames, [[], []], {}, labels=["Original", "wavelet_soft"], panel_width=64
    )
    assert image.dtype == np.uint8
    assert image.shape == (34 + round(64 * 32 / 48), 128, 3)
    with pytest.raises(ValueError, match="Expected two"):
        render_preprocessing_comparison(frames[:1], [[]], {}, labels=["Original"])


def test_preprocessing_comparison_own_range_avoids_saturation():
    source = np.linspace(100, 120, 32 * 48, dtype=np.float32).reshape(32, 48)
    stretched = np.linspace(0, 255, 32 * 48, dtype=np.float32).reshape(32, 48)
    kwargs = {"labels": ["Original", "fshs"], "panel_width": 48}
    shared = render_preprocessing_comparison(
        [source, stretched], [[], []], {}, **kwargs
    )
    own = render_preprocessing_comparison(
        [source, stretched], [[], []], {}, own_range=(False, True), **kwargs
    )
    right_shared = shared[34:, 48:, 0]
    right_own = own[34:, 48:, 0]
    assert np.mean((right_shared == 0) | (right_shared == 255)) > 0.8
    assert np.mean((right_own == 0) | (right_own == 255)) < 0.1
    np.testing.assert_array_equal(shared[:, :48], own[:, :48])

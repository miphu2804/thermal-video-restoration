import numpy as np
import pytest

from thermal_restore.detection import Detection
from thermal_restore.detection.torchvision_backend import normalize_detector_frame


def test_detection_serializes_to_coco():
    prediction = Detection(7, 1, (1, 2, 10, 20), 0.75)

    assert prediction.to_coco() == {
        "image_id": 7,
        "category_id": 1,
        "bbox": [1.0, 2.0, 10.0, 20.0],
        "score": 0.75,
    }


@pytest.mark.parametrize("bbox", [(0, 0, -1, 2), (0, 0, 2), (0, 0, float("nan"), 2)])
def test_detection_rejects_invalid_bbox(bbox):
    with pytest.raises(ValueError):
        Detection(1, 1, bbox, 0.5)


def test_detection_rejects_invalid_score():
    with pytest.raises(ValueError, match="score"):
        Detection(1, 1, (0, 0, 2, 2), 1.1)


def test_detector_frame_uses_fixed_standardization():
    frame = np.array([[100.0, 110.0]], dtype=np.float32)

    normalized = normalize_detector_frame(frame, mean=100.0, std=5.0)

    assert normalized.dtype == np.float32
    assert np.allclose(normalized, [[0.0, 2.0]])


def test_detector_frame_rejects_incomplete_statistics():
    with pytest.raises(ValueError, match="mean/std"):
        normalize_detector_frame(np.zeros((2, 2), dtype=np.float32), mean=100.0)

import pytest

from thermal_restore.detection import Detection


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

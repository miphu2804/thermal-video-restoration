from pathlib import Path

import pytest

from thermal_restore.data.annotations import CocoAnnotation, CocoDataset, CocoImage
from thermal_restore.detection import Detection
from thermal_restore.metrics import bbox_iou, mean_average_precision


def _dataset():
    return CocoDataset(
        images=(CocoImage(1, "frame.jpg", Path("frame.tiff"), 32, 32),),
        annotations=(CocoAnnotation(1, 1, 1, (0, 0, 10, 10), 100, False),),
        categories={1: "person"},
    )


def test_bbox_iou():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_perfect_predictions_have_perfect_map():
    result = mean_average_precision(
        _dataset(), [Detection(1, 1, (0, 0, 10, 10), 0.9)]
    )

    assert result.map50 == 1.0
    assert result.map50_95 == 1.0
    assert result.ap50_by_category == {1: 1.0}


def test_empty_predictions_score_zero():
    result = mean_average_precision(_dataset(), [])

    assert result.map50 == 0.0
    assert result.map50_95 == 0.0


def test_imperfect_box_lowers_high_iou_map():
    result = mean_average_precision(
        _dataset(), [Detection(1, 1, (0, 0, 8, 10), 0.9)]
    )

    assert result.map50 == 1.0
    assert 0.0 < result.map50_95 < 1.0


def test_map_requires_iou_point_five():
    with pytest.raises(ValueError, match="contain 0.5"):
        mean_average_precision(_dataset(), [], iou_thresholds=(0.75,))

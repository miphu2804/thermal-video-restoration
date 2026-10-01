import json

import pytest

from thermal_restore.data.annotations import load_coco


def _write_coco(tmp_path):
    split = tmp_path / "images_thermal_train"
    analytics = split / "analyticsData"
    analytics.mkdir(parents=True)
    (analytics / "video-a-frame-000001-test.tiff").touch()
    (analytics / "video-a-frame-000001-test.jpg").touch()
    payload = {
        "images": [
            {
                "id": 7,
                "file_name": "data/video-a-frame-000001-test.jpg",
                "width": 640,
                "height": 512,
            }
        ],
        "annotations": [
            {
                "id": 3,
                "image_id": 7,
                "category_id": 1,
                "bbox": [1, 2, 10, 20],
                "area": 200,
                "iscrowd": False,
            }
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    path = split / "coco.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, split


def test_load_coco_resolves_jpeg_name_to_thermal_tiff(tmp_path):
    annotation_path, image_root = _write_coco(tmp_path)

    dataset = load_coco(annotation_path, image_root)

    assert dataset.image_ids == (7,)
    assert dataset.images[0].path.name.endswith(".tiff")
    assert dataset.annotations[0].bbox == (1.0, 2.0, 10.0, 20.0)
    assert dataset.used_category_ids == (1,)


def test_load_coco_rejects_missing_image(tmp_path):
    annotation_path, image_root = _write_coco(tmp_path)
    for path in image_root.joinpath("analyticsData").iterdir():
        path.unlink()

    with pytest.raises(FileNotFoundError, match="video-a-frame"):
        load_coco(annotation_path, image_root)


def test_load_coco_rejects_unknown_annotation_image(tmp_path):
    annotation_path, image_root = _write_coco(tmp_path)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["annotations"][0]["image_id"] = 99
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown image"):
        load_coco(annotation_path, image_root)

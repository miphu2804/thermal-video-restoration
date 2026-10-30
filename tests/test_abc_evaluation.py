import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from thermal_restore.data import CocoAnnotation, CocoDataset, CocoImage
from thermal_restore.detection import Detection


@pytest.mark.parametrize("sample_limit", [None, 1])
def test_evaluation_keeps_ids_policy_and_paired_noise_without_real_torch(
    tmp_path, monkeypatch, sample_limit
):
    spec = importlib.util.spec_from_file_location(
        "abc_script", Path("scripts/evaluate_abc.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    images = []
    for image_id, dtype in ((10, np.uint16), (20, np.uint8), (30, np.uint16)):
        path = tmp_path / f"video-a-frame-{image_id}-x.tiff"
        tifffile.imwrite(
            path, np.full((16, 16), 7000 if dtype == np.uint16 else 120, dtype=dtype)
        )
        images.append(CocoImage(image_id, path.name, path, 16, 16))
    dataset = CocoDataset(
        tuple(images),
        tuple(
            CocoAnnotation(index, image.image_id, 1, (2, 2, 8, 8), 64, False)
            for index, image in enumerate(images, 100)
        ),
        {1: "person"},
    )
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"mock checkpoint")
    output = tmp_path / "results"
    args = SimpleNamespace(
        restoration_config=Path("configs/abc_baselines.json"),
        pipelines=["m0"],
        data_root=tmp_path,
        split="val",
        max_images=sample_limit,
        sample_seed=0,
        source_policy="raw14",
        cache_root=None,
        checkpoint=checkpoint,
        device="cpu",
        output_dir=output,
        levels=["light", "medium"],
        seeds=[0, 1],
        batch_size=1,
        progress_interval=0,
        degradation_steps=["fixed_stripes", "noise"],
        score_threshold=0.001,
        save_predictions=True,
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "load_coco", lambda *args: dataset)
    monkeypatch.setattr(
        module,
        "require_device",
        lambda *args: SimpleNamespace(
            selected="cpu", as_dict=lambda: {"selected": "cpu"}
        ),
    )
    monkeypatch.setattr(module, "frame_to_tensor", lambda frame, **kwargs: frame.copy())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    received = []

    class FakeDetector:
        input_mean = 113.0
        input_std = 5.0
        input_normalization = "thermal_standardize"
        category_ids = (1,)
        device = SimpleNamespace(type="cpu")
        model = SimpleNamespace(eval=lambda: None, requires_grad_=lambda value: None)

        def predict(self, frames, ids, threshold):
            received.extend(
                (image_id, frame.copy()) for frame, image_id in zip(frames, ids)
            )
            return [Detection(image_id, 1, (2, 2, 8, 8), 0.9) for image_id in ids]

    monkeypatch.setattr(
        module,
        "TorchvisionDetector",
        SimpleNamespace(from_checkpoint=lambda *args: FakeDetector()),
    )
    module.main()
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    expected_ids = [10, 30] if sample_limit is None else [30]
    assert report["image_ids"] == expected_ids
    assert report["annotation_ids"] == ([100, 102] if sample_limit is None else [102])
    assert report["excluded_by_source_policy_image_ids"] == [20]
    assert report["evaluation_scope"] == (
        "full_source_policy_subset" if sample_limit is None else "smoke_sample"
    )
    assert len(report["conditions"]) == 9
    assert len([row for row in report["conditions"] if row["condition"] == "A"]) == 1
    for row in report["conditions"]:
        assert row["map50"] == 1.0
        if row["condition"] == "C:m0":
            assert row["delta_map50_vs_B"] == 0
            counterpart = next(
                other
                for other in report["conditions"]
                if other["condition"] == "B"
                and other["seed"] == row["seed"]
                and other["level"] == row["level"]
            )
            assert row["image_quality"] == counterpart["image_quality"]
    assert {image_id for image_id, _ in received} == set(expected_ids)
    assert report["checkpoint_sha256"]
    first_prediction = json.loads(
        (output / "predictions_A_clean_none.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert first_prediction["image_id"] == expected_ids[0]

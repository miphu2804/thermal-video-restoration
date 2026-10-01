"""Evaluate a frozen Issue #4 detector on one clean FLIR split.

The same detector wrapper can later be called with degraded/restored tensors for
conditions B and C. This entry point covers condition A on the clean split.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from thermal_restore.data.annotations import CocoDataset, load_coco
from thermal_restore.detection.device import require_device
from thermal_restore.detection.torchvision_backend import (
    TorchvisionCocoDataset,
    TorchvisionDetector,
    collate_detection_batch,
)
from thermal_restore.metrics import mean_average_precision


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise SystemExit(
            "Torch is not installed. Run `uv sync --extra detector` first."
        ) from error
    return torch, DataLoader


def _split(root: Path, name: str) -> CocoDataset:
    split_root = root / name
    return load_coco(split_root / "coco.json", split_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/FLIR_ADAS_v2"))
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Clean split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/detector/clean_test_metrics.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch, DataLoader = _require_torch()
    try:
        device_info = require_device(torch, args.device)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    device = device_info.selected
    print(f"device={device} cuda={device_info.as_dict()}")

    split_name = "images_thermal_val" if args.split == "val" else "video_thermal_test"
    dataset = _split(args.data_root, split_name)
    detector = TorchvisionDetector.from_checkpoint(args.checkpoint, device)
    loader = DataLoader(
        TorchvisionCocoDataset(dataset, detector.category_ids),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_detection_batch,
        pin_memory=torch.cuda.is_available(),
    )

    predictions = []
    started = time.perf_counter()
    for images, targets in loader:
        image_ids = [int(target["image_id"].item()) for target in targets]
        predictions.extend(detector.predict(images, image_ids, args.score_threshold))
    elapsed = time.perf_counter() - started
    metrics = mean_average_precision(dataset, predictions)
    result = {
        "condition": "A-clean",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "device": device,
        "images": len(dataset.images),
        "predictions": len(predictions),
        "seconds": elapsed,
        "seconds_per_image": elapsed / len(dataset.images),
        "map50": metrics.map50,
        "map50_95": metrics.map50_95,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

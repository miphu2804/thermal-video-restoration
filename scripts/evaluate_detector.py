"""Evaluate a frozen Issue #4 detector on one clean FLIR split.

The same detector wrapper can later be called with degraded/restored tensors for
conditions B and C. This entry point covers condition A on the clean split.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from thermal_restore.data.annotations import CocoDataset, load_coco
from thermal_restore.detection.device import require_device
from thermal_restore.detection.diagnostics import subset_dataset
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
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=None,
        help="Export dtype/sequence metrics, prediction JSON and comparison PNGs.",
    )
    parser.add_argument("--preview-count", type=int, default=40)
    parser.add_argument(
        "--preview-score-threshold",
        type=float,
        default=0.3,
        help="Display-only confidence threshold; never changes mAP predictions.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Smoke-test only: evaluate a seeded sample, not the full split.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help=(
            "Show a tqdm progress bar and refresh predictions every N batches; "
            "0 disables."
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional cache root created by scripts/cache_detector_frames.py.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.001,
        help="Low confidence floor used to build the precision-recall curve.",
    )
    parser.add_argument(
        "--allow-uint8",
        action="store_true",
        help="Read FLIR mirror uint8 TIFFs directly on the 0-255 display scale.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/detector/clean_test_metrics.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.score_threshold <= 1 or args.progress_interval < 0:
        raise SystemExit(
            "score-threshold must be in [0, 1]; progress-interval cannot be negative"
        )
    if (
        args.preview_count < 0
        or not 0 <= args.preview_score_threshold <= 1
        or args.seed < 0
        or (args.max_images is not None and args.max_images < 1)
    ):
        raise SystemExit(
            "preview-count and seed must be non-negative; preview-score-threshold "
            "must be in [0, 1]; max-images must be positive"
        )
    torch, DataLoader = _require_torch()
    try:
        device_info = require_device(torch, args.device)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    device = device_info.selected
    print(f"device={device} cuda={device_info.as_dict()}")

    split_name = "images_thermal_val" if args.split == "val" else "video_thermal_test"
    dataset = _split(args.data_root, split_name)
    full_split_images = len(dataset.images)
    detector = TorchvisionDetector.from_checkpoint(args.checkpoint, device)
    torch_dataset = TorchvisionCocoDataset(
        dataset,
        detector.category_ids,
        allow_uint8=args.allow_uint8,
        cache_dir=args.cache_root / args.split if args.cache_root else None,
        input_mean=detector.input_mean,
        input_std=detector.input_std,
    )
    if args.max_images is not None and args.max_images < full_split_images:
        indices = np.sort(
            np.random.default_rng(args.seed).choice(
                full_split_images, size=args.max_images, replace=False
            )
        )
        dataset = subset_dataset(
            dataset, [dataset.images[int(index)].image_id for index in indices]
        )
        # Subset the original torch dataset so cache indexing remains valid.
        torch_dataset = torch.utils.data.Subset(torch_dataset, indices.tolist())
        print(f"SMOKE SAMPLE: {len(dataset.images)}/{full_split_images} images")
    loader = DataLoader(
        torch_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_detection_batch,
        pin_memory=torch.cuda.is_available(),
    )

    predictions = []
    started = time.perf_counter()
    total_batches = len(loader)
    progress = tqdm(
        total=total_batches,
        desc=f"evaluate split={args.split}",
        unit="batch",
        dynamic_ncols=True,
        disable=args.progress_interval == 0,
    )
    for batch_index, (images, targets) in enumerate(loader, start=1):
        image_ids = [int(target["image_id"].item()) for target in targets]
        predictions.extend(detector.predict(images, image_ids, args.score_threshold))
        progress.update(1)
        if args.progress_interval > 0 and (
            batch_index == total_batches or batch_index % args.progress_interval == 0
        ):
            progress.set_postfix(
                predictions=len(predictions),
                refresh=True,
            )
    progress.close()
    if args.progress_interval > 0:
        tqdm.write("Inference complete; computing mAP...")
    elapsed = time.perf_counter() - started
    metrics = mean_average_precision(dataset, predictions)
    result = {
        "condition": "A-clean",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "device": device,
        "images": len(dataset.images),
        "full_split_images": full_split_images,
        "evaluation_scope": (
            "full_split" if len(dataset.images) == full_split_images else "smoke_sample"
        ),
        "seed": args.seed,
        "score_threshold": args.score_threshold,
        "predictions": len(predictions),
        "seconds": elapsed,
        "seconds_per_image": elapsed / len(dataset.images),
        "map50": metrics.map50,
        "map50_95": metrics.map50_95,
        "allow_uint8": args.allow_uint8,
        "cache_root": str(args.cache_root) if args.cache_root else None,
        "input_normalization": detector.input_normalization,
        "input_mean": detector.input_mean,
        "input_std": detector.input_std,
        "ap50_by_category": {
            str(category_id): {
                "name": dataset.categories.get(category_id, str(category_id)),
                "value": metrics.ap50_by_category.get(category_id, 0.0),
            }
            for category_id in sorted(metrics.ap50_by_category)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if args.diagnostics_dir is not None:
        from detector_diagnostics import export_diagnostics

        export_diagnostics(
            dataset,
            predictions,
            result,
            args.diagnostics_dir,
            preview_count=args.preview_count,
            preview_score_threshold=args.preview_score_threshold,
            seed=args.seed,
            show_progress=args.progress_interval > 0,
        )
        print(f"diagnostics={args.diagnostics_dir.resolve()}")


if __name__ == "__main__":
    main()

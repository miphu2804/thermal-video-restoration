"""Train and freeze the Issue #4 torchvision detector on clean FLIR data.

This script is intentionally a user-run entry point. It downloads pretrained
weights only when the user starts it and writes checkpoints under ``outputs/``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from thermal_restore.data.annotations import CocoDataset, load_coco
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


def _device(torch, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but no CUDA device is available")
    return requested


def _split(root: Path, name: str) -> CocoDataset:
    split_root = root / name
    return load_coco(split_root / "coco.json", split_root)


def _evaluate(
    torch,
    DataLoader,
    detector: TorchvisionDetector,
    dataset: CocoDataset,
    batch_size: int,
    workers: int,
    score_threshold: float,
) -> tuple[float, float, float]:
    loader = DataLoader(
        TorchvisionCocoDataset(dataset, detector.category_ids),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_detection_batch,
        pin_memory=torch.cuda.is_available(),
    )
    predictions = []
    for images, targets in loader:
        image_ids = [int(target["image_id"].item()) for target in targets]
        predictions.extend(detector.predict(images, image_ids, score_threshold))
    metrics = mean_average_precision(dataset, predictions)
    return metrics.map50, metrics.map50_95, len(predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/FLIR_ADAS_v2"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/detector/fasterrcnn_mobilenet_best.pt"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not download or use torchvision COCO pretrained weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise SystemExit("epochs and batch-size must be positive; workers cannot be negative")

    torch, DataLoader = _require_torch()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(torch, args.device)

    train = _split(args.data_root, "images_thermal_train")
    validation = _split(args.data_root, "images_thermal_val")
    category_ids = train.used_category_ids
    unknown = set(validation.used_category_ids) - set(category_ids)
    if unknown:
        raise SystemExit(f"Validation contains categories absent from training: {unknown}")

    train_loader = DataLoader(
        TorchvisionCocoDataset(train, category_ids),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_detection_batch,
        pin_memory=torch.cuda.is_available(),
    )
    detector = TorchvisionDetector.create(
        category_ids, device, pretrained=not args.no_pretrained
    )
    optimizer = torch.optim.AdamW(detector.model.parameters(), lr=args.learning_rate)
    best_map50 = -1.0

    for epoch in range(1, args.epochs + 1):
        detector.model.train()
        losses = []
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = detector.training_loss(images, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        map50, map50_95, n_predictions = _evaluate(
            torch,
            DataLoader,
            detector,
            validation,
            args.batch_size,
            args.workers,
            args.score_threshold,
        )
        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(
            f"epoch={epoch} loss={mean_loss:.4f} "
            f"val_map50={map50:.4f} val_map50_95={map50_95:.4f} "
            f"predictions={n_predictions} device={device}"
        )
        if map50 >= best_map50:
            best_map50 = map50
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": detector.model.state_dict(),
                    "category_ids": list(category_ids),
                    "epoch": epoch,
                    "val_map50": map50,
                    "val_map50_95": map50_95,
                    "device": device,
                    "seed": args.seed,
                },
                args.output,
            )
            print(f"saved={args.output}")


if __name__ == "__main__":
    main()

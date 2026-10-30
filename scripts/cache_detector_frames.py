"""Build reusable float32 frame caches for detector training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from thermal_restore.data.annotations import load_coco
from thermal_restore.data.frame_cache import build_frame_cache, estimate_cache_bytes

SPLIT_DIRECTORIES = {
    "train": "images_thermal_train",
    "val": "images_thermal_val",
    "test": "video_thermal_test",
}


def _format_bytes(value: int) -> str:
    gibibytes = value / (1024**3)
    return f"{gibibytes:.2f} GiB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/FLIR_ADAS_v2"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/cache/detector_float32_v1"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_DIRECTORIES),
        default=("train", "val"),
    )
    parser.add_argument(
        "--allow-uint8-test",
        action="store_true",
        help="Cache test mirror sequences as float32 on their existing 0-255 scale.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in args.splits:
        split_root = args.data_root / SPLIT_DIRECTORIES[split]
        dataset = load_coco(split_root / "coco.json", split_root)
        size = estimate_cache_bytes(dataset)
        output = args.cache_root / split
        print(
            f"split={split} images={len(dataset.images)} "
            f"estimated_size={_format_bytes(size)} output={output}"
        )
        if args.dry_run:
            continue

        last_reported = 0

        def report(completed: int, total: int, split_name: str = split) -> None:
            nonlocal last_reported
            if completed == total or completed - last_reported >= 250:
                print(f"split={split_name} cached={completed}/{total}")
                last_reported = completed

        build_frame_cache(
            dataset,
            output,
            allow_uint8=split == "test" and args.allow_uint8_test,
            overwrite=args.overwrite,
            progress=report,
        )
        print(f"completed={output}")


if __name__ == "__main__":
    main()

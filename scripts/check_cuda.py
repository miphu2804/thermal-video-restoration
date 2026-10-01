"""Check the installed PyTorch build and whether CUDA is usable."""

from __future__ import annotations

import argparse
import json

from thermal_restore.detection.device import require_device


def _require_torch():
    try:
        import torch
    except ImportError as error:
        raise SystemExit(
            "Torch is not installed. Install the detector dependencies first."
        ) from error
    return torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    torch = _require_torch()
    try:
        info = require_device(torch, args.device)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(info.as_dict(), indent=2))


if __name__ == "__main__":
    main()

"""Memory-mapped frame cache for repeated detector experiments."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from thermal_restore.data.annotations import CocoDataset
from thermal_restore.data.loader import load_frame

CACHE_VERSION = 1
FRAME_FILE = "frames.npy"
MANIFEST_FILE = "manifest.json"


def estimate_cache_bytes(dataset: CocoDataset) -> int:
    """Estimate bytes required for an uncompressed float32 frame cache."""
    itemsize = np.dtype(np.float32).itemsize
    return sum(image.width * image.height * itemsize for image in dataset.images)


class FrameCache:
    """Read validated float32 frames lazily through a NumPy memory map."""

    def __init__(
        self,
        directory: Path,
        dataset: CocoDataset,
        *,
        allow_uint8: bool = False,
    ) -> None:
        self.directory = Path(directory)
        self.frame_path = self.directory / FRAME_FILE
        manifest_path = self.directory / MANIFEST_FILE
        if not self.frame_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Incomplete frame cache at {self.directory}; expected "
                f"{FRAME_FILE} and {MANIFEST_FILE}"
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != CACHE_VERSION:
            raise ValueError(
                f"Unsupported frame cache version: {manifest.get('version')!r}"
            )
        if manifest.get("dtype") != "float32" or manifest.get("scale") != "0-255":
            raise ValueError("Frame cache must contain float32 frames on the 0-255 scale")
        if bool(manifest.get("allow_uint8")) != bool(allow_uint8):
            raise ValueError(
                "Frame cache uint8 policy does not match the requested dataset policy"
            )

        expected_images = [
            {
                "image_id": image.image_id,
                "file_name": image.file_name,
                "width": image.width,
                "height": image.height,
            }
            for image in dataset.images
        ]
        if manifest.get("images") != expected_images:
            raise ValueError("Frame cache image manifest does not match the COCO dataset")

        shape = tuple(int(value) for value in manifest.get("shape", ()))
        if len(shape) != 3 or shape[0] != len(dataset.images):
            raise ValueError(f"Invalid frame cache shape: {shape}")
        self.shape = shape
        self._frames: np.memmap | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Drop the open memory map when a Windows DataLoader spawns workers."""
        state = self.__dict__.copy()
        state["_frames"] = None
        return state

    def _open(self) -> np.memmap:
        if self._frames is None:
            frames = np.load(self.frame_path, mmap_mode="r", allow_pickle=False)
            if frames.dtype != np.float32 or frames.shape != self.shape:
                raise ValueError(
                    f"Frame cache array mismatch: dtype={frames.dtype}, "
                    f"shape={frames.shape}"
                )
            self._frames = frames
        return self._frames

    def load(self, index: int) -> np.ndarray:
        """Load one writable frame copy by dataset index."""
        return np.array(self._open()[index], dtype=np.float32, copy=True)

    def sampled_mean_std(
        self,
        *,
        frame_stride: int = 10,
        pixel_stride: int = 8,
    ) -> tuple[float, float]:
        """Estimate global statistics from a deterministic strided cache view."""
        if frame_stride < 1 or pixel_stride < 1:
            raise ValueError("Sampling strides must be positive")
        sample = self._open()[::frame_stride, ::pixel_stride, ::pixel_stride]
        mean = float(np.mean(sample, dtype=np.float64))
        std = float(np.std(sample, dtype=np.float64))
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError("Sampled cache statistics must be finite and non-zero")
        return mean, std


def build_frame_cache(
    dataset: CocoDataset,
    directory: Path,
    *,
    allow_uint8: bool = False,
    overwrite: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Build an atomic float32 cache from a validated COCO dataset."""
    if not dataset.images:
        raise ValueError("Cannot cache an empty dataset")

    dimensions = {(image.height, image.width) for image in dataset.images}
    if len(dimensions) != 1:
        raise ValueError("Memory-mapped frame caches require one shared frame shape")
    height, width = next(iter(dimensions))
    shape = (len(dataset.images), height, width)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame_path = directory / FRAME_FILE
    manifest_path = directory / MANIFEST_FILE
    if not overwrite and (frame_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"Frame cache already exists at {directory}; use overwrite=True to rebuild"
        )

    temporary_frame = directory / "frames.tmp.npy"
    temporary_manifest = directory / "manifest.tmp.json"
    temporary_frame.unlink(missing_ok=True)
    temporary_manifest.unlink(missing_ok=True)

    completed = False
    try:
        frames = np.lib.format.open_memmap(
            temporary_frame,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        total = len(dataset.images)
        for index, image in enumerate(dataset.images):
            frame = load_frame(image.path, allow_uint8=allow_uint8)
            if frame.shape != (image.height, image.width):
                raise ValueError(
                    f"{image.path}: COCO shape {(image.height, image.width)} "
                    f"does not match TIFF shape {frame.shape}"
                )
            frames[index] = frame
            if progress is not None:
                progress(index + 1, total)
        frames.flush()
        del frames

        manifest = {
            "version": CACHE_VERSION,
            "dtype": "float32",
            "scale": "0-255",
            "allow_uint8": bool(allow_uint8),
            "shape": list(shape),
            "images": [
                {
                    "image_id": image.image_id,
                    "file_name": image.file_name,
                    "width": image.width,
                    "height": image.height,
                }
                for image in dataset.images
            ],
        }
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_frame.replace(frame_path)
        temporary_manifest.replace(manifest_path)
        completed = True
    finally:
        if not completed:
            temporary_frame.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)

    return directory

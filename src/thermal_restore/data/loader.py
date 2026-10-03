"""FLIR ADAS data loading. Other modules read frames only through here."""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

# 14-bit sensor counts -> 8-bit-equivalent scale (0-255), the scale the
# degradation parameters in the proposal are defined on.
_SCALE_14_TO_8 = 255.0 / (2**14 - 1)

# Video frames are named like "video-<seq id>-frame-<index>-<hash>.tiff".
_FRAME_RE = re.compile(
    r"""
    ^(?P<seq>video-[^-]+)   # sequence id, e.g. "video-24ysbPEGoEKKDvRt6"
    -frame-
    (?P<idx>\d+)            # zero-padded frame number, e.g. "000015"
    """,
    re.VERBOSE,
)


def load_frame(path: Path) -> np.ndarray:
    """Read one 14-bit TIFF as float32 ``(H, W)`` on the 0-255 scale.

    Raises:
        ValueError: If the frame is not a single-channel ``uint16`` image. Some
            FLIR mirror sequences are contrast-stretched ``uint8`` and cannot be
            converted back to 14-bit counts, so they must not be rescaled.
    """
    raw = tifffile.imread(path)
    if raw.dtype != np.uint16:
        raise ValueError(
            f"{path}: expected a 14-bit uint16 frame, got dtype {raw.dtype}"
        )
    if raw.ndim != 2:
        raise ValueError(
            f"{path}: expected a single-channel frame, got shape {raw.shape}"
        )
    return raw.astype(np.float32) * np.float32(_SCALE_14_TO_8)


def load_sequence(paths: list[Path]) -> np.ndarray:
    """Stack frames, in the given order, into ``(T, H, W)`` float32."""
    return np.stack([load_frame(p) for p in paths])


def list_sequences(root: Path) -> dict[str, list[Path]]:
    """Group TIFF frames under ``root`` by video sequence, in frame order."""
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in Path(root).rglob("*.tif*"):
        match = _FRAME_RE.match(path.name)
        if match:
            groups[match["seq"]].append((int(match["idx"]), path))
    return {
        seq: [p for _, p in sorted(frames)] for seq, frames in sorted(groups.items())
    }


def frame_index(path: Path) -> int:
    """Return the frame number encoded in a FLIR frame file name."""
    match = _FRAME_RE.match(Path(path).name)
    if not match:
        raise ValueError(f"{path}: not a FLIR video frame name")
    return int(match["idx"])


def sample_frames(
    sequences: dict[str, list[Path]], n: int, seed: int = 0
) -> list[tuple[str, Path]]:
    """Pick ``n`` distinct ``(sequence id, path)`` frames uniformly at random."""
    frames = [(seq, p) for seq, paths in sequences.items() for p in paths]
    picks = np.random.default_rng(seed).choice(len(frames), size=n, replace=False)
    return [frames[k] for k in picks]

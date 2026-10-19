"""Read and validate public JSON restoration experiment configurations."""

import json
import re
from pathlib import Path

from thermal_restore.experiments import build_restoration


def load_restoration_config(path: Path) -> dict:
    """Load explicit stage parameters and reject unsafe/ambiguous labels."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("version") != 1
        or not isinstance(payload.get("pipelines"), dict)
        or not payload["pipelines"]
    ):
        raise ValueError("Expected version=1 and a non-empty pipelines object")
    for name, stages in payload["pipelines"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"Invalid pipeline label: {name!r}")
        build_restoration(stages)
    if payload.get("default_pipeline") not in payload["pipelines"]:
        raise ValueError("default_pipeline must name a configured pipeline")
    return payload


def source_sequence(path: Path) -> str:
    """Extract a FLIR sequence ID, or use a stable generic source identity."""
    match = re.match(r"^(video-[^-]+)-frame-", path.name)
    return match.group(1) if match else str(path.parent.resolve())

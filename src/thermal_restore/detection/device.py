"""Device selection and CUDA runtime diagnostics for the detector backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceInfo:
    """Describe the selected device and the CUDA runtime probe."""

    requested: str
    selected: str
    cuda_available: bool
    cuda_version: str | None
    device_count: int
    device_name: str | None
    device_capability: tuple[int, int] | None
    cuda_error: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the probe."""
        return asdict(self)


def inspect_device(torch: Any, requested: str = "auto") -> DeviceInfo:
    """Probe CUDA and select a device without starting model execution.

    ``auto`` selects CUDA only when the driver is visible and a small CUDA
    tensor allocation succeeds. A failed probe falls back to CPU so the user
    receives a usable run with a diagnostic instead of a late model error.
    """
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unknown device request: {requested!r}")

    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    cuda_available = False
    device_count = 0
    device_name = None
    device_capability = None
    cuda_error = None

    try:
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device_count = int(torch.cuda.device_count())
            device_name = str(torch.cuda.get_device_name(0))
            device_capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
            torch.empty(1, device="cuda").item()
    except (AssertionError, OSError, RuntimeError) as error:
        cuda_available = False
        cuda_error = f"{type(error).__name__}: {error}"

    selected = "cuda" if requested == "cuda" or (requested == "auto" and cuda_available) else "cpu"
    return DeviceInfo(
        requested=requested,
        selected=selected,
        cuda_available=cuda_available,
        cuda_version=str(cuda_version) if cuda_version else None,
        device_count=device_count,
        device_name=device_name,
        device_capability=device_capability,
        cuda_error=cuda_error,
    )


def require_device(torch: Any, requested: str = "auto") -> DeviceInfo:
    """Select a device and fail early when the user explicitly requires CUDA."""
    info = inspect_device(torch, requested)
    if requested == "cuda" and not info.cuda_available:
        detail = f" ({info.cuda_error})" if info.cuda_error else ""
        raise RuntimeError(
            "CUDA was requested but the CUDA runtime probe failed. "
            "Install the CUDA-enabled detector extra and check the NVIDIA driver"
            f"{detail}"
        )
    return info

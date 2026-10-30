from types import SimpleNamespace

import pytest

from thermal_restore.detection.device import inspect_device, require_device


class _Cuda:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return 1

    def get_device_name(self, index: int) -> str:
        return "Test GPU"

    def get_device_capability(self, index: int) -> tuple[int, int]:
        return (13, 0)


class _Torch:
    version = SimpleNamespace(cuda="13.0")

    def __init__(self, available: bool) -> None:
        self.cuda = _Cuda(available)

    @staticmethod
    def empty(size: int, device: str):
        return SimpleNamespace(item=lambda: 0.0)


def test_auto_selects_cuda_when_probe_succeeds() -> None:
    info = inspect_device(_Torch(available=True), "auto")

    assert info.selected == "cuda"
    assert info.cuda_available is True
    assert info.device_name == "Test GPU"


def test_auto_falls_back_to_cpu_without_cuda() -> None:
    info = inspect_device(_Torch(available=False), "auto")

    assert info.selected == "cpu"
    assert info.cuda_available is False


def test_explicit_cuda_fails_early_without_cuda() -> None:
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        require_device(_Torch(available=False), "cuda")

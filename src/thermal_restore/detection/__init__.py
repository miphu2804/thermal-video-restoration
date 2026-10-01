"""Detector data structures and backend integrations."""

from thermal_restore.detection.device import DeviceInfo, inspect_device, require_device
from thermal_restore.detection.types import Detection

__all__ = ["Detection", "DeviceInfo", "inspect_device", "require_device"]

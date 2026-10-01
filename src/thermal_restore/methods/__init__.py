"""Restoration methods M0-M7, built by name through the registry."""

from thermal_restore.methods import m0_identity, m1_contrast, m2_spatial  # noqa: F401
from thermal_restore.methods.registry import available_methods, get_method, register

__all__ = ["available_methods", "get_method", "register"]

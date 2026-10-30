"""Restoration methods M0-M7, built by name through the registry."""

from thermal_restore.methods import (  # noqa: F401
    m0_identity,
    m1_contrast,
    m2_spatial,
    m3_preprocessing,
)
from thermal_restore.methods.registry import available_methods, get_method, register

__all__ = ["available_methods", "get_method", "register"]

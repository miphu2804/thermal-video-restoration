"""Reproducible comparisons using a single frozen detector."""

from thermal_restore.experiments.comparison import (
    QualityTotals,
    build_restoration,
    degrade_frame,
    range_remapping_stages,
)

__all__ = [
    "QualityTotals",
    "build_restoration",
    "degrade_frame",
    "range_remapping_stages",
]

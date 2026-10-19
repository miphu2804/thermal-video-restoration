"""Small helpers shared by command-line progress reporting."""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Format a non-negative duration as ``HH:MM:SS``."""
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

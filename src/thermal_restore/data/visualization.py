"""Display helpers. Display-only: never feed their output back into the pipeline."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def stretch_contrast(frame: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    """Stretch a frame to ``[0, 1]`` between two percentiles for viewing.

    Thermal frames occupy a narrow band of the sensor range, so they look flat
    without this.
    """
    lo, hi = np.percentile(frame, [low, high])
    return np.clip((frame - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def plot_frame_grid(
    frames: list[np.ndarray], titles: list[str], ncols: int = 5
) -> Figure:
    """Show frames in a grid, each contrast-stretched, with one title each."""
    nrows = -(-len(frames) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, frame, title in zip(np.atleast_1d(axes).ravel(), frames, titles):
        ax.imshow(stretch_contrast(frame), cmap="gray")
        ax.set_title(title)
    fig.tight_layout()
    return fig

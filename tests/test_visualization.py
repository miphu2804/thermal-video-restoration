import numpy as np

from thermal_restore.data.visualization import stretch_contrast


def test_stretch_contrast_maps_to_unit_range():
    out = stretch_contrast(np.linspace(100, 130, 1000).reshape(10, 100))
    assert out.min() == 0.0 and out.max() == 1.0


def test_stretch_contrast_constant_frame_is_finite():
    assert np.isfinite(stretch_contrast(np.full((4, 4), 7.0))).all()

import numpy as np
import pytest
import tifffile

from thermal_restore.data.annotations import CocoDataset, CocoImage
from thermal_restore.data.frame_cache import (
    FrameCache,
    build_frame_cache,
    estimate_cache_bytes,
)


def _dataset(tmp_path, dtype=np.uint16, value=2**14 - 1):
    images = []
    for index in range(2):
        path = tmp_path / f"frame-{index}.tiff"
        tifffile.imwrite(path, np.full((4, 6), value, dtype=dtype))
        images.append(
            CocoImage(
                image_id=index + 1,
                file_name=path.name,
                path=path,
                width=6,
                height=4,
            )
        )
    return CocoDataset(tuple(images), (), {})


def test_build_and_load_float32_frame_cache(tmp_path):
    dataset = _dataset(tmp_path)
    cache_dir = tmp_path / "cache"

    build_frame_cache(dataset, cache_dir)
    cache = FrameCache(cache_dir, dataset)

    frame = cache.load(1)
    assert frame.dtype == np.float32
    assert frame.shape == (4, 6)
    assert np.allclose(frame, 255.0)
    assert frame.flags.writeable
    assert estimate_cache_bytes(dataset) == 2 * 4 * 6 * 4


def test_frame_cache_validates_dataset_manifest(tmp_path):
    dataset = _dataset(tmp_path)
    cache_dir = tmp_path / "cache"
    build_frame_cache(dataset, cache_dir)
    reversed_dataset = CocoDataset(tuple(reversed(dataset.images)), (), {})

    with pytest.raises(ValueError, match="manifest"):
        FrameCache(cache_dir, reversed_dataset)


def test_frame_cache_records_explicit_uint8_policy(tmp_path):
    dataset = _dataset(tmp_path, dtype=np.uint8, value=200)
    cache_dir = tmp_path / "cache"
    build_frame_cache(dataset, cache_dir, allow_uint8=True)

    with pytest.raises(ValueError, match="uint8 policy"):
        FrameCache(cache_dir, dataset)

    frame = FrameCache(cache_dir, dataset, allow_uint8=True).load(0)
    assert np.allclose(frame, 200.0)


def test_frame_cache_refuses_overwrite_by_default(tmp_path):
    dataset = _dataset(tmp_path)
    cache_dir = tmp_path / "cache"
    build_frame_cache(dataset, cache_dir)

    with pytest.raises(FileExistsError, match="already exists"):
        build_frame_cache(dataset, cache_dir)


def test_frame_cache_estimates_strided_statistics(tmp_path):
    dataset = _dataset(tmp_path)
    cache_dir = tmp_path / "cache"
    build_frame_cache(dataset, cache_dir)
    frames = np.load(cache_dir / "frames.npy", mmap_mode="r+")
    frames[1] = 128.0
    frames.flush()
    del frames

    mean, std = FrameCache(cache_dir, dataset).sampled_mean_std(
        frame_stride=1,
        pixel_stride=1,
    )

    assert mean == pytest.approx(191.5)
    assert std == pytest.approx(63.5)

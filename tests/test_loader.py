import numpy as np
import tifffile

from thermal_restore.data.loader import (
    frame_index,
    list_sequences,
    load_frame,
    load_sequence,
    sample_frames,
)


def _write(path, value):
    tifffile.imwrite(path, np.full((4, 6), value, dtype=np.uint16))


def test_load_frame_scales_14bit_to_8bit(tmp_path):
    _write(tmp_path / "f.tiff", 2**14 - 1)
    frame = load_frame(tmp_path / "f.tiff")
    assert frame.dtype == np.float32 and frame.shape == (4, 6)
    assert np.allclose(frame, 255.0)


def test_list_and_load_sequences(tmp_path):
    for seq, idx in [("aaa", 10), ("aaa", 2), ("bbb", 1)]:
        _write(tmp_path / f"video-{seq}-frame-{idx:06d}-xyz.tiff", idx)
    seqs = list_sequences(tmp_path)
    assert list(seqs) == ["video-aaa", "video-bbb"]
    assert [p.name.split("-")[3] for p in seqs["video-aaa"]] == ["000002", "000010"]
    assert load_sequence(seqs["video-aaa"]).shape == (2, 4, 6)


def test_frame_index_and_sample_frames(tmp_path):
    for seq, idx in [("aaa", 1), ("aaa", 2), ("bbb", 7)]:
        _write(tmp_path / f"video-{seq}-frame-{idx:06d}-xyz.tiff", idx)
    seqs = list_sequences(tmp_path)
    picks = sample_frames(seqs, 3, seed=1)
    assert sorted(frame_index(p) for _, p in picks) == [1, 2, 7]
    assert picks == sample_frames(seqs, 3, seed=1)

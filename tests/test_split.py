from thermal_restore.data.split import split_by_sequence

IDS = [f"video-{i:04d}" for i in range(40)]


def test_no_sequence_in_two_sets():
    parts = split_by_sequence(IDS, seed=1)
    flat = [s for part in parts.values() for s in part]
    assert sorted(flat) == sorted(IDS)
    assert len(flat) == len(set(flat))


def test_ratios_and_determinism():
    parts = split_by_sequence(IDS, seed=1)
    assert [len(parts[k]) for k in ("train", "val", "test")] == [28, 6, 6]
    assert parts == split_by_sequence(IDS[::-1], seed=1)

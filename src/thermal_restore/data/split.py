"""Train / validation / test split by video sequence."""

import random


def split_by_sequence(
    seq_ids: list[str],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, list[str]]:
    """Split sequence ids into ``train`` / ``val`` / ``test``.

    Splitting by sequence rather than frame avoids leaking near-identical
    neighbouring frames across sets.
    """
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must be 3 values summing to 1, got {ratios}")
    ids = sorted(set(seq_ids))
    random.Random(seed).shuffle(ids)
    n_train = round(ratios[0] * len(ids))
    n_val = round(ratios[1] * len(ids))
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }

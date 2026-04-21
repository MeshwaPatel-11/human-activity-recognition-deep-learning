import numpy as np
from typing import Tuple, List, Dict

def subject_independent_split(subjects: np.ndarray, seed: int = 42,
                              train_frac: float = 0.7, val_frac: float = 0.15
                              ) -> Dict[str, np.ndarray]:
    """
    Create subject-independent indices: train/val/test have disjoint subjects.
    Returns dict with keys: train_idx, val_idx, test_idx.
    """
    uniq = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n = len(uniq)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError("Not enough subjects for split. Adjust fractions.")

    train_sub = set(uniq[:n_train])
    val_sub = set(uniq[n_train:n_train + n_val])
    test_sub = set(uniq[n_train + n_val:])

    train_idx = np.where(np.isin(subjects, list(train_sub)))[0]
    val_idx = np.where(np.isin(subjects, list(val_sub)))[0]
    test_idx = np.where(np.isin(subjects, list(test_sub)))[0]

    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}

def check_no_subject_overlap(subjects: np.ndarray, split: Dict[str, np.ndarray]) -> None:
    tr = set(subjects[split["train_idx"]])
    va = set(subjects[split["val_idx"]])
    te = set(subjects[split["test_idx"]])

    assert tr.isdisjoint(va), "Train/Val subjects overlap!"
    assert tr.isdisjoint(te), "Train/Test subjects overlap!"
    assert va.isdisjoint(te), "Val/Test subjects overlap!"
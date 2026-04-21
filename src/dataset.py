import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List


@dataclass
class HARData:
    X: np.ndarray         # (N, T, C)
    y: np.ndarray         # (N,)
    subjects: np.ndarray  # (N,)
    label_map: Dict[int, str]


def _read_txt_matrix(path: str) -> np.ndarray:
    # Loads space-separated numeric matrix from txt
    return np.loadtxt(path)


def _load_inertial_block(folder: str, split: str) -> np.ndarray:
    """
    Loads UCI HAR inertial signals and returns (N, T, C).
    Expected files (9 total):
      total_acc_[x|y|z]_{train|test}.txt
      body_acc_[x|y|z]_{train|test}.txt
      body_gyro_[x|y|z]_{train|test}.txt
    Each file is (N, 128). We'll stack into channels C=9.
    """
    base = os.path.join(folder, split, "Inertial Signals")
    signal_names = [
        "total_acc_x", "total_acc_y", "total_acc_z",
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
    ]
    signals = []
    for name in signal_names:
        f = os.path.join(base, f"{name}_{split}.txt")
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing file: {f}")
        arr = _read_txt_matrix(f)  # (N, 128)
        signals.append(arr)

    # Stack channels last: (N, 128, 9)
    X = np.stack(signals, axis=-1)
    return X.astype(np.float32)


def _load_labels(folder: str, split: str) -> np.ndarray:
    y_path = os.path.join(folder, split, f"y_{split}.txt")
    y = np.loadtxt(y_path).astype(int)  # labels are 1..6
    return y


def _load_subjects(folder: str, split: str) -> np.ndarray:
    s_path = os.path.join(folder, split, f"subject_{split}.txt")
    s = np.loadtxt(s_path).astype(int)
    return s


def _load_label_map(folder: str) -> Dict[int, str]:
    p = os.path.join(folder, "activity_labels.txt")
    mapping = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            idx, name = line.strip().split()
            mapping[int(idx)] = name
    return mapping


def load_ucihar(data_dir: str) -> Tuple[HARData, HARData]:
    """
    Returns (train_data, test_data) based on dataset's provided split.
    data_dir is the path to UCI_HAR_Dataset folder.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    label_map = _load_label_map(data_dir)

    X_train = _load_inertial_block(data_dir, "train")
    y_train = _load_labels(data_dir, "train")
    s_train = _load_subjects(data_dir, "train")

    X_test = _load_inertial_block(data_dir, "test")
    y_test = _load_labels(data_dir, "test")
    s_test = _load_subjects(data_dir, "test")

    train = HARData(X_train, y_train, s_train, label_map)
    test = HARData(X_test, y_test, s_test, label_map)
    return train, test


def select_sensors(X: np.ndarray, sensors: str) -> np.ndarray:
    """
    sensors:
      - "both": all 9 channels
      - "accel": keep accelerometer channels only (total_acc + body_acc) = 6
      - "gyro": keep gyro only (body_gyro) = 3
    Channel order in this loader:
      0..2 total_acc, 3..5 body_acc, 6..8 body_gyro
    """
    sensors = sensors.lower()
    if sensors == "both":
        return X
    if sensors == "accel":
        return X[:, :, :6]
    if sensors == "gyro":
        return X[:, :, 6:]
    raise ValueError("sensors must be one of: both, accel, gyro")


def normalize_train_apply(X_train: np.ndarray, X_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize using train stats only (no leakage).
    Computes mean/std over (N,T) per channel.
    Returns normalized (X_train_n, X_other_n, mean, std).
    """
    # shape (C,)
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train_n = (X_train - mean) / std
    X_other_n = (X_other - mean) / std
    return X_train_n, X_other_n, mean.squeeze(), std.squeeze()
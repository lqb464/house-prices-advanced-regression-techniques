"""Scaling chỉ trên tập train và tạo chuỗi theo thời gian."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN


def fit_scaler(train: pd.DataFrame) -> StandardScaler:
    return StandardScaler().fit(train[FEATURE_COLUMNS])


def make_sequences(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    sequence_length: int,
    *,
    start: int = 0,
    end: int | None = None,
    require_target: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    end = len(frame) if end is None else end
    values = scaler.transform(frame[FEATURE_COLUMNS]).astype("float32")
    targets = frame[TARGET_COLUMN].to_numpy(dtype="float32") if require_target else None
    sequences, labels, positions = [], [], []
    for position in range(max(sequence_length - 1, start), end):
        window = values[position - sequence_length + 1 : position + 1]
        if not np.isfinite(window).all():
            continue
        if require_target and not np.isfinite(targets[position]):
            continue
        sequences.append(window)
        if require_target:
            labels.append(targets[position])
        positions.append(position)
    if not sequences:
        raise ValueError("Không thể tạo chuỗi hợp lệ từ các record đã cung cấp")
    x = np.stack(sequences)
    y = np.asarray(labels, dtype="float32") if require_target else None
    return x, y, np.asarray(positions)

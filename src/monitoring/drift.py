"""Thống kê tham chiếu và giám sát độ ổn định của population."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reference_statistics(frame: pd.DataFrame, columns: list[str]) -> dict:
    return {
        column: {
            "mean": float(frame[column].mean()),
            "std": float(frame[column].std()),
            "quantiles": [
                float(x) for x in frame[column].quantile([0, 0.1, 0.5, 0.9, 1])
            ],
        }
        for column in columns
    }


def standardized_mean_drift(frame: pd.DataFrame, reference: dict) -> dict[str, float]:
    scores = {}
    for column, stats in reference.items():
        denominator = max(abs(stats["std"]), 1e-8)
        scores[column] = float(abs(frame[column].mean() - stats["mean"]) / denominator)
    return scores


def input_warnings(
    frame: pd.DataFrame, reference: dict, threshold: float = 2.0
) -> list[str]:
    scores = standardized_mean_drift(frame, reference)
    return [
        f"{column}: mean drift={score:.2f} độ lệch chuẩn"
        for column, score in scores.items()
        if np.isfinite(score) and score > threshold
    ]

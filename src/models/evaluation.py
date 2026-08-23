"""Metric kỹ thuật, calibration và deployment gate."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(prediction))),
    }


def interval_half_width(
    actual: np.ndarray, prediction: np.ndarray, coverage: float
) -> float:
    return float(np.quantile(np.abs(actual - prediction), coverage))


def deployment_status(
    *,
    model_mae: float,
    zero_baseline_mae: float,
    interval_coverage: float,
    max_mae_ratio: float,
    minimum_coverage: float,
) -> tuple[str, list[str]]:
    reasons = []
    if model_mae > zero_baseline_mae * max_mae_ratio:
        reasons.append("MAE trên holdout không tốt hơn zero-return baseline")
    if interval_coverage < minimum_coverage:
        reasons.append("interval coverage trên holdout thấp hơn ngưỡng triển khai")
    return ("deployment_blocked" if reasons else "deployment_ready"), reasons

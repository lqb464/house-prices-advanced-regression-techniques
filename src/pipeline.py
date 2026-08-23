"""Pipeline raw OHLCV đến dự báo dùng chung cho batch và API inference."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.data.preprocessor import make_sequences
from src.data.schema import validate_ohlcv
from src.features.engineering import FEATURE_COLUMNS, build_feature_frame
from src.models.artifact import load_artifact
from src.models.trainers import predict_gru
from src.monitoring.drift import input_warnings


def predict_records(records, artifact_path: str | Path) -> dict:
    bundle, model = load_artifact(artifact_path)
    raw, warnings = validate_ohlcv(records, min_rows=bundle["minimum_inference_rows"])
    features = build_feature_frame(raw, bundle["horizon_sessions"])
    usable = features.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    sequences, _, positions = make_sequences(
        usable,
        bundle["scaler"],
        bundle["sequence_length"],
        require_target=False,
    )
    component_predictions = [float(predict_gru(model, sequences[-1:])[0])]
    traditional = bundle.get("traditional_models", {})
    if traditional:
        latest_row = usable.iloc[[-1]][FEATURE_COLUMNS]
        component_predictions.extend(
            float(estimator.predict(latest_row)[0])
            for estimator in traditional.values()
        )
    raw_prediction = float(np.mean(component_predictions))
    cap = float(bundle["prediction_cap"])
    displayed = float(np.clip(raw_prediction, -cap, cap))
    half_width = float(bundle["interval_half_width"])
    latest_features = usable.iloc[positions[-1] : positions[-1] + 1]
    warnings.extend(input_warnings(latest_features, bundle["reference_statistics"]))
    low, high = displayed - half_width, displayed + half_width
    warnings.extend(bundle.get("deployment_reasons", []))
    return {
        "ticker": bundle["ticker"],
        "as_of": usable.iloc[positions[-1]]["date"].date().isoformat(),
        "horizon_sessions": bundle["horizon_sessions"],
        "prediction_log_return": displayed,
        "prediction_percent": 100 * (math.exp(displayed) - 1),
        "interval": {
            "low": low,
            "high": high,
            "coverage_target": bundle["interval_target_coverage"],
        },
        "manual_review": low <= 0 <= high or bool(warnings),
        "input_warnings": warnings,
        "model_version": bundle["model_version"],
        "deployment_status": bundle["deployment_status"],
        "component_count": len(component_predictions),
        "disclaimer": "Chỉ dùng cho kịch bản nghiên cứu; không phải khuyến nghị đầu tư hay tín hiệu giao dịch tự động.",
    }

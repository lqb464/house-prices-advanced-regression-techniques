"""Train và đóng gói hybrid ensemble cho ticker đang được cấu hình."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_project_path
from src.data.loader import load_csv
from src.data.preprocessor import fit_scaler, make_sequences
from src.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN, build_feature_frame
from src.models.artifact import save_artifact
from src.models.evaluation import deployment_status, interval_half_width
from src.models.trainers import fit_traditional_models, predict_gru, train_gru
from src.monitoring.drift import reference_statistics


def train(config_path: str | Path, smoke_test: bool = False) -> dict:
    config = load_config(config_path)
    raw, data_warnings = load_csv(
        resolve_project_path(config["data_path"]),
        min_rows=config["population"]["min_rows"],
    )
    frame = build_feature_frame(raw, config["target"]["horizon_sessions"])
    frame = frame.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(
        drop=True
    )
    train_end = int(len(frame) * config["split"]["train_fraction"])
    validation_end = int(
        len(frame)
        * (config["split"]["train_fraction"] + config["split"]["validation_fraction"])
    )
    if smoke_test:
        train_end = min(train_end, 350)
        validation_end = min(validation_end, train_end + 100)

    scaler = fit_scaler(frame.iloc[:train_end])
    sequence_length = config["model"]["sequence_length"]
    x_train, y_train, _ = make_sequences(frame, scaler, sequence_length, end=train_end)
    x_validation, y_validation, _ = make_sequences(
        frame, scaler, sequence_length, start=train_end, end=validation_end
    )
    model, training = train_gru(
        x_train,
        y_train,
        x_validation,
        y_validation,
        hidden_size=config["model"]["hidden_size"],
        epochs=3 if smoke_test else config["model"]["epochs"],
        learning_rate=config["model"]["learning_rate"],
        batch_size=config["model"]["batch_size"],
        seed=config["split"]["random_seed"],
    )
    validation_prediction = predict_gru(model, x_validation)
    traditional = fit_traditional_models(
        frame, FEATURE_COLUMNS, TARGET_COLUMN, train_end, config["split"]["random_seed"]
    )
    half_width = interval_half_width(
        y_validation, validation_prediction, config["interval"]["coverage"]
    )

    report = json.loads(
        resolve_project_path(config["report_path"]).read_text(encoding="utf-8")
    )
    technical = report["metrics"]["technical"]
    zero_mae = report["metrics"]["zero_return_baseline_mae"]
    status, reasons = deployment_status(
        model_mae=technical["mae"],
        zero_baseline_mae=zero_mae,
        interval_coverage=report["metrics"]["interval_coverage_80"],
        max_mae_ratio=config["deployment_gate"]["max_mae_vs_zero_baseline"],
        minimum_coverage=config["deployment_gate"]["min_interval_coverage"],
    )
    trained_at = datetime.now(timezone.utc).isoformat()
    bundle = {
        "artifact_version": 1,
        "model_version": f"{config['ticker'].lower().replace('.', '-')}-hybrid-{trained_at[:10]}",
        "model_type": "hybrid_ensemble",
        "ticker": config["ticker"],
        "trained_at": trained_at,
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "horizon_sessions": config["target"]["horizon_sessions"],
        "sequence_length": sequence_length,
        "hidden_size": config["model"]["hidden_size"],
        "minimum_inference_rows": 50 + sequence_length,
        "prediction_cap": config["business"]["prediction_cap"],
        "interval_target_coverage": config["interval"]["coverage"],
        "interval_half_width": half_width,
        "scaler": scaler,
        "model_state": model.state_dict(),
        "traditional_models": traditional,
        "reference_statistics": reference_statistics(
            frame.iloc[:train_end], FEATURE_COLUMNS
        ),
        "training": training,
        "holdout_metrics": technical,
        "zero_return_baseline_mae": zero_mae,
        "deployment_status": status,
        "deployment_reasons": reasons,
        "data_warnings": data_warnings,
    }
    artifact_path = resolve_project_path(config["artifact_path"])
    checksum = save_artifact(bundle, artifact_path)
    metadata = {
        key: value
        for key, value in bundle.items()
        if key not in {"model_state", "scaler", "traditional_models"}
    }
    metadata["component_models"] = ["gru_rnn", *traditional.keys()]
    metadata["checksum_sha256"] = checksum
    metadata_path = artifact_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"artifact": str(artifact_path), "metadata": str(metadata_path), **metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.smoke_test), indent=2))


if __name__ == "__main__":
    main()

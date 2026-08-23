from sklearn.preprocessing import StandardScaler

from src.data.preprocessor import make_sequences
from src.features.engineering import FEATURE_COLUMNS, build_feature_frame
from src.models.artifact import load_artifact, save_artifact, sha256
from src.models.evaluation import deployment_status
from src.models.rnn import ReturnGRU
from src.pipeline import predict_records


def artifact_bundle(prices):
    frame = build_feature_frame(prices)
    usable = frame.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    scaler = StandardScaler().fit(usable[FEATURE_COLUMNS])
    model = ReturnGRU(len(FEATURE_COLUMNS), hidden_size=8)
    return {
        "artifact_version": 1,
        "model_version": "test-v1",
        "model_type": "gru_rnn",
        "ticker": "VIC.VN",
        "trained_at": "2026-08-26T00:00:00Z",
        "feature_columns": FEATURE_COLUMNS,
        "target": "target_return_5d",
        "horizon_sessions": 5,
        "sequence_length": 20,
        "hidden_size": 8,
        "minimum_inference_rows": 70,
        "prediction_cap": 0.20,
        "interval_target_coverage": 0.80,
        "interval_half_width": 0.05,
        "scaler": scaler,
        "model_state": model.state_dict(),
        "reference_statistics": {
            column: {
                "mean": float(usable[column].mean()),
                "std": float(usable[column].std()),
                "quantiles": [],
            }
            for column in FEATURE_COLUMNS
        },
        "deployment_status": "deployment_blocked",
        "deployment_reasons": ["gate kiểm thử"],
    }


def test_sequence_has_real_temporal_dimension(prices):
    frame = (
        build_feature_frame(prices)
        .dropna(subset=FEATURE_COLUMNS)
        .reset_index(drop=True)
    )
    scaler = StandardScaler().fit(frame[FEATURE_COLUMNS])
    sequences, _, _ = make_sequences(frame, scaler, 20, require_target=False)
    assert sequences.shape[1:] == (20, len(FEATURE_COLUMNS))


def test_artifact_roundtrip_and_checksum(tmp_path, prices):
    path = tmp_path / "model.pt"
    checksum = save_artifact(artifact_bundle(prices), path)
    loaded, model = load_artifact(path)
    assert checksum == sha256(path)
    assert loaded["model_version"] == "test-v1"
    assert isinstance(model, ReturnGRU)


def test_raw_to_prediction_handles_unseen_numeric_strings(tmp_path, prices):
    path = tmp_path / "model.pt"
    save_artifact(artifact_bundle(prices), path)
    raw = prices.copy()
    raw["volume"] = raw["volume"].astype(str)
    result = predict_records(raw, path)
    assert -20 <= result["prediction_percent"] <= 25
    assert result["manual_review"] is True
    assert result["deployment_status"] == "deployment_blocked"


def test_deployment_gate_has_business_meaning():
    status, reasons = deployment_status(
        model_mae=0.07,
        zero_baseline_mae=0.06,
        interval_coverage=0.44,
        max_mae_ratio=1.0,
        minimum_coverage=0.70,
    )
    assert status == "deployment_blocked"
    assert len(reasons) == 2

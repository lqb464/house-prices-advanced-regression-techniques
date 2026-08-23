"""Kiểm tra drift của feature mean trên batch OHLCV gần nhất."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_project_path
from src.data.schema import validate_ohlcv
from src.features.engineering import FEATURE_COLUMNS, build_feature_frame
from src.models.artifact import load_artifact
from src.monitoring.drift import standardized_mean_drift


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--input")
    args = parser.parse_args()
    config = load_config(args.config)
    path = resolve_project_path(args.input or config["data_path"])
    bundle, _ = load_artifact(resolve_project_path(config["artifact_path"]))
    raw, _ = validate_ohlcv(pd.read_csv(path), min_rows=70)
    recent = build_feature_frame(raw).dropna(subset=FEATURE_COLUMNS).tail(60)
    print(
        json.dumps(
            standardized_mean_drift(recent, bundle["reference_statistics"]), indent=2
        )
    )


if __name__ == "__main__":
    main()

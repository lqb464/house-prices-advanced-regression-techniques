"""Kiểm tra population đã cấu hình trước khi chạy notebook hoặc training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_project_path
from src.data.loader import load_csv


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    frame, warnings = load_csv(
        resolve_project_path(config["data_path"]),
        min_rows=config["population"]["min_rows"],
    )
    print(
        f"số dòng hợp lệ={len(frame)} bắt đầu={frame.date.min().date()} "
        f"kết thúc={frame.date.max().date()} cảnh báo={len(warnings)}"
    )


if __name__ == "__main__":
    main()

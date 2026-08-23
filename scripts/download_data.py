"""Tải và kiểm tra OHLCV ngày cho ticker được chọn rõ ràng."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import fetch_stock_data, provider_symbol


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="VIC")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frame, warnings = fetch_stock_data(args.ticker, start=args.start, end=args.end)
    output = args.output or Path("data/raw") / f"{provider_symbol(args.ticker).replace('.', '_')}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(
        f"Đã lưu {len(frame)} dòng {provider_symbol(args.ticker)} vào {output} "
        f"({frame.date.min().date()} đến {frame.date.max().date()})"
    )
    for warning in warnings:
        print(f"cảnh báo: {warning}")


if __name__ == "__main__":
    main()

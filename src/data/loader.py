"""Nạp dữ liệu mà không dùng synthetic fallback."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.data.schema import validate_ohlcv


def provider_symbol(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    return "VIC.VN" if cleaned == "VIC" else cleaned


def load_csv(
    path: str | Path, *, min_rows: int = 500
) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path)
    return validate_ohlcv(frame, min_rows=min_rows)


def fetch_stock_data(
    ticker: str,
    *,
    start: str = "2018-01-01",
    end: str | None = None,
    cache_directory: str | Path = "data/yfinance-cache",
    min_rows: int = 500,
) -> tuple[pd.DataFrame, list[str]]:
    yf.set_tz_cache_location(str(cache_directory))
    raw = yf.download(
        provider_symbol(ticker),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        timeout=30,
    )
    if raw.empty:
        raise ValueError(f"Không nhận được dữ liệu thị trường cho {provider_symbol(ticker)}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index().rename(columns=str.lower)
    return validate_ohlcv(raw, min_rows=min_rows)

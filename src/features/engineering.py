"""Các feature chống leakage đã được kiểm chứng trong notebook của project."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "volatility_10",
    "volatility_20",
    "volume_z_20",
    "rsi_14",
    "macd_scaled",
    "price_ma20",
    "price_ma50",
]
TARGET_COLUMN = "target_return_5d"


def build_feature_frame(raw: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    frame = raw.copy().sort_values("date").reset_index(drop=True)
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    returns = np.log(close / close.shift(1))
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26

    frame["return_1d"] = returns
    frame["return_5d"] = np.log(close / close.shift(5))
    frame["volatility_10"] = returns.rolling(10).std()
    frame["volatility_20"] = returns.rolling(20).std()
    frame["volume_z_20"] = (volume - volume.rolling(20).mean()) / volume.rolling(
        20
    ).std()
    frame["rsi_14"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))) / 100
    frame["macd_scaled"] = macd / close
    frame["price_ma20"] = close / close.rolling(20).mean() - 1
    frame["price_ma50"] = close / close.rolling(50).mean() - 1
    frame[TARGET_COLUMN] = np.log(close.shift(-horizon) / close)
    return frame.replace([np.inf, -np.inf], np.nan)


def production_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    return build_feature_frame(raw)[FEATURE_COLUMNS].dropna()

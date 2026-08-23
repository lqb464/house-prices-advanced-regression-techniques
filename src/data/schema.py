"""Kiểm tra schema OHLCV thô và các quy tắc population."""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataValidationError(ValueError):
    """Lỗi khi dữ liệu đầu vào nằm ngoài population được hỗ trợ."""


def validate_ohlcv(
    records: pd.DataFrame,
    *,
    min_rows: int = 50,
    sentinel_values: tuple[float, ...] = (-999, -9999),
) -> tuple[pd.DataFrame, list[str]]:
    missing = sorted(set(RAW_COLUMNS) - set(records.columns))
    if missing:
        raise DataValidationError(f"Thiếu các cột bắt buộc: {', '.join(missing)}")

    frame = records[RAW_COLUMNS].copy()
    warnings: list[str] = []
    frame["date"] = pd.to_datetime(
        frame["date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    for column in NUMERIC_COLUMNS:
        original_missing = frame[column].isna()
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].isna() & ~original_missing
        if invalid.any():
            warnings.append(
                f"{column}: đã chuyển {int(invalid.sum())} giá trị số không hợp lệ thành missing"
            )
        frame[column] = frame[column].replace(list(sentinel_values), np.nan)

    if frame["date"].isna().any():
        raise DataValidationError("Cột date chứa giá trị không hợp lệ")
    if frame["date"].duplicated().any():
        raise DataValidationError("Cột date chứa giá trị trùng lặp")
    if not frame["date"].is_monotonic_increasing:
        warnings.append("Các dòng đã được sắp xếp theo thời gian")
        frame = frame.sort_values("date").reset_index(drop=True)
    if frame[NUMERIC_COLUMNS].isna().any().any():
        bad = frame[NUMERIC_COLUMNS].isna().sum()
        details = ", ".join(f"{name}={count}" for name, count in bad.items() if count)
        raise DataValidationError(
            f"Các cột numeric chứa giá trị missing/không hợp lệ: {details}"
        )
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError("Giá phải lớn hơn 0")
    if (frame["volume"] < 0).any():
        raise DataValidationError("Volume không được âm")
    if len(frame) < min_rows:
        raise DataValidationError(
            f"Population yêu cầu ít nhất {min_rows} dòng; nhận được {len(frame)}"
        )
    return frame.reset_index(drop=True), warnings

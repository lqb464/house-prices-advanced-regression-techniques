import numpy as np
import pytest

from src.data.schema import DataValidationError, validate_ohlcv
from src.features.engineering import FEATURE_COLUMNS, TARGET_COLUMN, build_feature_frame


def test_schema_coerces_numeric_strings_and_sorts(prices):
    frame = prices.iloc[:80].copy().iloc[::-1].reset_index(drop=True)
    frame["volume"] = frame["volume"].astype(str)
    validated, warnings = validate_ohlcv(frame, min_rows=70)
    assert validated.date.is_monotonic_increasing
    assert np.issubdtype(validated.volume.dtype, np.number)
    assert any("sắp xếp" in warning for warning in warnings)


@pytest.mark.parametrize("bad_value", [None, "not-a-number", -999])
def test_schema_rejects_missing_invalid_and_sentinel(prices, bad_value):
    frame = prices.iloc[:80].copy()
    frame["close"] = frame["close"].astype(object)
    frame.loc[10, "close"] = bad_value
    with pytest.raises(DataValidationError, match="cột numeric"):
        validate_ohlcv(frame, min_rows=70)


def test_features_are_finite_and_exclude_audit_columns(prices):
    frame = build_feature_frame(prices).dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    assert np.isfinite(frame[FEATURE_COLUMNS].to_numpy()).all()
    assert not {"date", "ticker", TARGET_COLUMN}.intersection(FEATURE_COLUMNS)


def test_features_do_not_change_when_distant_future_changes(prices):
    changed = prices.copy()
    changed.loc[150:, "close"] *= 3
    left = build_feature_frame(prices).loc[120, FEATURE_COLUMNS].to_numpy(dtype=float)
    right = build_feature_frame(changed).loc[120, FEATURE_COLUMNS].to_numpy(dtype=float)
    assert np.allclose(left, right, equal_nan=True)


def test_target_is_future_five_session_return(prices):
    frame = build_feature_frame(prices)
    expected = np.log(prices.loc[65, "close"] / prices.loc[60, "close"])
    assert np.isclose(frame.loc[60, TARGET_COLUMN], expected)

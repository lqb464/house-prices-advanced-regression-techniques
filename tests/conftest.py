from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 180
    returns = rng.normal(0.0003, 0.015, rows)
    close = 80_000 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=rows),
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 8_000_000, rows),
        }
    )

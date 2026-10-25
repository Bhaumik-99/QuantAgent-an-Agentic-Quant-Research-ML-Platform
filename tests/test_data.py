import pandas as pd
import pytest

from quantagent.data import synthetic_prices, validate_prices


def test_synthetic_data_is_reproducible():
    pd.testing.assert_frame_equal(synthetic_prices(100, 42), synthetic_prices(100, 42))


def test_validator_rejects_nonpositive_prices():
    frame = synthetic_prices(100)
    frame.loc[0, "close"] = 0
    with pytest.raises(ValueError, match="positive"):
        validate_prices(frame)


def test_validator_sorts_timestamps():
    frame = synthetic_prices(100).iloc[::-1]
    result = validate_prices(frame)
    assert result.timestamp.is_monotonic_increasing

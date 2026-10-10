"""Strategies must decide using only the past. This is the #1 backtest killer."""

import pandas as pd
import pytest

from deflection.data import synthetic_ohlc
from deflection.strategies import STRATEGIES
from deflection.validation import assert_no_lookahead


@pytest.fixture(scope="module")
def prices():
    return synthetic_ohlc(n=6000, seed=7)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_shipped_strategies_only_use_the_past(prices, name):
    assert_no_lookahead(STRATEGIES[name], prices)


def test_checker_catches_a_strategy_that_peeks(prices):
    def cheater(df: pd.DataFrame) -> pd.DataFrame:
        next_close = df["close"].shift(-1)          # tomorrow's price - not allowed
        signal = (next_close > df["close"]).astype("int64") * 2 - 1
        return pd.DataFrame(
            {"signal": signal, "stop_dist": 0.001, "target_dist": 0.001}, index=df.index
        )

    with pytest.raises(AssertionError, match="Lookahead detected"):
        assert_no_lookahead(cheater, prices)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategies_actually_produce_signals(prices, name):
    out = STRATEGIES[name](prices)
    assert set(out.columns) == {"signal", "stop_dist", "target_dist"}
    assert out.index.equals(prices.index)
    assert (out["signal"] != 0).sum() > 10
    active = out[out["signal"] != 0]
    assert (active["stop_dist"] > 0).all() and (active["target_dist"] > 0).all()

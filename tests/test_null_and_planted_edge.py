"""The two tests that keep the backtester honest.

1. Random-walk data has no edge. Strategies must NOT look profitable on it,
   and with costs they must lose. A "profitable" result here means a bug.
2. Data with a real, planted edge MUST show up as profitable. Otherwise the
   engine could be hiding real edges.
"""

import numpy as np
import pandas as pd
import pytest

from deflection.backtest import BacktestConfig, run_backtest
from deflection.data import synthetic_ohlc
from deflection.instruments import get_instrument
from deflection.metrics import summarize
from deflection.strategies import STRATEGIES

EURUSD = get_instrument("EUR_USD")


@pytest.fixture(scope="module")
def random_walk():
    return synthetic_ohlc(n=100_000, seed=11)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_no_edge_appears_in_random_data_without_costs(random_walk, name):
    trades = run_backtest(
        random_walk, STRATEGIES[name](random_walk), EURUSD,
        BacktestConfig(spread=0.0, slippage_frac=0.0),
    ).trades
    s = summarize(trades)
    assert s["trades"] > 300
    assert abs(s["t_stat"]) < 3.0, s     # nothing statistically real in either direction


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_costs_make_random_data_lose(random_walk, name):
    s = summarize(run_backtest(random_walk, STRATEGIES[name](random_walk), EURUSD).trades)
    assert s["expectancy_r"] < 0
    assert s["costs_usd"] > 0


def test_planted_edge_is_found():
    rng = np.random.default_rng(3)
    n, every = 12_000, 40
    price = 1.1000
    rows, sig = [], np.zeros(n, dtype=np.int64)
    for i in range(n):
        o = price
        c = o + rng.normal(0, 0.00003)
        h, lo = max(o, c) + 0.00002, min(o, c) - 0.00002
        if i % every == 1 and i > 1:       # the bar after each signal
            if rng.random() < 0.6:
                h = o + 0.0040            # reaches the +0.0030 target
            else:
                lo = o - 0.0030           # reaches the -0.0020 stop
        rows.append((o, h, lo, c))
        price = c
        if i % every == 0 and i > 0:
            sig[i] = 1
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC", name="time")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    signals = pd.DataFrame({"signal": sig, "stop_dist": 0.0020, "target_dist": 0.0030}, index=idx)

    s = summarize(run_backtest(df, signals, EURUSD, BacktestConfig()).trades)
    # 60% x +1.5R + 40% x -1R = about +0.5R; ~300 trades, so allow sampling noise
    assert 0.25 < s["expectancy_r"] < 0.75, s
    assert s["t_stat"] > 3.0, s

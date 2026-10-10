"""Hand-built price paths where the right answer is known exactly."""

import pandas as pd
import pytest

from deflection.backtest import BacktestConfig, run_backtest
from deflection.instruments import get_instrument

EURUSD = get_instrument("EUR_USD")
NO_COSTS = dict(spread=0.0, slippage_frac=0.0, risk_per_trade=0.01)  # risk $100 of $10,000


def bars(rows, start="2024-01-02 08:00"):
    idx = pd.date_range(start, periods=len(rows), freq="15min", tz="UTC", name="time")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def signals(df, at=0, direction=1, stop=0.0020, target=0.0030):
    s = pd.DataFrame({"signal": 0, "stop_dist": stop, "target_dist": target}, index=df.index)
    s.loc[s.index[at], "signal"] = direction
    return s


QUIET = (1.1000, 1.1005, 1.0995, 1.1000)


def run(df, sig, instrument=EURUSD, **cfg):
    return run_backtest(df, sig, instrument, BacktestConfig(**cfg)).trades


def test_long_take_profit():
    df = bars([QUIET, (1.1000, 1.1035, 1.0990, 1.1030), QUIET])
    t = run(df, signals(df), **NO_COSTS)
    assert len(t) == 1
    row = t.iloc[0]
    assert row.exit_reason == "target"
    assert row.entry_price == pytest.approx(1.1000)
    assert row.exit_price == pytest.approx(1.1030)
    assert row.units == pytest.approx(50_000)          # $100 risk / 0.0020 stop
    assert row.pnl_usd == pytest.approx(150.0)
    assert row.r == pytest.approx(1.5)


def test_long_stop_loss():
    df = bars([QUIET, (1.1000, 1.1010, 1.0975, 1.0985), QUIET])
    row = run(df, signals(df), **NO_COSTS).iloc[0]
    assert row.exit_reason == "stop"
    assert row.r == pytest.approx(-1.0)


def test_stop_and_target_in_same_bar_assumes_the_worst():
    df = bars([QUIET, (1.1000, 1.1040, 1.0970, 1.1000), QUIET])
    row = run(df, signals(df), **NO_COSTS).iloc[0]
    assert row.exit_reason == "stop"
    assert row.r == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_the_open():
    df = bars([QUIET, (1.1000, 1.1005, 1.0995, 1.0990), (1.0950, 1.0960, 1.0940, 1.0955)])
    row = run(df, signals(df), **NO_COSTS).iloc[0]
    assert row.exit_reason == "stop_gap"
    assert row.exit_price == pytest.approx(1.0950)
    assert row.r == pytest.approx(-2.5)               # worse than -1R, like real gaps


def test_short_take_profit():
    df = bars([QUIET, (1.1000, 1.1010, 1.0965, 1.0970), QUIET])
    row = run(df, signals(df, direction=-1), **NO_COSTS).iloc[0]
    assert row.direction == "short"
    assert row.exit_reason == "target"
    assert row.r == pytest.approx(1.5)


def test_time_stop_exits_at_close():
    flat = (1.1000, 1.1005, 1.0995, 1.1002)
    df = bars([QUIET, flat, flat, flat, flat])
    row = run(df, signals(df), max_bars=3, **NO_COSTS).iloc[0]
    assert row.exit_reason == "time"
    assert row.bars_held == 3
    assert row.exit_time == df.index[3]
    assert row.r == pytest.approx(0.1)


def test_spread_is_paid_on_entry_and_exit():
    df = bars([QUIET, (1.1000, 1.1040, 1.0990, 1.1030), QUIET])
    row = run(df, signals(df), spread=0.0002, slippage_frac=0.0, risk_per_trade=0.01).iloc[0]
    assert row.entry_price == pytest.approx(1.1001)    # bought at the ask
    assert row.exit_price == pytest.approx(1.1031)     # target is measured from the real fill
    assert row.cost_usd == pytest.approx(0.0002 * row.units)


def test_spread_makes_stops_trigger_earlier():
    # Mid-price low of 1.0982 doesn't reach a 1.0980 stop - but the bid (mid - half spread) does
    df = bars([QUIET, (1.1000, 1.1005, 1.0982, 1.0990), QUIET])
    assert len(run(df, signals(df), **NO_COSTS).query("exit_reason == 'stop'")) == 0
    with_spread = run(df, signals(df), spread=0.0002, slippage_frac=0.0, risk_per_trade=0.01)
    assert with_spread.iloc[0].exit_reason == "stop"


def test_usd_jpy_pnl_is_converted_to_usd():
    df = bars([(150.00, 150.05, 149.95, 150.00), (150.00, 150.10, 149.60, 149.80), (149.8, 149.9, 149.7, 149.8)])
    sig = signals(df, stop=0.30, target=0.45)
    row = run(df, sig, instrument=get_instrument("USD_JPY"), **NO_COSTS).iloc[0]
    assert row.exit_reason == "stop"
    assert row.units == pytest.approx(50_000)                   # $100 / (0.30 JPY / 150)
    assert row.pnl_usd == pytest.approx(-0.30 * 50_000 / 149.70)  # JPY loss converted at the exit price
    assert row.r == pytest.approx(-1.002, abs=1e-3)


def test_leverage_cap_limits_size():
    df = bars([QUIET, (1.1000, 1.1005, 1.0995, 1.1000), QUIET])
    row = run(df, signals(df, stop=0.00001, target=0.00002), max_leverage=20.0, **NO_COSTS).iloc[0]
    assert row.units * row.entry_price <= 20.0 * 10_000 + 1e-6
    assert row.risk_usd < 100.0                        # cap shrank the risk actually taken


def test_one_position_at_a_time_and_no_entry_after_last_bar():
    df = bars([QUIET] * 6)
    sig = pd.DataFrame({"signal": 1, "stop_dist": 0.0020, "target_dist": 0.0030}, index=df.index)
    t = run(df, sig, max_bars=10, **NO_COSTS)
    assert len(t) == 1                                 # signals while in a trade are ignored
    assert t.iloc[0].exit_reason == "end_of_data"


def test_rejects_unknown_cross_instrument():
    with pytest.raises(ValueError, match="cross"):
        get_instrument("EUR_GBP")

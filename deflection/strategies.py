"""Strategies: turn candles into buy / sell / do-nothing decisions.

Contract every strategy follows (the backtester and, later, the live bot
rely on it):

    strategy(df) -> DataFrame with the same index as df and columns
        signal       +1 = buy, -1 = sell, 0 = nothing. Decided at the CLOSE
                     of that bar; the order is placed at the NEXT bar's open.
        stop_dist    distance from entry to stop-loss, in price units (> 0)
        target_dist  distance from entry to take-profit, in price units (> 0)

Rules:
  * Only use data up to and including the current bar. Never .shift(-1),
    never centered windows. deflection.validation.assert_no_lookahead
    checks this automatically - run it on every new strategy.
  * Parameters below are reasonable starting points, NOT tuned values.
    Tune only on in-sample data; judge only on out-of-sample data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import atr, bollinger, ema, rsi
from .patterns import bearish_engulfing, bearish_pin_bar, bullish_engulfing, bullish_pin_bar


@dataclass(frozen=True)
class PatternPullbackParams:
    ema_fast: int = 20             # "value area" the price pulls back into
    ema_slow: int = 200            # trend direction
    atr_period: int = 14
    stop_atr: float = 1.5          # stop = 1.5 x ATR away from entry
    target_r: float = 1.5          # take-profit = 1.5 x the stop distance
    min_range_atr: float = 0.5     # ignore tiny, meaningless candles
    session_start_utc: int | None = 6   # only enter between these UTC hours
    session_end_utc: int | None = 20    # (avoids thin, wide-spread hours)
    warmup_bars: int = 600         # let the slow EMA settle before trading


@dataclass(frozen=True)
class ReversionParams:
    bb_window: int = 20
    bb_stds: float = 2.0
    rsi_period: int = 14
    rsi_low: float = 30.0
    rsi_high: float = 70.0
    atr_period: int = 14
    stop_atr: float = 1.5
    min_target_atr: float = 0.25   # don't take trades whose target is basically zero
    session_start_utc: int | None = 6
    session_end_utc: int | None = 20
    warmup_bars: int = 100


def _session_mask(index: pd.DatetimeIndex, start: int | None, end: int | None) -> np.ndarray:
    if start is None or end is None:
        return np.ones(len(index), dtype=bool)
    hours = index.hour
    return np.asarray((hours >= start) & (hours < end))


def _warmup_mask(n: int, warmup: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[warmup:] = True
    return mask


def _frame(index, long_mask, short_mask, stop_dist, target_dist) -> pd.DataFrame:
    signal = np.where(long_mask, 1, np.where(short_mask, -1, 0)).astype(np.int64)
    out = pd.DataFrame(
        {
            "signal": signal,
            "stop_dist": np.asarray(stop_dist, dtype=float),
            "target_dist": np.asarray(target_dist, dtype=float),
        },
        index=index,
    )
    bad = ~np.isfinite(out["stop_dist"]) | (out["stop_dist"] <= 0) | ~np.isfinite(out["target_dist"]) | (out["target_dist"] <= 0)
    out.loc[bad, "signal"] = 0
    return out


def pattern_pullback(df: pd.DataFrame, p: PatternPullbackParams = PatternPullbackParams()) -> pd.DataFrame:
    """Trade WITH the trend after a deflection.

    Buy when: the trend is up (price and the fast EMA above the slow EMA),
    the bar dipped into the fast EMA, a bullish reversal candle formed
    (engulfing or hammer), and the bar closed back above the fast EMA.
    Sell is the mirror image. Stop and target are sized from ATR so they
    adapt to how volatile the market currently is.
    """
    close = df["close"]
    fast = ema(close, p.ema_fast)
    slow = ema(close, p.ema_slow)
    vol = atr(df, p.atr_period)

    uptrend = (close > slow) & (fast > slow)
    downtrend = (close < slow) & (fast < slow)
    big_enough = (df["high"] - df["low"]) >= p.min_range_atr * vol
    allowed = _session_mask(df.index, p.session_start_utc, p.session_end_utc) & _warmup_mask(len(df), p.warmup_bars)

    bullish = bullish_engulfing(df) | bullish_pin_bar(df)
    bearish = bearish_engulfing(df) | bearish_pin_bar(df)

    go_long = uptrend & (df["low"] <= fast) & (close > fast) & bullish & big_enough
    go_short = downtrend & (df["high"] >= fast) & (close < fast) & bearish & big_enough

    stop_dist = p.stop_atr * vol
    return _frame(
        df.index,
        np.asarray(go_long) & allowed,
        np.asarray(go_short) & allowed,
        stop_dist,
        p.target_r * stop_dist,
    )


def bb_rsi_reversion(df: pd.DataFrame, p: ReversionParams = ReversionParams()) -> pd.DataFrame:
    """Fade stretches: the MiniReversionStrategy idea, ported, in both directions.

    Buy when price closes at/below the lower Bollinger Band with RSI
    oversold; sell when it closes at/above the upper band with RSI
    overbought. Target = back to the middle band.
    """
    close = df["close"]
    lower, mid, upper = bollinger(close, p.bb_window, p.bb_stds)
    strength = rsi(close, p.rsi_period)
    vol = atr(df, p.atr_period)
    allowed = _session_mask(df.index, p.session_start_utc, p.session_end_utc) & _warmup_mask(len(df), p.warmup_bars)

    go_long = (close <= lower) & (strength < p.rsi_low)
    go_short = (close >= upper) & (strength > p.rsi_high)

    stop_dist = p.stop_atr * vol
    target_dist = (mid - close).abs()
    too_small = target_dist < p.min_target_atr * vol
    return _frame(
        df.index,
        np.asarray(go_long & ~too_small) & allowed,
        np.asarray(go_short & ~too_small) & allowed,
        stop_dist,
        target_dist,
    )


STRATEGIES = {
    "pattern": pattern_pullback,
    "reversion": bb_rsi_reversion,
}

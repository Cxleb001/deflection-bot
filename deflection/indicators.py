"""Technical indicators, written in plain pandas so every line is readable.

Every function here is causal: the value at bar t uses only bars 0..t.
tests/test_no_lookahead.py checks this for the strategies built on top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing). Measures typical bar size."""
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing), 0..100."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0, 100.0)                       # only gains -> 100
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)  # flat -> 50
    return out


def bollinger(close: pd.Series, window: int = 20, stds: float = 2.0):
    """Bollinger Bands. Returns (lower, middle, upper)."""
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    return mid - stds * sd, mid, mid + stds * sd

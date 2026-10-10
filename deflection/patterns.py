"""Bullish and bearish "deflection" candles.

Each function returns a boolean Series: True on the bar where the pattern
completes. Patterns are defined in plain words below so you can check them
against a chart. More patterns can be added, but each one has to earn its
place in out-of-sample testing - every extra rule is another chance to
overfit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _candle_parts(df: pd.DataFrame):
    body_top = df[["open", "close"]].max(axis=1)
    body_bottom = df[["open", "close"]].min(axis=1)
    bar_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    upper_wick = df["high"] - body_top
    lower_wick = body_bottom - df["low"]
    return bar_range, upper_wick, lower_wick


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """A down candle followed by an up candle whose body covers it."""
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    return (
        (prev_close < prev_open)
        & (df["close"] > df["open"])
        & (df["open"] <= prev_close)
        & (df["close"] >= prev_open)
    )


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """An up candle followed by a down candle whose body covers it."""
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    return (
        (prev_close > prev_open)
        & (df["close"] < df["open"])
        & (df["open"] >= prev_close)
        & (df["close"] <= prev_open)
    )


def bullish_pin_bar(df: pd.DataFrame, min_wick: float = 0.6, max_opposite_wick: float = 0.25) -> pd.Series:
    """Hammer: long lower wick (sellers pushed down and got rejected), small upper wick."""
    bar_range, upper_wick, lower_wick = _candle_parts(df)
    return (lower_wick / bar_range >= min_wick) & (upper_wick / bar_range <= max_opposite_wick)


def bearish_pin_bar(df: pd.DataFrame, min_wick: float = 0.6, max_opposite_wick: float = 0.25) -> pd.Series:
    """Shooting star: long upper wick (buyers pushed up and got rejected), small lower wick."""
    bar_range, upper_wick, lower_wick = _candle_parts(df)
    return (upper_wick / bar_range >= min_wick) & (lower_wick / bar_range <= max_opposite_wick)

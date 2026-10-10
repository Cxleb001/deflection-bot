"""Guards against the most common way backtests lie: using future data."""

from __future__ import annotations

from typing import Callable

import pandas as pd


def assert_no_lookahead(
    strategy: Callable[[pd.DataFrame], pd.DataFrame],
    df: pd.DataFrame,
    cut_fractions: tuple[float, ...] = (0.3, 0.5, 0.8),
) -> None:
    """Fail if a strategy's decision at bar t changes when future bars are removed.

    Runs the strategy on the full history and on truncated copies. A strategy
    that only uses the past produces identical signals on the shared bars. A
    strategy that peeks ahead (shift(-1), centered windows, normalizing with
    whole-dataset statistics, ...) will not, and this raises AssertionError.
    """
    full = strategy(df)
    for frac in cut_fractions:
        cut = int(len(df) * frac)
        partial = strategy(df.iloc[:cut])
        try:
            pd.testing.assert_frame_equal(full.iloc[:cut], partial, check_freq=False)
        except AssertionError as exc:
            raise AssertionError(
                f"Lookahead detected: signals for bars before the cut at {cut} "
                f"changed when later bars were removed.\n{exc}"
            ) from None

"""Bar-by-bar backtester with realistic costs.

How a trade is simulated (all deliberately pessimistic where it matters):
  * A signal decided at the close of bar t is entered at the OPEN of bar t+1.
  * Buys fill at the ask (mid + half spread) plus slippage; sells at the bid.
  * Stop-loss and take-profit are checked against bid prices for longs and
    ask prices for shorts, so the spread is paid on the way out too.
  * If one bar touches both the stop and the target, we assume the STOP was
    hit first. Candles don't tell us the order, so we take the worse case.
  * If price gaps past the stop at a bar's open, the fill is at that open,
    which can be worse than -1R. Real markets do this; so does this model.
  * Stops and time exits are market orders and pay slippage. Take-profits
    are limit orders and don't.
  * Position size = (equity x risk_per_trade) / loss-at-stop, capped by
    max_leverage. One position at a time per instrument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .instruments import Instrument


@dataclass
class BacktestConfig:
    starting_equity: float = 10_000.0
    risk_per_trade: float = 0.005      # 0.5% of equity lost if the stop is hit
    max_leverage: float = 20.0         # notional / equity cap
    max_bars: int = 48                 # time stop: exit at the close of this many bars
    slippage_frac: float = 0.25        # extra cost on market fills, as a fraction of the spread
    spread: float | None = None        # force a fixed spread (price units); else use data, else default


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    config: BacktestConfig
    instrument: Instrument
    final_equity: float
    spread_source: str = field(default="data")


TRADE_COLUMNS = [
    "entry_time", "exit_time", "direction", "entry_price", "exit_price",
    "stop_price", "target_price", "units", "risk_usd", "pnl_usd", "r",
    "cost_usd", "bars_held", "exit_reason",
]


def _resolve_spread(df: pd.DataFrame, instrument: Instrument, cfg: BacktestConfig) -> tuple[np.ndarray, str]:
    n = len(df)
    if cfg.spread is not None:
        return np.full(n, float(cfg.spread)), f"fixed {cfg.spread}"
    if "spread" in df.columns and df["spread"].notna().all():
        return df["spread"].to_numpy(dtype=float), "per-bar 'spread' column in the data"
    if instrument.default_spread is not None:
        return np.full(n, instrument.default_spread), f"placeholder default {instrument.default_spread}"
    raise ValueError(
        f"No spread available for {instrument.name}: the data has no 'spread' column and there's "
        f"no default. Pass a spread explicitly (e.g. --spread 0.0002)."
    )


def run_backtest(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    instrument: Instrument,
    cfg: BacktestConfig | None = None,
) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    if not df.index.equals(signals.index):
        raise ValueError("signals must have exactly the same index as the price data")

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    spread, spread_source = _resolve_spread(df, instrument, cfg)
    sig = signals["signal"].to_numpy()
    stop_d = signals["stop_dist"].to_numpy(dtype=float)
    target_d = signals["target_dist"].to_numpy(dtype=float)
    times = df.index
    n = len(df)

    equity = cfg.starting_equity
    pending = None   # (direction, stop_dist, target_dist)
    pos = None       # dict while a trade is open
    trades = []

    def close_position(i: int, exit_price: float, reason: str, exit_cost: float):
        nonlocal equity, pos
        d = pos["direction"]
        pnl_quote = d * (exit_price - pos["entry_price"]) * pos["units"]
        pnl_usd = instrument.to_usd(pnl_quote, exit_price)
        cost_usd = instrument.to_usd(pos["units"] * (pos["entry_cost"] + exit_cost), exit_price)
        equity += pnl_usd
        trades.append({
            "entry_time": times[pos["entry_i"]],
            "exit_time": times[i],
            "direction": "long" if d == 1 else "short",
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "stop_price": pos["stop"],
            "target_price": pos["target"],
            "units": pos["units"],
            "risk_usd": pos["risk_usd"],
            "pnl_usd": pnl_usd,
            "r": pnl_usd / pos["risk_usd"],
            "cost_usd": cost_usd,
            "bars_held": pos["bars_held"],
            "exit_reason": reason,
        })
        pos = None

    for i in range(n):
        half = spread[i] / 2.0
        slip = cfg.slippage_frac * spread[i]

        # 1) Fill an entry queued at the previous bar's close.
        if pending is not None and pos is None:
            d, sdist, tdist = pending
            pending = None
            entry = o[i] + d * (half + slip)
            risk_usd = equity * cfg.risk_per_trade
            loss_per_unit_usd = instrument.to_usd(sdist, entry)
            units = risk_usd / loss_per_unit_usd if loss_per_unit_usd > 0 else 0.0
            max_units = cfg.max_leverage * equity / instrument.usd_notional_per_unit(entry)
            units = min(units, max_units)
            if units > 0 and math.isfinite(units) and equity > 0:
                # Risk actually taken can be smaller than planned if the leverage cap bit.
                actual_risk = units * loss_per_unit_usd
                pos = {
                    "direction": d, "entry_i": i, "entry_price": entry,
                    "stop": entry - d * sdist, "target": entry + d * tdist,
                    "units": units, "risk_usd": actual_risk, "entry_cost": half + slip,
                    "bars_held": 0,
                }

        # 2) Manage the open position during this bar (including the entry bar).
        if pos is not None:
            d = pos["direction"]
            first_bar = pos["entry_i"] == i
            if d == 1:   # long: we would sell at the bid
                at_open, worst, best = o[i] - half, l[i] - half, h[i] - half
                stop_gapped = not first_bar and at_open <= pos["stop"]
                target_gapped = not first_bar and at_open >= pos["target"]
                stop_hit = worst <= pos["stop"]
                target_hit = best >= pos["target"]
            else:        # short: we would buy back at the ask
                at_open, worst, best = o[i] + half, h[i] + half, l[i] + half
                stop_gapped = not first_bar and at_open >= pos["stop"]
                target_gapped = not first_bar and at_open <= pos["target"]
                stop_hit = worst >= pos["stop"]
                target_hit = best <= pos["target"]

            if stop_gapped:
                close_position(i, at_open - d * slip, "stop_gap", half + slip)
            elif target_gapped:
                close_position(i, at_open, "target_gap", half)
            elif stop_hit:                       # checked before target on purpose
                close_position(i, pos["stop"] - d * slip, "stop", half + slip)
            elif target_hit:
                close_position(i, pos["target"], "target", half)
            else:
                pos["bars_held"] += 1
                if pos["bars_held"] >= cfg.max_bars:
                    exit_px = c[i] - d * (half + slip)
                    close_position(i, exit_px, "time", half + slip)

        # 3) Queue a new entry from this bar's signal (filled at the next open).
        if pos is None and pending is None and i + 1 < n and sig[i] != 0:
            if math.isfinite(stop_d[i]) and stop_d[i] > 0 and math.isfinite(target_d[i]) and target_d[i] > 0:
                pending = (int(sig[i]), float(stop_d[i]), float(target_d[i]))

    if pos is not None:   # still open when the data ends
        half = spread[-1] / 2.0
        slip = cfg.slippage_frac * spread[-1]
        d = pos["direction"]
        close_position(n - 1, c[-1] - d * (half + slip), "end_of_data", half + slip)

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    return BacktestResult(trades_df, cfg, instrument, equity, spread_source)

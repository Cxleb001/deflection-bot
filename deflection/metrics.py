"""Turn a list of trades into the numbers that actually matter.

Win rate on its own says almost nothing: a 35% win rate can make money and
a 75% win rate can lose it. The core number is EXPECTANCY in R - the average
result per trade, measured in units of the amount risked. +0.10R means each
trade earns 10% of what it risks, on average, after costs.

The t-stat answers "could this just be luck?". Roughly, t >= 2 is the bare
minimum before believing a result - and if you have tried many variants,
demand more, because one of twenty random variants will clear t = 2 by
chance alone.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _max_drawdown_r(r: pd.Series) -> float:
    curve = r.cumsum()
    peak = np.maximum.accumulate(np.concatenate([[0.0], curve.to_numpy()]))[1:]
    return float((curve.to_numpy() - peak).min()) if len(curve) else 0.0


def summarize(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    r = trades["r"].astype(float)
    wins = r[r > 0]
    losses = r[r <= 0]
    gross_win = trades.loc[trades["pnl_usd"] > 0, "pnl_usd"].sum()
    gross_loss = -trades.loc[trades["pnl_usd"] <= 0, "pnl_usd"].sum()
    net = trades["pnl_usd"].sum()
    costs = trades["cost_usd"].sum()
    std = r.std(ddof=1) if n > 1 else float("nan")
    se = std / math.sqrt(n) if n > 1 else float("nan")
    return {
        "trades": n,
        "win_rate": float((r > 0).mean()),
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "expectancy_r": float(r.mean()),
        "expectancy_se": float(se),
        "t_stat": float(r.mean() / se) if se and se > 0 else float("nan"),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown_r": _max_drawdown_r(r),
        "net_pnl_usd": float(net),
        "costs_usd": float(costs),
        "pnl_before_costs_usd": float(net + costs),
        "avg_bars_held": float(trades["bars_held"].mean()),
        "longs": int((trades["direction"] == "long").sum()),
        "shorts": int((trades["direction"] == "short").sum()),
    }


def by_year(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["trades", "expectancy_r", "net_pnl_usd"])
    years = pd.to_datetime(trades["entry_time"]).dt.year
    grouped = trades.groupby(years)
    return pd.DataFrame({
        "trades": grouped.size(),
        "expectancy_r": grouped["r"].mean(),
        "net_pnl_usd": grouped["pnl_usd"].sum(),
    })


def split_in_out(trades: pd.DataFrame, oos_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    entry = pd.to_datetime(trades["entry_time"])
    return trades[entry < oos_start], trades[entry >= oos_start]


def format_summary(title: str, s: dict) -> str:
    if s.get("trades", 0) == 0:
        return f"{title}\n  no trades\n"
    lines = [
        title,
        f"  trades            {s['trades']}  ({s['longs']} long / {s['shorts']} short)",
        f"  expectancy        {s['expectancy_r']:+.3f} R per trade  (+/- {s['expectancy_se']:.3f}, t = {s['t_stat']:.2f})",
        f"  win rate          {s['win_rate']:.1%}   avg win {s['avg_win_r']:+.2f} R   avg loss {s['avg_loss_r']:+.2f} R",
        f"  profit factor     {s['profit_factor']:.2f}",
        f"  max drawdown      {s['max_drawdown_r']:.1f} R",
        f"  net P&L           ${s['net_pnl_usd']:,.2f}   (before costs ${s['pnl_before_costs_usd']:,.2f}, costs ${s['costs_usd']:,.2f})",
        f"  avg hold          {s['avg_bars_held']:.1f} bars",
    ]
    return "\n".join(lines) + "\n"

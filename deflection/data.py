"""Price data: load/save CSVs, generate synthetic test data, download from OANDA.

CSV format (UTC times):
    time,open,high,low,close[,volume][,spread]

`spread` is optional and in price units. When present (OANDA downloads
include it), the backtester charges the real historical spread instead of a
guess.
"""

from __future__ import annotations

import os
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

PRACTICE_URL = "https://api-fxpractice.oanda.com"
LIVE_URL = "https://api-fxtrade.oanda.com"
MAX_CANDLES_PER_REQUEST = 5000

PRICE_COLUMNS = ["open", "high", "low", "close"]


# ---------------------------------------------------------------- CSV files

def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = next((c for c in ("time", "timestamp", "date", "datetime") if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"{path}: needs a 'time' column")
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.rename(columns={time_col: "time"}).set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for col in PRICE_COLUMNS + [c for c in ("volume", "spread") if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=PRICE_COLUMNS)
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    if bad.any():
        raise ValueError(f"{path}: {int(bad.sum())} rows have high/low inconsistent with open/close")
    return df


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = out.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.index.name = "time"
    out.to_csv(path)


# ---------------------------------------------------------- synthetic data

def synthetic_ohlc(
    n: int = 100_000,
    start_price: float = 1.10,
    bar_vol: float = 0.0006,
    seed: int = 0,
    freq: str = "15min",
    start: str = "2020-01-01",
    spread: float | None = 0.00014,
) -> pd.DataFrame:
    """A pure random walk. There is NO edge in this data by construction.

    Any strategy that shows a reliable profit here has a bug (usually
    lookahead). An honest strategy should lose roughly its trading costs.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, bar_vol, n)
    close = start_price * np.exp(np.cumsum(log_returns))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick_up = np.abs(rng.normal(0.0, bar_vol * 0.6, n)) * close
    wick_dn = np.abs(rng.normal(0.0, bar_vol * 0.6, n)) * close
    high = np.maximum(open_, close) + wick_up
    low = np.minimum(open_, close) - wick_dn
    index = pd.date_range(start=start, periods=n, freq=freq, tz="UTC", name="time")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=index)
    if spread is not None:
        df["spread"] = spread
    return df


# ------------------------------------------------------------------- OANDA

def candles_to_frame(candles: list[dict]) -> pd.DataFrame:
    """Turn OANDA bid/ask candles into mid OHLC plus the spread at each bar's open."""
    rows = []
    for cdl in candles:
        if not cdl.get("complete", False):
            continue  # the still-forming current candle
        bid, ask = cdl["bid"], cdl["ask"]
        b = {k: float(bid[k]) for k in ("o", "h", "l", "c")}
        a = {k: float(ask[k]) for k in ("o", "h", "l", "c")}
        rows.append({
            "time": pd.Timestamp(cdl["time"]),
            "open": (b["o"] + a["o"]) / 2,
            "high": (b["h"] + a["h"]) / 2,
            "low": (b["l"] + a["l"]) / 2,
            "close": (b["c"] + a["c"]) / 2,
            "volume": int(cdl.get("volume", 0)),
            "spread": a["o"] - b["o"],
        })
    if not rows:
        return pd.DataFrame(columns=PRICE_COLUMNS + ["volume", "spread"],
                            index=pd.DatetimeIndex([], tz="UTC", name="time"))
    df = pd.DataFrame(rows).set_index("time")
    df.index = pd.DatetimeIndex(df.index, tz="UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df.index.name = "time"
    return df


def _rfc3339(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def download_oanda(
    instrument: str,
    granularity: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    *,
    token: str | None = None,
    env: str = "practice",
    session=None,
    page_size: int = MAX_CANDLES_PER_REQUEST,
    pause_seconds: float = 0.1,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Download historical bid/ask candles from OANDA's v20 REST API.

    The token is read from the OANDA_API_TOKEN environment variable unless
    passed in. It is only ever sent in the Authorization header - never put
    it in code, config files or chat.
    """
    token = token or os.environ.get("OANDA_API_TOKEN")
    if not token:
        raise RuntimeError(
            "OANDA_API_TOKEN is not set. Create a practice account, generate an API token "
            "in OANDA's account hub, and set it as an environment variable (or a Codespaces secret)."
        )
    if env not in ("practice", "live"):
        raise ValueError("env must be 'practice' or 'live'")
    if session is None:
        import requests
        session = requests.Session()

    base = PRACTICE_URL if env == "practice" else LIVE_URL
    url = f"{base}/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {token}", "Accept-Datetime-Format": "RFC3339"}

    start_ts = pd.Timestamp(start)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tz is None else start_ts.tz_convert("UTC")
    end_ts = None
    if end is not None:
        end_ts = pd.Timestamp(end)
        end_ts = end_ts.tz_localize("UTC") if end_ts.tz is None else end_ts.tz_convert("UTC")

    frames = []
    cursor = start_ts
    first_page = True
    while True:
        params = {
            "price": "BA",
            "granularity": granularity,
            "from": _rfc3339(cursor),
            "count": page_size,
            "includeFirst": "true" if first_page else "false",
        }
        for attempt in range(max_retries):
            resp = session.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:          # rate limited: back off and retry
                _time.sleep(1.0 * (attempt + 1))
                continue
            break
        if resp.status_code != 200:
            raise RuntimeError(f"OANDA returned HTTP {resp.status_code} for {instrument}: {resp.text[:300]}")

        candles = resp.json().get("candles", [])
        if not candles:
            break
        frames.append(candles_to_frame(candles))
        last_time = pd.Timestamp(candles[-1]["time"])
        last_time = last_time.tz_localize("UTC") if last_time.tz is None else last_time.tz_convert("UTC")
        if end_ts is not None and last_time >= end_ts:
            break
        if len(candles) < page_size:
            break                                # reached the present
        cursor = last_time
        first_page = False
        _time.sleep(pause_seconds)

    if not frames:
        return candles_to_frame([])
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df[df.index >= start_ts]
    if end_ts is not None:
        df = df[df.index < end_ts]
    return df

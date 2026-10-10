"""Command line:

    python -m deflection synthetic --out data/SYNTH_M15.csv
    python -m deflection download --instrument EUR_USD --granularity M15 --start 2022-01-01 --out data/EUR_USD_M15.csv
    python -m deflection backtest --csv data/EUR_USD_M15.csv --instrument EUR_USD --strategy pattern
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .data import download_oanda, load_csv, save_csv, synthetic_ohlc
from .instruments import get_instrument
from .metrics import by_year, format_summary, split_in_out, summarize
from .strategies import STRATEGIES


def cmd_synthetic(args) -> int:
    df = synthetic_ohlc(n=args.bars, seed=args.seed)
    save_csv(df, args.out)
    print(f"Wrote {len(df):,} random-walk bars to {args.out} (no edge exists in this data by design).")
    return 0


def cmd_download(args) -> int:
    print(f"Downloading {args.instrument} {args.granularity} from {args.start} ({args.env} API)...")
    df = download_oanda(args.instrument, args.granularity, args.start, args.end, env=args.env)
    if df.empty:
        print("No candles returned - check the instrument name, dates and token.")
        return 1
    save_csv(df, args.out)
    print(f"Wrote {len(df):,} candles ({df.index[0]} -> {df.index[-1]}) to {args.out}")
    print(f"Median spread: {df['spread'].median():.6f}   95th percentile: {df['spread'].quantile(0.95):.6f}")
    return 0


def cmd_backtest(args) -> int:
    df = load_csv(args.csv)
    instrument = get_instrument(args.instrument)
    strategy = STRATEGIES[args.strategy]
    signals = strategy(df)
    cfg = BacktestConfig(
        risk_per_trade=args.risk,
        max_bars=args.max_bars,
        spread=args.spread,
    )
    result = run_backtest(df, signals, instrument, cfg)
    trades = result.trades

    if args.oos_start:
        oos_start = pd.Timestamp(args.oos_start, tz="UTC")
    else:
        oos_start = df.index[int(len(df) * 0.7)]
    in_sample, out_sample = split_in_out(trades, oos_start)

    print("=" * 72)
    print(f"{args.strategy} strategy on {instrument.name}   {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}   ({len(df):,} bars)")
    print(f"spread: {result.spread_source}   risk/trade: {cfg.risk_per_trade:.2%}   time stop: {cfg.max_bars} bars")
    print("=" * 72)
    print(format_summary(f"IN-SAMPLE (before {oos_start:%Y-%m-%d}) - tune here", summarize(in_sample)))
    print(format_summary(f"OUT-OF-SAMPLE (from {oos_start:%Y-%m-%d}) - judge here, never tune here", summarize(out_sample)))
    total = summarize(trades)
    print(format_summary("ALL TRADES", total))
    print(f"Equity: ${cfg.starting_equity:,.0f} -> ${result.final_equity:,.2f}")
    print()
    print("By year:")
    yearly = by_year(trades)
    if yearly.empty:
        print("  no trades")
    else:
        for year, row in yearly.iterrows():
            print(f"  {year}   {int(row['trades']):5d} trades   {row['expectancy_r']:+.3f} R/trade   ${row['net_pnl_usd']:,.2f}")
    if not trades.empty:
        print()
        print("Exit reasons:", trades["exit_reason"].value_counts().to_dict())

    if args.trades_out:
        Path(args.trades_out).parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(args.trades_out, index=False)
        print(f"\nTrade log written to {args.trades_out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m deflection")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synthetic", help="write random-walk test data (no edge by design)")
    s.add_argument("--out", default="data/SYNTH_M15.csv")
    s.add_argument("--bars", type=int, default=100_000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synthetic)

    d = sub.add_parser("download", help="download bid/ask candles from OANDA (needs OANDA_API_TOKEN)")
    d.add_argument("--instrument", required=True, help="e.g. EUR_USD, XAU_USD")
    d.add_argument("--granularity", default="M15", help="M5, M15, M30, H1, H4, D")
    d.add_argument("--start", required=True, help="e.g. 2022-01-01")
    d.add_argument("--end", default=None)
    d.add_argument("--env", default="practice", choices=["practice", "live"])
    d.add_argument("--out", required=True)
    d.set_defaults(func=cmd_download)

    b = sub.add_parser("backtest", help="run a strategy over a CSV and report in/out-of-sample results")
    b.add_argument("--csv", required=True)
    b.add_argument("--instrument", required=True, help="e.g. EUR_USD (sets currency conversion and fallback spread)")
    b.add_argument("--strategy", default="pattern", choices=sorted(STRATEGIES))
    b.add_argument("--oos-start", default=None, help="out-of-sample start date; default = last 30%% of the data")
    b.add_argument("--risk", type=float, default=0.005, help="fraction of equity risked per trade (default 0.005)")
    b.add_argument("--max-bars", type=int, default=48, help="time stop in bars (default 48)")
    b.add_argument("--spread", type=float, default=None, help="force a fixed spread in price units")
    b.add_argument("--trades-out", default=None, help="optional CSV path for the trade log")
    b.set_defaults(func=cmd_backtest)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

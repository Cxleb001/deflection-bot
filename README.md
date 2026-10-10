# deflection-bot

An autonomous FX and metals trading bot, built in the order that keeps it honest:

1. **Research engine (this release).** Detect bullish/bearish "deflection" patterns, turn them into buy/sell decisions, and backtest them with real costs - so we find out whether a signal actually has an edge *before* it touches an order.
2. **Execution engine** on an OANDA practice (demo) account: orders with broker-side stop-loss and take-profit, a risk manager, a trade journal, Telegram alerts.
3. **Forward test** on the demo account, comparing live results against the backtest.

See [ROADMAP.md](ROADMAP.md) for dates and the exit criteria for each phase.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest                  # 27 tests: accounting, no-lookahead, random-walk, planted edge, downloader

# A baseline on fake data with NO edge in it (by design):
python -m deflection synthetic --out data/SYNTH_M15.csv
python -m deflection backtest --csv data/SYNTH_M15.csv --instrument EUR_USD --strategy pattern
```

That baseline should **lose** a little - roughly its trading costs. That's the engine working: on random data, an honest backtester can't find profit. If a future change makes random data look profitable, something is broken.

## Real data (OANDA practice account)

1. Open a free practice (demo) account at OANDA and generate an API token in its account hub. Check that OANDA accepts accounts in your country first.
2. Store the token as a **Codespaces secret** named `OANDA_API_TOKEN` (GitHub - Settings - Codespaces - Secrets), or in a local `.env` that git ignores. Never in code, commits or chats.
3. Download history and backtest:

```bash
python -m deflection download --instrument EUR_USD --granularity M15 --start 2022-01-01 --out data/EUR_USD_M15.csv
python -m deflection backtest --csv data/EUR_USD_M15.csv --instrument EUR_USD --strategy pattern --oos-start 2025-01-01
```

Downloads use bid/ask candles, so backtests charge the real historical spread, not a guess. Good first instruments: `EUR_USD`, `GBP_USD`, `USD_JPY`, `XAU_USD` (gold), `XAG_USD` (silver). Try `M15`, `H1` and `H4`.

## Reading the report

| Number | What it tells you |
|---|---|
| **Expectancy (R)** | Average result per trade in units of what was risked, after costs. The number that matters. |
| **t-stat** | Could this be luck? Below 2: probably. Tried many variants? Demand more. |
| **Out-of-sample** | Results on data you did not tune on. Tune on in-sample only. Judge on out-of-sample only. |
| **Before costs vs costs** | How much the spread and slippage ate. Smaller timeframes pay far more of their edge in costs. |
| **By year** | An edge that only exists in one year is probably not an edge. |
| Win rate | On its own, close to meaningless. 35% can make money; 75% can lose it. |

## Research rules

- Write down every variant you test (a simple `research_log.md`). Twenty variants tested means one will look good by chance.
- Never tune on the out-of-sample period. Once you have looked at it to choose parameters, it isn't out-of-sample anymore.
- Need a few hundred trades before believing anything.
- Every new strategy must pass `deflection.validation.assert_no_lookahead`. Add it to `tests/test_no_lookahead.py` - it runs automatically for everything in `STRATEGIES`.

## Layout

```
deflection/
  indicators.py   EMA, ATR, RSI, Bollinger - causal, plain pandas
  patterns.py     engulfing and pin-bar (hammer / shooting star) detection
  strategies.py   pattern_pullback (trend + deflection) and bb_rsi_reversion
  backtest.py     bar-by-bar simulator: next-bar entries, bid/ask, slippage, gaps, sizing
  metrics.py      expectancy, t-stat, profit factor, drawdown, by-year
  instruments.py  USD conversion, placeholder spreads
  data.py         CSV I/O, random-walk generator, OANDA downloader
  validation.py   lookahead detector
  cli.py          python -m deflection ...
tests/
```

## Adding a strategy

Write a function `df -> DataFrame[signal, stop_dist, target_dist]` (see the contract at the top of `strategies.py`), add it to `STRATEGIES`, run `python -m pytest`, then backtest it.

## Not built yet

Live order execution, the risk manager, the trade journal and Telegram control are Phase 2. The backtester and the live bot will share the same strategy code, so what gets tested is what gets traded.

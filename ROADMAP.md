# Roadmap: Oct 10 - Dec 31, 2026

Goal for year end: an autonomous bot that scans FX and metals, decides buy or sell on its own, runs unattended on a demo account, and has a measured, honest track record - plus a data-backed go/no-go on whether any of it deserves real money.

Being profitable is a hoped-for outcome, not a milestone with a date. Each phase has a gate. If a gate isn't met, the plan changes - we don't skip ahead.

## Phase 0 - Setup (Oct 10 - Oct 18)

- [ ] New GitHub repo for this project; open a Codespace from that repo
- [ ] OANDA practice account + API token stored as a Codespaces secret
- [ ] `python -m pytest` passes in the Codespace
- [ ] Download 3+ years of M15, H1 and H4 for EUR_USD, GBP_USD, USD_JPY, XAU_USD, XAG_USD
- [ ] Run both strategies on all of it; save the reports as the baseline

**Gate:** real data downloads and backtests run end to end.

## Phase 1 - Find an edge, or prove there isn't one (Oct 19 - Nov 8)

- [ ] Keep `research_log.md`: every variant, its in-sample and out-of-sample result
- [ ] Explore: timeframes, session filters, pattern definitions, stop/target sizes, trend filters
- [ ] Walk-forward check: does a config chosen on 2022-2023 hold up on 2024 and 2025?
- [ ] Port DeflectionHunter's idea in as a third strategy, if it's worth keeping

**Gate (all of these, out-of-sample, after costs):**
- expectancy of at least about +0.10 R per trade
- t-stat of 2 or more, on 150+ trades
- positive in most years, and on at least two instruments

If nothing passes: that's a real result. Try a different idea family (breakouts, multi-timeframe trend) rather than squeezing parameters until something passes by luck.

## Phase 2 - Execution engine on the demo account (Nov 9 - Nov 29)

- [ ] OANDA order placement with stop-loss and take-profit attached on the broker's side, so positions stay protected if the bot crashes
- [ ] Reconcile on startup: compare the bot's view of open positions with the broker's
- [ ] SQLite trade journal (every signal, order, fill, exit, with reasons)
- [ ] Risk manager, enforced in code no matter what the strategy says: risk per trade, max open positions, max total exposure, daily loss circuit breaker, manual kill switch
- [ ] Telegram: alerts, /status, /pause, /kill (reusing what you built for the Avalanche bot)
- [ ] Always-on host: Codespaces shut down when idle, so use a small VPS or a spare computer

**Gate:** runs unattended for 5 trading days on demo with no crashes, no orphaned orders, and no position the bot doesn't know about.

## Phase 3 - Forward test (Nov 30 - Dec 31)

- [ ] Run continuously on demo
- [ ] Weekly: compare live fills, spreads and results with what the backtest predicted for the same weeks
- [ ] Fix every gap between the two (slippage model, spread at news times, weekend gaps)
- [ ] Optional: an LLM layer for news-risk filtering (stand aside around major releases) and a daily plain-English journal. Not as the order-placing brain - see below.

**Gate (end of year):** the live demo results are consistent with the backtest. Then decide, on the numbers, whether a small live account is justified in 2027.

## About the LLM

An LLM can't be honestly backtested on historical markets: its training data may already contain what happened next, which is a built-in lookahead bias no test can remove. It also gives different answers to the same input. That makes it a poor fit for the decision that places orders, and a good fit for the jobs around it: summarizing news risk, explaining trades in the journal, natural-language control in Telegram.

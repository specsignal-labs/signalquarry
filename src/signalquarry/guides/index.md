# SignalQuarry

SignalQuarry turns a trading-strategy idea into an **honest** backtest on Alpaca
data and then into an Alpaca **paper** forward test. It is built to be driven by
your own coding agent: one CLI (`sqy`), one JSON envelope per command, a reason
code for every outcome, and an `AGENTS.md` in every project.

The framework owns time, data, fills and the evidence record. You (or your
agent) write one pure function, `decide(ctx, params)`, and declare everything
else in `strategy.yaml`. That split is what makes the results trustworthy:

- **No look-ahead by structure.** `decide` sees only completed bars, adjusted
  point-in-time, and no clock, files or network.
- **Every trial counts.** Each configuration evaluated on real data is appended
  to a hash-chained trial ledger; the deflated Sharpe ratio uses the count.
- **A sealed holdout.** `sqy spec freeze` seals the most recent months; they can
  be opened once per strategy family, after the walk-forward gates pass.
- **Claims follow evidence.** Every envelope carries `evidence.claim_level`
  (`none` → `in_sample` → `walk_forward` → `holdout_passed` → `paper_forward`).
- **The same step in backtest and paper.** The paper runner reuses the
  backtest's decide and sizing code; a parity test keeps them identical.

SignalQuarry has **no live-trading path**. Passing tests, backtests or paper
results never authorize live trading.

Start with the [quickstart](quickstart.md).

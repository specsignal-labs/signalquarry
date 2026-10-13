# 0018. Scope decisions before 1.0

Status: accepted (2026-10-13).

## Context

The research workflow added in 0.2 (benchmark-relative results, run comparison, studies,
diagnostics, an offline factor workflow) left a list of things that each needed a ruling rather
than more code. This ADR records the rulings that do not belong to a single earlier ADR.
Settled on 2026-10-13. The owner delegated these decisions to the maintainer of this change; the reasoning is written down so that each one can be revisited.

## Decisions

1. **The engine may report positions per session, on request.** Attribution by symbol and by
   year needs the holdings and marks the engine already computes. Reconstructing them outside
   the engine would duplicate its accounting for splits, dividends, settlement and delistings,
   and two implementations of that will disagree. The equity engine gains an opt-in output of
   per-session positions and marks. It is never part of the ledger hash, it is off by default
   so the benchmark targets are unaffected, and `sqy backtest --positions` writes it as an
   additional run artifact.
2. **Short and market-neutral daily portfolios are the next execution model,** after the
   0.5 line, under their own ADR. A cross-sectional factor cannot be judged as a portfolio
   without them. That ADR must cover borrow cost, dividends owed, locate failure and a leverage
   cap, and must change the backtest planner and the paper kernel together.
3. **Intraday equities and multi-leg options stay non-goals until 1.0.** Each needs a
   different data contract and fill model, and neither is needed to make daily research honest.
4. **The 500-symbol cap is raised only by measurement.** The cap moves when
   `tools/bench.py --full` meets its targets at the new size, not by decision.
5. **A `notebook` extra** (`pandas`, `matplotlib`) is added for `signalquarry.research`. The
   four runtime dependencies of the core do not change.
6. **Slow statistical tests run weekly and on demand,** beside the full benchmark, not on
   every pull request. This covers the 200-panel null test of ADR 0014.
7. **Unchanged, and still the owner's alone:** publishing a release, the live provider smoke
   test, extending a trial budget, opening a holdout and arming paper trading. Nothing in this
   ADR, and no delegation of design decisions, covers them.

## Consequences

The order of work is: local data import (ADR 0016), prefix-verified universe builds and the
factor search they unblock (ADR 0014), position attribution, the notebook extra, then feature
datasets (ADR 0017). Short portfolios follow under a new ADR.

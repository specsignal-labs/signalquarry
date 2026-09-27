# SPDX-License-Identifier: Apache-2.0
"""Engine performance against the documented targets (ARCHITECTURE.md → Performance targets).

    uv run python tools/bench.py [--full]

Default runs the CI set (10 years × 1 and × 100 symbols, a walk-forward evaluation with
stress reruns). ``--full`` adds 10 years × 500 symbols. Each case warns above 1.5× its
target and fails above 2×. Data is synthetic; timings exclude data generation.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from datetime import date
from decimal import Decimal

from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.backtest import run_backtest
from signalquarry._internal.validation.evaluate import evaluate
from signalquarry.sdk import Ctx, Decision, Params, definition_of, strategy, ta


class TrendParams(Params):
    period: int = 50


@strategy(params=TrendParams, lookback=lambda p: p.period)
def trend(ctx: Ctx, p: TrendParams) -> Decision:
    picks = [s for s in ctx.symbols if ctx.bars(s).close[-1] > ta.sma(ctx.bars(s).close, p.period)[-1]]
    weight = (Decimal("0.95") / Decimal(max(len(picks), 1))).quantize(Decimal("0.000001"))
    return Decision.target({s: weight for s in picks}, "GO")


def _spec(symbols: tuple[str, ...]) -> StrategySpecV1:
    return StrategySpecV1.model_validate(
        {
            "schema": "signalquarry.strategy/v1",
            "id": "bench",
            "family": "bench",
            "version": "1",
            "hypothesis": {"statement": "Benchmark workload only.", "falsification": "Not a hypothesis."},
            "data": {"symbols": list(symbols), "feed": "synthetic"},
            "reason_codes": {"GO": "benchmark"},
        }
    )


def _case(symbols: int, *, walk_forward: bool) -> tuple[float, float]:
    names = tuple(f"B{i:03d}" for i in range(symbols))
    data = synthetic_dataset(date(2016, 1, 4), date(2025, 12, 31), symbols=names)
    spec, definition = _spec(names), definition_of(trend)
    started = time.perf_counter()
    if walk_forward:
        result = evaluate(
            spec, definition, TrendParams(), data, holdout_start=None, project_trials=1, sharpe_variance=0.0
        )
        assert len(result.folds) >= 10, len(result.folds)
    else:
        run_backtest(spec, definition, TrendParams(), data)
    elapsed = time.perf_counter() - started
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)
    return elapsed, rss_mb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--full", action="store_true", help="also run 10 years × 500 symbols")
    args = parser.parse_args(argv)
    cases = [
        ("10y x 1 symbol", 1, False, 1.0),
        ("10y x 100 symbols", 100, False, 10.0),
        ("walk-forward (>=10 folds, stress)", 1, True, 15.0),
    ]
    if args.full:
        cases.append(("10y x 500 symbols", 500, False, 60.0))
    failed = False
    for label, symbols, walk_forward, target in cases:
        elapsed, rss = _case(symbols, walk_forward=walk_forward)
        verdict = "ok" if elapsed <= 1.5 * target else "WARN" if elapsed <= 2 * target else "FAIL"
        failed |= verdict == "FAIL"
        print(f"{label:<36} {elapsed:7.2f}s  target {target:5.1f}s  peak {rss:6.0f} MB  {verdict}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

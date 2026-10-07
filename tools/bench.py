# SPDX-License-Identifier: Apache-2.0
"""Engine performance against the documented targets (ARCHITECTURE.md → Performance targets).

    uv run python tools/bench.py [--full]

Default runs the CI set (10 years × 1 and × 100 symbols, a walk-forward evaluation with
stress reruns). ``--full`` adds 10 years × 500 symbols, a 3,000-symbol spooled
engine diagnostic, and a 3,000-symbol research panel. The diagnostic bypasses
the public 500-symbol spec cap only through an in-memory model copy.
Engine cases warn above 1.5× their target and fail above 2×; 3,000-symbol cases
have a strict 3-minute / 2-GB ceiling. Data is synthetic; timings exclude generation.

``--spooled-ceiling SECONDS`` raises only the spooled diagnostic's time ceiling (default 180). The
180-second target is calibrated on Apple Silicon; the scheduled CI workflow passes a looser ceiling
because shared Linux runners are slower and noisier. Memory stays capped at 2 GB either way.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from datetime import date
from decimal import Decimal
from pathlib import Path

from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.data.panel import PanelStore
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.backtest import run_backtest
from signalquarry._internal.engine.run import simulate_equity_ticks
from signalquarry._internal.evidence.run_spool import spool_equity_run
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


def _panel_case() -> tuple[float, float]:
    names = tuple(f"F{i:04d}" for i in range(3000))
    data = synthetic_dataset(date(2016, 1, 4), date(2025, 12, 31), symbols=names)
    with tempfile.TemporaryDirectory(prefix="signalquarry-panel-bench-") as directory:
        store = PanelStore(Path(directory))
        started = time.perf_counter()
        store.build(data)
        loaded = store.load(data.identity())
        for session in data.sessions[1200:1300]:
            window = loaded.window(decision_session=session, lookback=20)
            assert window.present.shape == (20, 3000)
        elapsed = time.perf_counter() - started
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)
    return elapsed, rss_mb


def _spooled_case() -> tuple[float, float, str]:
    names = tuple(f"B{i:03d}" for i in range(3000))
    data = synthetic_dataset(date(2016, 1, 4), date(2025, 12, 31), symbols=names)
    limited = _spec(names[:500])
    diagnostic = limited.model_copy(update={"data": limited.data.model_copy(update={"symbols": names})})
    with tempfile.TemporaryDirectory(prefix="signalquarry-spool-bench-") as directory:
        started = time.perf_counter()
        spool = spool_equity_run(
            simulate_equity_ticks(diagnostic, definition_of(trend), TrendParams(), data), Path(directory)
        )
        elapsed = time.perf_counter() - started
        ledger_hash = spool.result.ledger_hash
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)
    return elapsed, rss_mb, ledger_hash


def _isolated_case(name: str) -> dict[str, float | str]:
    """Measure peak RSS in a fresh process instead of cumulative ru_maxrss."""
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--case", name],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


MEMORY_CEILING_MB = 2000.0
DEFAULT_CEILING_SECONDS = 180.0


def ceiling_verdict(elapsed: float, rss: float, ceiling_seconds: float = DEFAULT_CEILING_SECONDS) -> str:
    """``ok`` strictly below both ceilings, otherwise ``FAIL``."""
    return "ok" if elapsed < ceiling_seconds and rss < MEMORY_CEILING_MB else "FAIL"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--full", action="store_true", help="also run 500-symbol engine and 3,000-symbol diagnostics"
    )
    parser.add_argument(
        "--spooled-ceiling",
        type=float,
        default=DEFAULT_CEILING_SECONDS,
        metavar="SECONDS",
        help="time ceiling for the 3,000-symbol spooled diagnostic (default 180; CI passes a looser one)",
    )
    parser.add_argument("--case", choices=("spooled", "panel"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.case == "spooled":
        elapsed, rss, ledger_hash = _spooled_case()
        print(json.dumps({"elapsed": elapsed, "rss": rss, "ledger_hash": ledger_hash}))
        return 0
    if args.case == "panel":
        elapsed, rss = _panel_case()
        print(json.dumps({"elapsed": elapsed, "rss": rss}))
        return 0
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
    if args.full:
        spooled = _isolated_case("spooled")
        elapsed, rss, ledger_hash = (
            float(spooled["elapsed"]),
            float(spooled["rss"]),
            str(spooled["ledger_hash"]),
        )
        verdict = ceiling_verdict(elapsed, rss, args.spooled_ceiling)
        failed |= verdict == "FAIL"
        print(
            f"{'10y x 3000 spooled diagnostic':<36} {elapsed:7.2f}s  "
            f"target {args.spooled_ceiling:5.1f}s  peak {rss:6.0f} MB  {verdict}"
        )
        print(f"  ledger {ledger_hash}")
        panel = _isolated_case("panel")
        elapsed, rss = float(panel["elapsed"]), float(panel["rss"])
        verdict = ceiling_verdict(elapsed, rss)
        failed |= verdict == "FAIL"
        print(
            f"{'10y x 3000 research panel':<36} {elapsed:7.2f}s  target 180.0s  peak {rss:6.0f} MB  {verdict}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

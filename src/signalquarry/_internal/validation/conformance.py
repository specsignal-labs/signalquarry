# SPDX-License-Identifier: Apache-2.0
"""``sqy check``: structural guarantees a strategy must meet before any backtest counts.

* import policy — authored code may only use approved modules, with an SDK-only
  SignalQuarry boundary for factors; numpy, pydantic and pure
  standard-library modules and its own package; no clocks, randomness, files,
  processes or network;
* determinism — two runs on the same data produce the same ledger;
* look-ahead — perturbing every bar after a cutoff leaves earlier decisions unchanged;
* contract — the run completes (declared reason codes, declared symbols, weight limits).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np

from signalquarry._internal.data.dataset import Dataset
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.backtest import EngineError
from signalquarry._internal.engine.run import is_options, simulate
from signalquarry._internal.project.project import LoadedStrategy

ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "collections",
        "dataclasses",
        "decimal",
        "enum",
        "functools",
        "itertools",
        "math",
        "numpy",
        "operator",
        "pydantic",
        "signalquarry",
        "statistics",
        "typing",
    }
)
FORBIDDEN_CALLS = frozenset({"open", "eval", "exec", "compile", "__import__", "input", "breakpoint"})
FORBIDDEN_ATTRIBUTES = frozenset({"now", "today", "utcnow", "time", "perf_counter", "monotonic"})
CHECK_START, CHECK_END = date(2018, 1, 2), date(2023, 12, 29)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def import_policy(package_dir: Path, own_package: str, *, sdk_only: bool = False) -> CheckResult:
    problems: list[str] = []
    allowed = ALLOWED_IMPORTS | {own_package}
    for path in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imports: list[str] = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports = [
                    f"{node.module}.{alias.name}" if node.module == "signalquarry" else node.module
                    for alias in node.names
                ]
            problems += [
                f"{path.name}:{getattr(node, 'lineno', 0)}: import "
                f"{module if sdk_only else module.split('.')[0]}"
                for module in imports
                if module.split(".")[0] not in allowed
                or (
                    sdk_only
                    and (
                        (module == "signalquarry" or module.startswith("signalquarry."))
                        and module != "signalquarry.sdk"
                        and not module.startswith("signalquarry.sdk.")
                    )
                )
            ]
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in FORBIDDEN_CALLS
            ):
                problems.append(f"{path.name}:{node.lineno}: call {node.func.id}()")
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
                problems.append(f"{path.name}:{node.lineno}: .{node.attr}")
            if isinstance(node, ast.Global):
                problems.append(f"{path.name}:{node.lineno}: global statement")
    return CheckResult("import_policy", not problems, "; ".join(problems[:10]))


def perturb_after(
    dataset: Dataset, cut: int, first: float = 0.6, last: float = 1.7, *, keep_open_at_cut: bool = False
) -> Dataset:
    """Return a copy whose bars from index ``cut`` on are scaled by a ramp from ``first`` to ``last``.

    ``keep_open_at_cut`` leaves the cut session's open untouched (it is legitimately
    known at an options strategy's opening checkpoint).
    """
    series = {}
    for symbol, item in dataset.series.items():
        ramp = np.linspace(first, last, len(dataset.sessions) - cut)
        micro = {name: values.copy() for name, values in item.micro.items()}
        for name, values in micro.items():
            original_open = values[cut] if keep_open_at_cut and name == "open" else None
            values[cut:] = (values[cut:] * ramp).astype(np.int64)
            if original_open is not None:
                values[cut] = original_open
        series[symbol] = replace(item, micro=micro)
    return Dataset(
        dataset.sessions,
        series,
        dataset.splits,
        dataset.dividends,
        source=f"{dataset.source}:perturbed@{cut}",
    )


def run_checks(strategy: LoadedStrategy) -> list[CheckResult]:
    own_package = strategy.module.__name__.split(".")[0]
    results = [import_policy(strategy.package_dir, own_package)]
    data = synthetic_dataset(CHECK_START, CHECK_END, symbols=strategy.spec.data.symbols)
    try:
        first = simulate(strategy.spec, strategy.definition, strategy.params, data)
        second = simulate(strategy.spec, strategy.definition, strategy.params, data)
    except EngineError as exc:
        return [*results, CheckResult("contract", False, str(exc))]
    results.append(
        CheckResult(
            "contract", True, f"{len(first.decisions)} decisions, {len(first.fills)} fills on synthetic data"
        )
    )
    results.append(CheckResult("determinism", first.ledger_hash == second.ledger_hash, first.ledger_hash))
    # The decision for session ``cut`` may only use bars before ``cut``, so decisions through
    # ``cut`` inclusive must survive any change to bars from ``cut`` on. Several cuts and both
    # directions make a one-bar leak (seeing the traded session's bar) flip at least one decision.
    options = is_options(strategy.spec)
    leaks: list[str] = []
    # Options decisions react to a leaked bar only in some wheel states, so they get more cuts.
    fractions = (0.45, 0.6, 0.75) if not options else tuple(0.3 + 0.05 * k for k in range(12))
    for fraction in fractions:
        cut = int(len(data.sessions) * fraction)
        for low, high in ((0.5, 0.8), (2.0, 1.3)):
            try:
                mutated = simulate(
                    strategy.spec,
                    strategy.definition,
                    strategy.params,
                    perturb_after(data, cut, low, high, keep_open_at_cut=options),
                )
            except EngineError as exc:
                return [*results, CheckResult("lookahead", False, f"mutated run failed: {exc}")]
            cut_day = data.sessions[cut].isoformat()

            def known(record: dict, cut_day: str = cut_day) -> bool:
                # Decisions made before any bar from the cut on could be known. For options the
                # cut session's opening checkpoint is included too (its open is kept as it was).
                if record["session"] < cut_day:
                    return True
                return record["session"] == cut_day and (not options or record.get("checkpoint") == "open")

            before = [d for d in first.decisions if known(d)]
            after = [d for d in mutated.decisions if known(d)]
            if before != after:
                changed = next((a["session"] for a, b in zip(before, after, strict=False) if a != b), cut_day)
                leaks.append(
                    f"decision for {changed} changed when bars from {data.sessions[cut].isoformat()} changed"
                )
    results.append(
        CheckResult(
            "lookahead",
            not leaks,
            "decisions never changed when only later bars changed (3 cuts, 2 directions)"
            if not leaks
            else leaks[0],
        )
    )
    wanted = "open" if options else "target"
    active = [d for d in first.decisions if d["action"] == wanted]
    results.append(
        CheckResult(
            "produces_targets",
            bool(active),
            f"{len(active)} {wanted} decisions" if active else f"never returned {wanted} on synthetic data",
        )
    )
    return results

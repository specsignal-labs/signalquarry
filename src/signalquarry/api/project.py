# SPDX-License-Identifier: Apache-2.0
"""Project commands: ``init``, ``check`` and ``backtest``."""

from __future__ import annotations

import importlib
import json
import re
import sys
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from signalquarry._internal.contracts import progress
from signalquarry._internal.engine.backtest import EngineError
from signalquarry._internal.engine.run import is_options, simulate, simulate_equity_ticks
from signalquarry._internal.evidence.run_spool import spool_equity_run
from signalquarry._internal.evidence.runs import (
    csv_chunks,
    jsonl_chunks,
    new_run_id,
    result_document,
    unique_run_id,
    write_run,
)
from signalquarry._internal.project import agents_md
from signalquarry._internal.project.project import (
    LoadedStrategy,
    ProjectError,
    find_root,
    load_config,
    load_strategies,
)
from signalquarry._internal.validation.conformance import import_policy, run_checks
from signalquarry._internal.validation.factor_conformance import run_factor_checks
from signalquarry._internal.validation.metrics import summarize
from signalquarry.api.envelope import Envelope
from signalquarry.sdk.factors import ATTRIBUTE as FACTOR_ATTRIBUTE
from signalquarry.sdk.factors import FactorDef

SYNTHETIC_START, SYNTHETIC_END = date(2014, 1, 2), date(2025, 12, 31)
_RENAMES = {"gitignore.txt": ".gitignore", "gitkeep.txt": ".gitkeep", "github": ".github"}


def _package_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "strategies"
    return slug if slug[0].isalpha() else f"s_{slug}"


def init(
    path: Path, *, demo: bool = False, package: str | None = None, lab: bool = False, kind: str = "equity"
) -> Envelope:
    target = path.resolve()
    if target.exists() and any(target.iterdir()):
        return Envelope(
            command="init",
            status="invalid",
            reason_codes=["PROJECT_DIR_NOT_EMPTY"],
            summary=f"{target} is not empty",
        )
    package_name = package or _package_name(target.name)
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", package_name):
        return Envelope(
            command="init",
            status="usage",
            reason_codes=["USAGE_INVALID"],
            summary=f"invalid package name {package_name!r}",
        )
    values = {
        "project": target.name,
        "package": package_name,
        "provider": "synthetic" if demo else "alpaca",
        "symbol": "SYNA" if demo else "SPY",
        "feed": "synthetic" if demo else "sip",
        "paper_broker": "simulated" if demo else "alpaca-paper",
        "underlying": "QQQ",
    }
    if kind not in ("equity", "options") or (kind == "options" and lab):
        return Envelope(
            command="init",
            status="usage",
            reason_codes=["USAGE_INVALID"],
            summary="--kind is equity or options (not with --lab)",
        )
    created: list[str] = []
    if lab and demo:
        return Envelope(
            command="init",
            status="usage",
            reason_codes=["USAGE_INVALID"],
            summary="--lab projects use real data; try `sqy init --demo` separately",
        )
    root = resources.files("signalquarry") / "templates" / ("lab" if lab else "project")

    def copy(node: Traversable, relative: Path) -> None:
        for child in node.iterdir():
            name = (_RENAMES.get(child.name) or child.name).replace("__package__", package_name)
            if child.name == "__pycache__":
                continue
            if child.is_dir():
                copy(child, relative / name)
                continue
            text = child.read_text(encoding="utf-8")
            for key, value in values.items():
                text = text.replace("{{" + key + "}}", value)
            destination = target / relative / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            created.append((relative / name).as_posix())

    copy(root, Path("."))
    first = "sma-trend"
    if kind == "options":
        # Replace the equity starter with the reference wheel.
        import shutil

        starter = target / "src" / package_name / "sma_trend"
        shutil.rmtree(starter)
        created[:] = [c for c in created if "/sma_trend/" not in c]
        copy(resources.files("signalquarry") / "templates" / "strategies", Path("src") / package_name)
        for relative in ("signalquarry.toml", "paper/demo.paper.yaml", "README.md"):
            file = target / relative
            text = file.read_text(encoding="utf-8")
            text = text.replace(f"{package_name}.sma_trend.strategy", f"{package_name}.wheel.strategy")
            text = text.replace("strategy: sma-trend", "strategy: wheel")
            text = text.replace("--strategy sma-trend", "--strategy wheel")
            file.write_text(text, encoding="utf-8")
        first = "wheel"
    return Envelope(
        command="init",
        summary=f"created {target.name} ({'strategy lab' if lab else 'synthetic demo data' if demo else 'Alpaca data'})",
        data={
            "path": str(target),
            "package": package_name,
            "provider": values["provider"],
            "files": sorted(created),
        },
        next_actions=[
            {"command": f"cd {target}", "why": "Run the remaining commands inside the project."},
            {"command": "sqy check", "why": "Verify the starter strategy meets the contract."},
            {"command": "python tools/new_family.py NAME", "why": "Start the first real strategy family."}
            if lab
            else {"command": f"sqy backtest --strategy {first}", "why": "Run a first backtest."},
        ],
    )


def upgrade_agents_md(*, project: Path | None = None) -> Envelope:
    """Refresh the framework-owned blocks of the project's AGENTS.md from the current template."""
    try:
        root = find_root(project)
        config = load_config(root)
    except ProjectError as exc:
        return Envelope(
            command="init", status="invalid", reason_codes=[exc.code], summary=exc.detail or exc.code
        )
    path = root / "AGENTS.md"
    current = path.read_text(encoding="utf-8") if path.is_file() else ""
    kind = "lab" if "signalquarry:begin lab/" in current else "project"
    template = (resources.files("signalquarry") / "templates" / kind / "AGENTS.md").read_text(
        encoding="utf-8"
    )
    package = config.modules[0].split(".", 1)[0] if config.modules else ""
    template = template.replace("{{package}}", package).replace("{{project}}", config.name)
    result = agents_md.upgrade(current, template)
    if result is None:
        return Envelope(
            command="init",
            status="invalid",
            reason_codes=["AGENTS_MD_UNMANAGED"],
            summary=f"{path} has no signalquarry:begin/end blocks",
        )
    if result.updated or result.added:
        path.write_text(result.text, encoding="utf-8")
    return Envelope(
        command="init",
        summary=f"AGENTS.md: {len(result.updated)} updated, {len(result.added)} added, "
        f"{len(result.unchanged)} unchanged",
        data={
            "path": str(path),
            "template": kind,
            "updated": list(result.updated),
            "added": list(result.added),
            "unchanged": list(result.unchanged),
        },
        next_actions=[
            {"command": "git diff AGENTS.md", "why": "Review the refreshed guidance before committing."}
        ]
        if result.updated or result.added
        else [],
    )


def load_project(project: Path | None, command: str) -> tuple[Path, dict[str, LoadedStrategy]] | Envelope:
    try:
        root = find_root(project)
        return root, load_strategies(load_config(root))
    except ProjectError as exc:
        return Envelope(
            command=command, status="invalid", reason_codes=[exc.code], summary=exc.detail or exc.code
        )


def _code_of(error: Exception) -> str:
    return str(error).split(":", 1)[0]


def check(strategy_id: str | None = None, *, project: Path | None = None, parity: bool = False) -> Envelope:
    loaded = load_project(project, "check")
    if isinstance(loaded, Envelope):
        return loaded
    _, strategies = loaded
    if strategy_id is not None and strategy_id not in strategies:
        return Envelope(
            command="check",
            status="invalid",
            reason_codes=["STRATEGY_NOT_FOUND"],
            summary=strategy_id,
            data={"strategies": sorted(strategies)},
        )
    selected = {strategy_id: strategies[strategy_id]} if strategy_id else strategies
    report: list[dict[str, Any]] = []
    failed: list[str] = []
    for number, (name, strategy) in enumerate(sorted(selected.items()), start=1):
        progress.emit("check", strategy=name, done=number - 1, total=len(selected))
        results = run_checks(strategy)
        if parity and all(item.ok for item in results):
            from signalquarry._internal.paper.parity import parity as parity_check

            results.append(parity_check(strategy))
        ok = all(item.ok for item in results)
        failed += [] if ok else [name]
        report.append(
            {
                "strategy": name,
                "ok": ok,
                "configuration_hash": strategy.configuration_hash,
                "checks": [item.as_dict() for item in results],
            }
        )
    envelope = Envelope(command="check", data={"strategies": report})
    if not selected:
        envelope.status, envelope.reason_codes, envelope.summary = (
            "invalid",
            ["STRATEGY_NOT_FOUND"],
            "no strategies configured",
        )
    elif failed:
        envelope.status, envelope.reason_codes = "blocked", ["CONFORMANCE_FAILED"]
        envelope.summary = f"{len(failed)} of {len(selected)} strategies failed: {', '.join(failed)}"
    else:
        envelope.summary = f"{len(selected)} strategies pass conformance"
        first = sorted(selected)[0]
        envelope.next_actions.append(
            {"command": f"sqy backtest --strategy {first}", "why": "Conformance passed."}
        )
    return envelope


def check_factor(module_name: str, *, project: Path | None = None, params_json: str = "{}") -> Envelope:
    """Check one explicitly named project factor against synthetic panels."""
    try:
        root = find_root(project)
        config = load_config(root)
        for directory in reversed(config.src_dirs):
            location = str(directory)
            if directory.is_dir() and location not in sys.path:
                sys.path.insert(0, location)
        module = importlib.import_module(module_name)
        module_file = Path(module.__file__ or "").resolve()
        if not module_file.is_relative_to(root):
            raise ValueError("factor module must be inside the project")
        # @factor attaches the definition to its function; imported factors do not count.
        found = {
            id(definition): definition
            for value in vars(module).values()
            if (definition := getattr(value, FACTOR_ATTRIBUTE, None)) is not None
            and isinstance(definition, FactorDef)
            and definition.module == module_name
        }
        if len(found) != 1:
            raise ValueError(f"{module_name} defines {len(found)} @factor functions; expected one")
        definition = next(iter(found.values()))
        raw_params = json.loads(params_json)
        if not isinstance(raw_params, dict):
            raise ValueError("factor params must be a JSON object")
        params = definition.params.model_validate(raw_params)
    except Exception as exc:
        code = exc.code if isinstance(exc, ProjectError) else "USAGE_INVALID"
        return Envelope(command="check", status="invalid", reason_codes=[code], summary=str(exc)[:500])
    results = [import_policy(module_file.parent, module_name.split(".")[0], sdk_only=True)]
    if results[0].ok:
        results.extend(run_factor_checks(definition, params))
    ok = all(item.ok for item in results)
    return Envelope(
        command="check",
        status="ok" if ok else "blocked",
        reason_codes=[] if ok else ["CONFORMANCE_FAILED"],
        summary=f"factor {module_name} {'passes' if ok else 'fails'} synthetic conformance",
        data={"factor": module_name, "checks": [item.as_dict() for item in results]},
    )


def backtest(
    strategy_id: str, *, start: date | None = None, end: date | None = None, project: Path | None = None
) -> Envelope:
    from signalquarry._internal.validation.evaluate import equity_returns
    from signalquarry._internal.validation.stats import moments
    from signalquarry.api.evidence import (
        budget_warnings,
        family_seal,
        holdout_state,
        record_run_trial,
        trial_evidence,
    )
    from signalquarry.api.resolve import resolve

    resolved = resolve("backtest", strategy_id, project)
    if isinstance(resolved, Envelope):
        return resolved
    root, strategy = resolved.root, resolved.strategy
    warnings: list[str] = []
    seal = family_seal(root, strategy.spec.family)
    if seal is not None and (end is None or end >= seal):
        end = seal - timedelta(
            days=1
        )  # the sealed holdout is only reachable through `sqy evaluate --holdout`
        warnings.append("HOLDOUT_CLIPPED")
    scratch: TemporaryDirectory[str] | None = None
    try:
        if is_options(strategy.spec):
            result = simulate(
                strategy.spec,
                strategy.definition,
                strategy.params,
                resolved.dataset,
                start=start,
                end=end,
                recorded_chains=resolved.recorded_chains(),
            )
            fill_count = len(result.fills)
            fees = sum((fill.fee for fill in result.fills), Decimal(0))
            fill_rows = (
                [
                    f.session.isoformat(),
                    f.symbol,
                    f.side,
                    f.quantity,
                    f.price,
                    f.fee,
                    f.settle_session.isoformat() if f.settle_session else "",
                ]
                for f in result.fills
            )
            decision_chunks = jsonl_chunks(result.decisions)
        else:
            scratch = TemporaryDirectory(prefix="signalquarry-backtest-")
            spool = spool_equity_run(
                simulate_equity_ticks(
                    strategy.spec,
                    strategy.definition,
                    strategy.params,
                    resolved.dataset,
                    start=start,
                    end=end,
                ),
                Path(scratch.name),
            )
            result = spool.result
            fill_count = spool.fill_count
            fees = spool.fees
            fill_rows = spool.fills_csv_rows()
            decision_chunks = spool.decisions_chunks()
    except EngineError as exc:
        if scratch is not None:
            scratch.cleanup()
        return Envelope(command="backtest", status="invalid", reason_codes=[_code_of(exc)], summary=str(exc))
    returns = equity_returns(result, strategy.spec.account.initial_cash)
    m = moments(returns)
    record_run_trial(
        resolved,
        command="backtest",
        returns=returns,
        moments={"n": m.n, "sharpe": m.sharpe, "skew": m.skew, "kurtosis": m.kurtosis},
        window=(result.sessions[0], result.sessions[-1]),
    )
    metrics = summarize(
        result.sessions, result.equity, strategy.spec.account.initial_cash, fills=fill_count, fees=fees
    )
    run_id = unique_run_id(root, new_run_id(strategy.configuration_hash, datetime.now(UTC)))
    evidence = {
        "grade": resolved.evidence_grade,
        "claim_level": "none" if resolved.grade == "synthetic" else "in_sample",
        "holdout": holdout_state(root, strategy.spec.family),
        "trial": None
        if resolved.grade == "synthetic"
        else trial_evidence(root, strategy.spec.family, strategy.configuration_hash),
    }
    artifacts = write_run(
        root,
        run_id,
        {
            "result.json": result_document(
                run_id=run_id,
                strategy_id=strategy.spec.id,
                strategy_version=strategy.spec.version,
                configuration_hash=strategy.configuration_hash,
                code_tree_hash=strategy.code_tree_hash,
                dataset_id=resolved.dataset_id,
                dataset_identity=result.dataset_identity,
                ledger_hash=result.ledger_hash,
                evidence=evidence,
                metrics=metrics,
                warnings=result.warnings,
            ),
            "equity.csv": csv_chunks(
                ["session", "equity", "settled_cash"],
                [
                    [s.isoformat(), e, c]
                    for s, e, c in zip(result.sessions, result.equity, result.cash, strict=True)
                ],
            ),
            "fills.csv": csv_chunks(
                ["session", "symbol", "side", "quantity", "price", "fee", "settle_session"],
                fill_rows,
            ),
            "decisions.jsonl": decision_chunks,
        },
    )
    if scratch is not None:
        scratch.cleanup()
    return Envelope(
        command="backtest",
        summary=f"{strategy_id}: total return {metrics['total_return']:.2%}, max drawdown {metrics['max_drawdown']:.2%} ({resolved.evidence_grade} data: {resolved.dataset_id})",
        metrics=metrics,
        evidence=evidence,
        artifacts=artifacts,
        warnings=sorted({*warnings, *(warning.split(":", 1)[0] for warning in result.warnings)})
        + (
            []
            if resolved.grade == "synthetic"
            else budget_warnings(root, strategy.spec.evaluation.trial_budget, strategy.spec.family)
        ),
        data={
            "run_id": run_id,
            "ledger_hash": result.ledger_hash,
            "configuration_hash": strategy.configuration_hash,
            "dataset_id": resolved.dataset_id,
            "dataset_identity": result.dataset_identity,
        },
        next_actions=[
            {
                "command": f"sqy spec freeze --strategy {strategy_id}",
                "why": "Freeze the configuration and seal the holdout before evaluating.",
            },
            {
                "command": f"sqy evaluate --strategy {strategy_id}",
                "why": "Walk-forward and stress gates decide what you may claim.",
            },
        ],
    )

# SPDX-License-Identifier: Apache-2.0
"""Project commands: ``init``, ``check`` and ``backtest``."""

from __future__ import annotations

import importlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import date, timedelta
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from signalquarry._internal.contracts import progress
from signalquarry._internal.engine.backtest import EngineError
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


def _benchmark_headline(block: dict[str, Any] | None) -> dict[str, Any]:
    """The few benchmark numbers an agent should see without opening ``result.json``."""
    if block is None:
        return {}
    return {
        "benchmark_total_return": block["metrics"].get("total_return"),
        "benchmark_max_drawdown": block["metrics"].get("max_drawdown"),
        "excess_total_return": block["relative"].get("excess_total_return"),
        "beta": block["relative"].get("beta"),
        "information_ratio": block["relative"].get("information_ratio"),
    }


def _versus(block: dict[str, Any] | None) -> str:
    if block is None or block["metrics"].get("total_return") is None:
        return ""
    return (
        f"; {block['symbol']} buy-and-hold {block['metrics']['total_return']:.2%}, "
        f"max drawdown {block['metrics']['max_drawdown']:.2%}"
    )


_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def _overrides(assignments: Sequence[str]) -> dict[str, Any] | str:
    """``NAME=VALUE`` parameter overrides for one run, or what is wrong with them."""
    values: dict[str, Any] = {}
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        name = name.strip()
        if not separator or not name or not value.strip():
            return f"expected NAME=VALUE, got {assignment!r}"
        if name in values:
            return f"parameter {name} is given more than once"
        values[name] = yaml.safe_load(value.strip())
    return values


def backtest(
    strategy_id: str,
    *,
    start: date | None = None,
    end: date | None = None,
    params: Sequence[str] = (),
    label: str | None = None,
    project: Path | None = None,
) -> Envelope:
    """Backtest a strategy and write its run directory.

    ``params`` (``NAME=VALUE``) overrides parameters for this run only: ``strategy.yaml`` is
    not changed, the run has its own configuration hash, and on real data it is a trial like
    any other, refused beforehand when it would exceed the family's budget. ``label`` is a
    short name stored with the run.
    """
    from signalquarry.api.evidence import budget_shortfall, budget_warnings, family_seal
    from signalquarry.api.resolve import resolve
    from signalquarry.api.runs import execute_run

    overrides = _overrides(params)
    if isinstance(overrides, str):
        return Envelope(command="backtest", status="usage", reason_codes=["USAGE_INVALID"], summary=overrides)
    if label is not None and not _LABEL.match(label):
        return Envelope(
            command="backtest",
            status="usage",
            reason_codes=["USAGE_INVALID"],
            summary="label: up to 64 letters, digits, dots, underscores or hyphens, starting with a letter or digit",
        )
    resolved = resolve("backtest", strategy_id, project)
    if isinstance(resolved, Envelope):
        return resolved
    if overrides:
        base = resolved.strategy
        try:
            changed = base.definition.params.model_validate({**base.params.model_dump(), **overrides})
        except ValidationError as exc:
            error = exc.errors()[0]
            return Envelope(
                command="backtest",
                status="invalid",
                reason_codes=["STRATEGY_PARAMS_INVALID"],
                summary=f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}",
            )
        resolved = replace(resolved, strategy=replace(base, params=changed))
        if resolved.grade != "synthetic":
            shortfall = budget_shortfall(
                resolved.root, resolved.strategy.spec, {resolved.strategy.configuration_hash}
            )
            if shortfall is not None:
                return Envelope(
                    command="backtest",
                    status="blocked",
                    reason_codes=["TRIAL_BUDGET_EXHAUSTED"],
                    summary=shortfall,
                )
    root, strategy = resolved.root, resolved.strategy
    warnings: list[str] = []
    seal = family_seal(root, strategy.spec.family)
    if seal is not None and (end is None or end >= seal):
        end = seal - timedelta(
            days=1
        )  # the sealed holdout is only reachable through `sqy evaluate --holdout`
        warnings.append("HOLDOUT_CLIPPED")
    try:
        run = execute_run(resolved, command="backtest", start=start, end=end, label=label)
    except EngineError as exc:
        return Envelope(command="backtest", status="invalid", reason_codes=[_code_of(exc)], summary=str(exc))
    metrics = run.metrics
    return Envelope(
        command="backtest",
        summary=f"{strategy_id}: total return {metrics['total_return']:.2%}, max drawdown {metrics['max_drawdown']:.2%}"
        + _versus(run.context.get("benchmark"))
        + f" ({resolved.evidence_grade} data: {resolved.dataset_id})",
        metrics={**metrics, **_benchmark_headline(run.context.get("benchmark"))},
        evidence=run.evidence,
        artifacts=run.artifacts,
        warnings=sorted(
            {*warnings, *run.warnings, *(warning.split(":", 1)[0] for warning in run.engine_warnings)}
        )
        + (
            []
            if resolved.grade == "synthetic"
            else budget_warnings(root, strategy.spec.evaluation.trial_budget, strategy.spec.family)
        ),
        data={
            "run_id": run.run_id,
            "ledger_hash": run.ledger_hash,
            "configuration_hash": strategy.configuration_hash,
            "dataset_id": resolved.dataset_id,
            "dataset_identity": run.dataset_identity,
            **({"params_override": overrides} if overrides else {}),
            **({} if label is None else {"label": label}),
        },
        next_actions=[
            *run.next_actions,
            *(
                [
                    {
                        "command": "edit strategy.yaml params, then sqy spec freeze",
                        "why": "This run used overridden parameters; evaluation reads strategy.yaml.",
                    }
                ]
                if overrides
                else [
                    {
                        "command": f"sqy spec freeze --strategy {strategy_id}",
                        "why": "Freeze the configuration and seal the holdout before evaluating.",
                    }
                ]
            ),
            {
                "command": f"sqy evaluate --strategy {strategy_id}",
                "why": "Walk-forward and stress gates decide what you may claim.",
            },
        ],
    )

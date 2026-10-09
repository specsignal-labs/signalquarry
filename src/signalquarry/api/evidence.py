# SPDX-License-Identifier: Apache-2.0
"""Evidence commands: ``spec freeze``, ``evaluate``, ``trials`` and ``holdout status``."""

from __future__ import annotations

import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from signalquarry import __version__
from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.contracts import progress
from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.engine.backtest import EngineError
from signalquarry._internal.engine.run import is_options
from signalquarry._internal.evidence.runs import new_run_id, result_document, unique_run_id, write_run
from signalquarry._internal.project.project import ProjectError, find_root
from signalquarry._internal.validation import ledger
from signalquarry._internal.validation.conformance import run_checks
from signalquarry._internal.validation.evaluate import (
    GATES_VERSION,
    Evaluation,
    add_months,
    claim_level,
    equity_returns,
    evaluate,
)
from signalquarry.api.envelope import Envelope
from signalquarry.api.resolve import Resolved, resolve
from signalquarry.plugins import GateContext, GateOutcome, discover

BUDGET_EXTENSION = 10


def _vcs_dirty(root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def family_seal(root: Path, family: str) -> date | None:
    for entry in ledger.freezes(root, family).entries():
        if entry["family"] == family and entry.get("holdout_start"):
            return date.fromisoformat(entry["holdout_start"])
    return None


def holdout_state(root: Path, family: str) -> str:
    if ledger.holdout_opened(root, family):
        return "opened"
    return "sealed" if family_seal(root, family) else "not_sealed"


def holdout_start_for(spec: StrategySpecV1, last_session: date) -> date | None:
    """First sealed day: the last ``months`` of data, or the day after a declared training
    cutoff when that is earlier. None when the strategy declares neither.

    Options strategies never seal one: there is no holdout gate for options (paper forward
    testing, G5, is their out-of-sample proof), so a seal would only discard scarce data.
    """
    if is_options(spec):
        return None
    holdout = spec.evaluation.holdout
    starts: list[date] = []
    if holdout.months > 0:
        starts.append(add_months(last_session, -holdout.months) + timedelta(days=1))
    if holdout.training_cutoff is not None:
        starts.append(holdout.training_cutoff + timedelta(days=1))
    return min(starts) if starts else None


def trial_budget(root: Path, spec_budget: int, family: str) -> int:
    return spec_budget + BUDGET_EXTENSION * ledger.trial_summary(root, family)["extensions"]


def budget_warnings(root: Path, spec_budget: int, family: str) -> list[str]:
    """``TRIAL_BUDGET_NEARLY_USED`` once a family has used 80% of its trial budget."""
    used = ledger.trial_summary(root, family)["family_count"]
    budget = trial_budget(root, spec_budget, family)
    return [f"TRIAL_BUDGET_NEARLY_USED:{used}/{budget}"] if used < budget and used * 5 >= budget * 4 else []


def trial_evidence(root: Path, family: str, configuration_hash: str) -> dict[str, Any]:
    entries = [e for e in ledger.trials(root, family).entries() if e.get("kind") == "trial"]
    family_configs = list(dict.fromkeys(e["configuration_hash"] for e in entries if e["family"] == family))
    summary = ledger.trial_summary(root, family)
    return {
        "family": family,
        "index": family_configs.index(configuration_hash) + 1
        if configuration_hash in family_configs
        else None,
        "count": len(family_configs),
        "project_count": summary["project_count"],
        "ledger_head": summary["head"],
    }


def record_run_trial(
    resolved: Resolved,
    *,
    command: str,
    returns: np.ndarray,
    moments: dict[str, float],
    window: tuple[date, date],
) -> None:
    if resolved.grade == "synthetic":
        return  # synthetic data is not evidence and is not counted
    strategy = resolved.strategy
    ledger.record_trial(
        resolved.root,
        {
            "kind": "trial",
            "at": datetime.now(UTC),
            "command": command,
            "family": strategy.spec.family,
            "strategy_id": strategy.spec.id,
            "configuration_hash": strategy.configuration_hash,
            "dataset_id": resolved.dataset_id,
            "dataset_identity": resolved.dataset.identity(),
            "window": [window[0].isoformat(), window[1].isoformat()],
            "n_obs": int(moments.get("n", len(returns))),
            "sharpe": None if not np.isfinite(moments.get("sharpe", np.nan)) else float(moments["sharpe"]),
            "skew": None if not np.isfinite(moments.get("skew", np.nan)) else float(moments["skew"]),
            "kurtosis": None
            if not np.isfinite(moments.get("kurtosis", np.nan))
            else float(moments["kurtosis"]),
            "returns_sha256": canonical_hash([round(float(x), 12) for x in returns]),
            "framework_version": __version__,
        },
    )


def spec_freeze(strategy_id: str, *, project: Path | None = None) -> Envelope:
    resolved = resolve("spec freeze", strategy_id, project)
    if isinstance(resolved, Envelope):
        return resolved
    strategy, root = resolved.strategy, resolved.root
    spec = strategy.spec
    holdout_start = family_seal(root, spec.family)
    newly_sealed = False
    if holdout_start is None:
        holdout_start = holdout_start_for(spec, resolved.dataset.sessions[-1])
        newly_sealed = holdout_start is not None
    freeze_body = {
        "configuration_hash": strategy.configuration_hash,
        "hypothesis": spec.hypothesis.model_dump(),
        "evaluation": spec.evaluation.model_dump(mode="json"),
        "gates": GATES_VERSION,
    }
    entry = ledger.append(
        root,
        "freezes",
        spec.family,
        {
            "at": datetime.now(UTC),
            "family": spec.family,
            "strategy_id": spec.id,
            "version": spec.version,
            "configuration_hash": strategy.configuration_hash,
            "freeze_hash": canonical_hash(freeze_body),
            "gates": GATES_VERSION,
            "dataset_id": resolved.dataset_id,
            "holdout_start": holdout_start.isoformat() if holdout_start else None,
            "vcs_dirty": _vcs_dirty(root),
        },
    )
    envelope = Envelope(
        command="spec freeze",
        summary=f"{spec.id} frozen as {entry['freeze_hash'][:19]}…; holdout {'sealed from ' + holdout_start.isoformat() if holdout_start else 'disabled'}",
        data={
            "freeze_hash": entry["freeze_hash"],
            "holdout_start": entry["holdout_start"],
            "newly_sealed": newly_sealed,
            "vcs_dirty": entry["vcs_dirty"],
        },
        evidence={
            "grade": resolved.grade,
            "claim_level": "none",
            "holdout": holdout_state(root, spec.family),
            "trial": trial_evidence(root, spec.family, strategy.configuration_hash),
        },
        next_actions=[
            {
                "command": f"sqy evaluate --strategy {spec.id}",
                "why": "Run walk-forward and stress gates on the pre-holdout data.",
            }
        ],
    )
    if entry["vcs_dirty"]:
        envelope.warnings.append("VCS_DIRTY")
    return envelope


def evaluate_command(
    strategy_id: str, *, stress: bool = True, open_holdout: bool = False, project: Path | None = None
) -> Envelope:
    resolved = resolve("evaluate", strategy_id, project)
    if isinstance(resolved, Envelope):
        return resolved
    root, strategy = resolved.root, resolved.strategy
    spec = strategy.spec
    envelope = Envelope(command="evaluate")
    freeze = ledger.latest_freeze(root, spec.id)
    frozen = freeze is not None and freeze["configuration_hash"] == strategy.configuration_hash
    if freeze is not None and not frozen:
        envelope.warnings.append("FREEZE_STALE")
    holdout_start = family_seal(root, spec.family)
    if open_holdout:
        if not frozen:
            envelope.status, envelope.reason_codes = "blocked", ["FREEZE_REQUIRED"]
            envelope.summary = "the holdout can only be opened for a frozen configuration"
            return envelope
        if ledger.holdout_opened(root, spec.family):
            envelope.status, envelope.reason_codes = "blocked", ["HOLDOUT_REUSED"]
            envelope.summary = f"family {spec.family} already used its holdout"
            return envelope
    summary = ledger.trial_summary(root, spec.family)
    already = any(
        e.get("kind") == "trial" and e["configuration_hash"] == strategy.configuration_hash
        for e in ledger.trials(root, spec.family).entries()
    )
    budget = trial_budget(root, spec.evaluation.trial_budget, spec.family)
    if resolved.grade != "synthetic" and not already and summary["family_count"] >= budget:
        envelope.status, envelope.reason_codes = "blocked", ["TRIAL_BUDGET_EXHAUSTED"]
        envelope.summary = f"family {spec.family} used {summary['family_count']} of {budget} trials"
        return envelope
    checks = run_checks(strategy)
    conformance_ok = all(item.ok for item in checks)
    project_trials = summary["project_count"] + (0 if already or resolved.grade == "synthetic" else 1)
    variance = float(np.var(summary["sharpes"], ddof=1)) if len(summary["sharpes"]) >= 2 else 0.0
    progress.emit("evaluate", step="walk_forward_and_stress", strategy=spec.id)
    chains = resolved.recorded_chains()
    benchmark = resolved.benchmark_source()
    if spec.benchmark is not None and benchmark is None:
        envelope.warnings.append("BENCHMARK_DATA_MISSING")
    try:
        result = evaluate(
            spec,
            strategy.definition,
            strategy.params,
            resolved.dataset,
            holdout_start=holdout_start,
            project_trials=project_trials,
            sharpe_variance=variance,
            stress=stress,
            open_holdout=False,
            recorded_chains=chains,
            benchmark=benchmark,
        )
        pre_gates_ok = result.passed("G1_sample", "G2_walk_forward", "G3_stress")
        if open_holdout and not pre_gates_ok:
            envelope.status, envelope.reason_codes = "blocked", ["HOLDOUT_REQUIRES_GATES"]
            envelope.summary = "G1–G3 must pass before the holdout may be opened"
            envelope.data = {"gates": result.gates, "folds": result.folds, "oos": result.oos}
            return envelope
        if open_holdout:
            progress.emit("evaluate", step="holdout", strategy=spec.id)
            result = evaluate(
                spec,
                strategy.definition,
                strategy.params,
                resolved.dataset,
                holdout_start=holdout_start,
                project_trials=project_trials,
                sharpe_variance=variance,
                stress=stress,
                open_holdout=True,
                recorded_chains=chains,
                benchmark=benchmark,
            )
    except EngineError as exc:
        envelope.status, envelope.reason_codes, envelope.summary = (
            "invalid",
            [str(exc).split(":", 1)[0]],
            str(exc),
        )
        return envelope
    base = result.base
    assert base is not None
    returns = equity_returns(base, spec.account.initial_cash)
    record_run_trial(
        resolved,
        command="evaluate",
        returns=returns,
        moments=result.trial_moments,
        window=(base.sessions[0], base.sessions[-1]),
    )
    if open_holdout and resolved.grade != "synthetic":
        ledger.append(
            root,
            "holdouts",
            spec.family,
            {
                "at": datetime.now(UTC),
                "family": spec.family,
                "strategy_id": spec.id,
                "configuration_hash": strategy.configuration_hash,
                "freeze_hash": freeze["freeze_hash"] if freeze else None,
                "result": result.gates.get("G4_holdout"),
            },
        )
    plugin_warnings = _run_plugin_gates(result, spec, resolved.evidence_grade)
    envelope.warnings += plugin_warnings
    claim = claim_level(result, grade=resolved.evidence_grade, frozen=frozen, conformance_ok=conformance_ok)
    failed = [name for name, gate in result.gates.items() if not gate.get("ok")]
    envelope.data = {
        "gates": result.gates,
        "folds": result.folds,
        "oos": result.oos,
        "conformance": [c.as_dict() for c in checks],
        "dataset_id": resolved.dataset_id,
    }
    envelope.metrics = {
        key: result.oos[key]
        for key in ("total_return", "sharpe_annual", "max_drawdown", "psr", "dsr")
        if key in result.oos
    }
    envelope.evidence = {
        "grade": resolved.evidence_grade,
        "claim_level": claim,
        "holdout": holdout_state(root, spec.family),
        "trial": trial_evidence(root, spec.family, strategy.configuration_hash),
    }
    if resolved.grade != "synthetic":
        envelope.warnings += budget_warnings(root, spec.evaluation.trial_budget, spec.family)
    if not frozen:
        envelope.warnings.append("FREEZE_REQUIRED")
        envelope.next_actions.append(
            {
                "command": f"sqy spec freeze --strategy {spec.id}",
                "why": "Claims above in_sample need a frozen configuration.",
            }
        )
    if not conformance_ok:
        envelope.status, envelope.reason_codes = "blocked", ["CONFORMANCE_FAILED"]
    elif failed:
        envelope.status, envelope.reason_codes = "blocked", ["GATE_FAILED"]
        envelope.next_actions += [
            {
                "command": f"sqy report --strategy {spec.id}",
                "why": "Record the result as it is; a failed gate is a finding, not an error.",
            },
            {
                "command": "sqy explain GATE_FAILED",
                "why": "Next is a new hypothesis, never looser gates or re-tuning on the same data.",
            },
        ]
    envelope.summary = f"{spec.id}: claim_level={claim}" + (f"; failed {', '.join(failed)}" if failed else "")
    run_id = unique_run_id(root, new_run_id(strategy.configuration_hash, datetime.now(UTC)) + "-evaluate")
    envelope.artifacts = write_run(
        root,
        run_id,
        {
            "evaluation.json": result_document(
                kind="evaluation",
                run_id=run_id,
                strategy_id=spec.id,
                strategy_version=spec.version,
                configuration_hash=strategy.configuration_hash,
                dataset_id=resolved.dataset_id,
                frozen=frozen,
                evidence=envelope.evidence,
                gates=result.gates,
                folds=result.folds,
                oos=result.oos,
                status=envelope.status,
            )
        },
    )
    envelope.data["run_id"] = run_id
    return envelope


def trials_ls(*, family: str | None = None, project: Path | None = None) -> Envelope:
    try:
        root = find_root(project)
        entries = ledger.all_entries(root, "trials")
    except (ProjectError, ledger.LedgerError) as exc:
        return Envelope(command="trials ls", status="invalid", reason_codes=[exc.code], summary=str(exc))
    rows = [e for e in entries if family is None or e["family"] == family]
    summary = ledger.trial_summary(root, family)
    return Envelope(
        command="trials ls",
        summary=f"{summary['family_count']} configurations ({summary['project_count']} project-wide)",
        data={"entries": rows, "summary": {k: v for k, v in summary.items() if k != "sharpes"}},
    )


def trials_extend(family: str, reason: str, *, project: Path | None = None) -> Envelope:
    if len(reason.strip()) < 15:
        return Envelope(
            command="trials extend",
            status="usage",
            reason_codes=["USAGE_INVALID"],
            summary="give a --reason of at least 15 characters",
        )
    try:
        root = find_root(project)
        entry = ledger.append(
            root,
            "trials",
            family,
            {
                "kind": "budget_extension",
                "at": datetime.now(UTC),
                "family": family,
                "added": BUDGET_EXTENSION,
                "reason": reason.strip(),
            },
        )
    except (ProjectError, ledger.LedgerError) as exc:
        return Envelope(command="trials extend", status="invalid", reason_codes=[exc.code], summary=str(exc))
    return Envelope(
        command="trials extend",
        summary=f"family {family} budget +{BUDGET_EXTENSION}",
        data={"entry": entry},
        warnings=["HUMAN_ACTION_RECORDED"],
    )


def holdout_status(*, project: Path | None = None) -> Envelope:
    try:
        root = find_root(project)
        freezes = ledger.all_entries(root, "freezes")
        openings = ledger.all_entries(root, "holdouts")
    except (ProjectError, ledger.LedgerError) as exc:
        return Envelope(command="holdout status", status="invalid", reason_codes=[exc.code], summary=str(exc))
    families = sorted({e["family"] for e in freezes})
    rows = [
        {
            "family": f,
            "holdout_start": (family_seal(root, f) or "").__str__() or None,
            "state": holdout_state(root, f),
        }
        for f in families
    ]
    return Envelope(
        command="holdout status",
        summary=f"{len(rows)} families",
        data={"families": rows, "openings": openings},
    )


def evidence_verify(*, base: str | None = None, project: Path | None = None) -> Envelope:
    """Verify every hash-chained log (evidence ledgers and paper journals) and, with ``base``
    (a git ref), that each log only grew by appending since that ref."""
    envelope = Envelope(command="evidence verify")
    try:
        root = find_root(project)
    except ProjectError as exc:
        envelope.status, envelope.reason_codes, envelope.summary = (
            "invalid",
            [exc.code],
            exc.detail or exc.code,
        )
        return envelope
    try:
        family_logs = {
            log.path for kind in ledger.KINDS for log in ledger.logs(root, kind) if log.path.is_file()
        }
    except ledger.LedgerError as exc:
        envelope.status, envelope.reason_codes, envelope.summary = "invalid", [exc.code], str(exc)
        return envelope
    logs = sorted({*root.glob("evidence/*.jsonl"), *family_logs, *root.glob("paper/*/journal.jsonl")})
    results: list[dict[str, Any]] = []
    for path in logs:
        relative = path.relative_to(root).as_posix()
        row: dict[str, Any] = {"path": relative, "ok": True}
        try:
            row["entries"] = len(ledger.ChainedLog(path, "any").entries())
        except ledger.LedgerError as exc:
            row.update(ok=False, code=exc.code)
        if base is not None and row["ok"]:
            shown = subprocess.run(
                ["git", "show", f"{base}:{relative}"], cwd=root, capture_output=True, check=False
            )
            if shown.returncode == 0 and not path.read_bytes().startswith(shown.stdout):
                row.update(ok=False, code="EVIDENCE_LOG_REWRITTEN")
        results.append(row)
    if base is not None:
        listed = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", base, "--", "evidence", "paper", "families"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        present = {row["path"] for row in results}
        for name in listed.stdout.split():
            if name.endswith(".jsonl") and name not in present:
                results.append({"path": name, "ok": False, "code": "EVIDENCE_LOG_REWRITTEN"})
    try:
        index_problems = ledger.verify_index(root)
    except ledger.LedgerError as exc:
        index_problems = [exc.code]
    results += [
        {
            "path": "evidence/project_index.jsonl",
            "ok": False,
            "code": problem.split(":")[0],
            "detail": problem,
        }
        for problem in index_problems
    ]
    failed = sorted({row["code"] for row in results if not row["ok"]})
    envelope.data = {"logs": results, "base": base}
    if failed:
        envelope.status, envelope.reason_codes = "blocked", failed
    envelope.summary = f"{sum(r['ok'] for r in results)} of {len(results)} logs verified" + (
        f" append-only since {base}" if base else ""
    )
    return envelope


def holdout_seal(strategy_id: str, *, project: Path | None = None) -> Envelope:
    """Seal the strategy family's holdout now, before any exploration (a freeze also seals)."""
    resolved = resolve("holdout seal", strategy_id, project)
    if isinstance(resolved, Envelope):
        return resolved
    root, spec = resolved.root, resolved.strategy.spec
    existing = family_seal(root, spec.family)
    if existing is not None:
        return Envelope(
            command="holdout seal",
            summary=f"family {spec.family} already sealed from {existing.isoformat()}",
            data={"family": spec.family, "holdout_start": existing.isoformat(), "newly_sealed": False},
        )
    start = holdout_start_for(spec, resolved.dataset.sessions[-1])
    if start is None:
        return Envelope(
            command="holdout seal",
            status="invalid",
            reason_codes=["STRATEGY_SPEC_INVALID"],
            summary="options strategies have no holdout (paper forward, G5, is their out-of-sample test)"
            if is_options(spec)
            else "evaluation.holdout.months is 0 and no training_cutoff: this strategy declares no holdout",
        )
    entry = ledger.append(
        root,
        "freezes",
        spec.family,
        {
            "kind": "seal",
            "at": datetime.now(UTC),
            "family": spec.family,
            "strategy_id": spec.id,
            "dataset_id": resolved.dataset_id,
            "holdout_start": start.isoformat(),
        },
    )
    return Envelope(
        command="holdout seal",
        summary=f"family {spec.family} sealed from {start.isoformat()}; every command now stops before it",
        data={
            "family": spec.family,
            "holdout_start": start.isoformat(),
            "newly_sealed": True,
            "entry": entry["hash"],
        },
    )


def trials_show(key: str, *, project: Path | None = None) -> Envelope:
    """Show ledger entries by sequence number or configuration-hash prefix."""
    try:
        root = find_root(project)
        entries = ledger.all_entries(root, "trials")
    except (ProjectError, ledger.LedgerError) as exc:
        return Envelope(command="trials show", status="invalid", reason_codes=[exc.code], summary=str(exc))
    needle = key.removeprefix("sha256:")
    matches = [
        e
        for e in entries
        if str(e["seq"]) == key
        or str(e.get("configuration_hash", "")).removeprefix("sha256:").startswith(needle)
    ]
    if not matches or len(needle) < 1:
        return Envelope(
            command="trials show", status="invalid", reason_codes=["TRIAL_NOT_FOUND"], summary=key
        )
    return Envelope(
        command="trials show",
        summary=f"{len(matches)} matching entr{'y' if len(matches) == 1 else 'ies'}",
        data={"entries": matches},
    )


def _run_plugin_gates(result: Evaluation, spec: StrategySpecV1, grade: str) -> list[str]:
    """Run installed gate plugins after the built-in gates (add-only; failures cap the claim)."""
    discovery = discover("gates")
    warnings = [f"PLUGIN_ERROR:{error}" for error in discovery.errors]
    context = GateContext(
        strategy_id=spec.id,
        family=spec.family,
        grade=grade,
        spec=spec.model_dump(mode="json"),
        gates={name: dict(gate) for name, gate in result.gates.items()},
        folds=tuple(dict(fold) for fold in result.folds),
        oos=dict(result.oos),
        oos_returns=result.oos_returns,
    )
    for loaded in discovery.plugins:
        try:
            outcome = loaded.plugin.check(context)
            if not isinstance(outcome, GateOutcome):
                raise TypeError(f"check returned {type(outcome).__name__}, not GateOutcome")
            gate: dict[str, Any] = {"ok": bool(outcome.ok), **dict(outcome.detail), "source": loaded.source}
        except Exception as exc:  # noqa: BLE001 - a broken gate fails closed
            gate = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "source": loaded.source}
            warnings.append(f"PLUGIN_GATE_ERROR:{loaded.name}")
        result.gates[f"plugin:{loaded.name}"] = gate
    return warnings

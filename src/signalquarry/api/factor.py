# SPDX-License-Identifier: Apache-2.0
"""Project factor discovery and descriptive, unverified factor diagnostics."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

from signalquarry._internal.contracts.reason_codes import REASON_CODES
from signalquarry._internal.contracts.spec import HoldoutSpec
from signalquarry._internal.data.library import LibraryError, dataset_from_manifest
from signalquarry._internal.data.panel import PanelStore
from signalquarry._internal.data.universe_build import verify_universe_manifest
from signalquarry._internal.factors.evaluate import UniverseAt, rank_ic, score_factor
from signalquarry._internal.factors.labels import derive_forward_return_labels
from signalquarry._internal.project.factors import LoadedFactor, load_factors
from signalquarry._internal.project.project import ProjectError, find_root, load_config
from signalquarry._internal.validation import ledger
from signalquarry._internal.validation.conformance import import_policy
from signalquarry._internal.validation.factor_conformance import run_factor_checks
from signalquarry._internal.validation.factor_holdouts import (
    effective_factor_declaration,
    factor_evidence_write,
    factor_family_seal,
    factor_holdout_start_for,
    validate_factor_family,
)
from signalquarry._internal.validation.factor_trials import factor_trial_accounting
from signalquarry.api.data import library_for
from signalquarry.api.envelope import Envelope
from signalquarry.api.universe import replay_universe_build


def _load(project: Path | None, command: str) -> tuple[dict[str, LoadedFactor], Path] | Envelope:
    try:
        root = find_root(project)
        return load_factors(load_config(root)), root
    except ProjectError as exc:
        return Envelope(command=command, status="invalid", reason_codes=[exc.code], summary=str(exc)[:500])


def factor_ls(*, project: Path | None = None) -> Envelope:
    loaded = _load(project, "factor ls")
    if isinstance(loaded, Envelope):
        return loaded
    factors, _ = loaded
    return Envelope(
        command="factor ls",
        summary=f"{len(factors)} registered factors",
        data={
            "factors": [
                {
                    "id": item.spec.id,
                    "family": item.spec.family,
                    "version": item.spec.version,
                    "module": item.module.__name__,
                    "configuration_hash": item.configuration_hash,
                    "code_tree_hash": item.code_tree_hash,
                }
                for item in sorted(factors.values(), key=lambda item: item.spec.id)
            ]
        },
    )


def check_registered_factor(factor_id: str, *, project: Path | None = None) -> Envelope:
    loaded = _load(project, "check")
    if isinstance(loaded, Envelope):
        return loaded
    factors, _ = loaded
    if factor_id not in factors:
        return Envelope(
            command="check",
            status="invalid",
            reason_codes=["FACTOR_NOT_FOUND"],
            summary=factor_id,
            data={"factors": sorted(factors)},
        )
    item = factors[factor_id]
    results = [import_policy(item.code_dir, item.module.__name__.split(".")[0], sdk_only=True)]
    if results[0].ok:
        results.extend(run_factor_checks(item.definition, item.params))
    ok = all(result.ok for result in results)
    return Envelope(
        command="check",
        status="ok" if ok else "blocked",
        reason_codes=[] if ok else ["CONFORMANCE_FAILED"],
        summary=f"factor {factor_id} {'passes' if ok else 'fails'} synthetic conformance",
        data={
            "factor": factor_id,
            "family": item.spec.family,
            "configuration_hash": item.configuration_hash,
            "checks": [result.as_dict() for result in results],
        },
    )


def _factor_holdout_error(command: str, exc: Exception) -> Envelope:
    code = getattr(exc, "code", None) or "DATA_MANIFEST_INVALID"
    if code not in REASON_CODES:
        code = "DATA_MANIFEST_INVALID"
    return Envelope(
        command=command,
        status="busy" if code == "FACTOR_EVIDENCE_BUSY" else "invalid",
        reason_codes=[code],
        summary=str(exc)[:500],
    )


def factor_holdout_seal(family: str, dataset_id: str, *, project: Path | None = None) -> Envelope:
    """Explicit human command: seal a factor family's holdout once, without opening it.

    This command is not an MCP tool. It fixes the conservative registered
    declarations and dataset boundary; later declarations never change the seal.
    """
    command = "factor holdout seal"
    try:
        root = find_root(project)
        validate_factor_family(family)
        with factor_evidence_write(root):
            existing = factor_family_seal(root, family)
            factors = load_factors(load_config(root))
            contributors = [
                (item.spec, item.configuration_hash)
                for item in factors.values()
                if item.spec.family == family
            ]
            declaration = effective_factor_declaration(contributors) if contributors else None
            if existing is not None:
                current = {**(declaration or {}), "dataset_id": dataset_id}
                differences = {
                    key: {"sealed": existing[key], "current": current.get(key)}
                    for key in ("trial_budget", "holdout", "factors", "dataset_id")
                    if existing[key] != current.get(key)
                }
                return Envelope(
                    command=command,
                    summary=f"family {family} already sealed from {existing['holdout_start']}",
                    data={
                        "family": family,
                        "holdout_start": existing["holdout_start"],
                        "newly_sealed": False,
                        "seal": existing,
                        "differences": differences,
                    },
                    warnings=["FACTOR_HOLDOUT_DECLARATION_CHANGED"] if differences else [],
                )
            if declaration is None:
                raise ledger.LedgerError("FACTOR_FAMILY_EMPTY", family)
            holdout = HoldoutSpec.model_validate(declaration["holdout"])
            if holdout.months == 0 and holdout.training_cutoff is None:
                raise ledger.LedgerError("FACTOR_HOLDOUT_UNDECLARED", family)
            library = library_for(root)
            manifests: list[dict[str, Any]] = []
            for value in cast(list[object], library.manifests()):
                if not isinstance(value, dict):
                    raise LibraryError("DATA_MANIFEST_INVALID", "manifest must be an object")
                manifests.append(cast(dict[str, Any], value))
            selected = [entry for entry in manifests if entry.get("dataset_id") == dataset_id]
            if len(selected) != 1:
                raise LibraryError("DATA_MANIFEST_INVALID", f"expected one manifest for dataset {dataset_id}")
            dataset = dataset_from_manifest(library, selected[0])
            last_session = dataset.sessions[-1]
            start = factor_holdout_start_for(holdout, last_session)
            entry = ledger.append(
                root,
                "factor_holdouts",
                family,
                {
                    "kind": "seal",
                    "at": datetime.now(UTC),
                    "family": family,
                    "holdout_start": start.isoformat() if start is not None else None,
                    "dataset_id": dataset_id,
                    "dataset_identity": dataset.identity(),
                    "dataset_last_session": last_session.isoformat(),
                    **declaration,
                },
            )
    except (
        ProjectError,
        ledger.LedgerError,
        LibraryError,
        OSError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
    ) as exc:
        return _factor_holdout_error(command, exc)
    return Envelope(
        command=command,
        summary=f"family {family} sealed from {entry['holdout_start']}; holdout has not been opened",
        data={"family": family, "holdout_start": entry["holdout_start"], "newly_sealed": True, "seal": entry},
        warnings=["HUMAN_ACTION_RECORDED"],
    )


def factor_holdout_status(*, family: str | None = None, project: Path | None = None) -> Envelope:
    """Read factor family seals and ledger-derived trial usage; never open a holdout."""
    command = "factor holdout status"
    try:
        root = find_root(project)
        if family is not None:
            validate_factor_family(family)
        factors = load_factors(load_config(root))
        seals = ledger.all_entries(root, "factor_holdouts")
        families = sorted(
            {item.spec.family for item in factors.values()} | {entry["family"] for entry in seals}
        )
        if family is not None:
            if family not in families:
                raise ledger.LedgerError("FACTOR_FAMILY_EMPTY", family)
            families = [family]
        rows: list[dict[str, Any]] = []
        for name in families:
            seal = factor_family_seal(root, name)
            accounting = factor_trial_accounting(root, name) if seal is not None else None
            rows.append(
                {
                    "family": name,
                    "seal": seal,
                    "holdout_start": seal["holdout_start"] if seal is not None else None,
                    "trial_budget": accounting.effective_budget if accounting is not None else None,
                    "holdout": seal["holdout"] if seal is not None else None,
                    "dataset_id": seal["dataset_id"] if seal is not None else None,
                    "trials_used": accounting.family_trials_used
                    if accounting is not None
                    else sum(
                        entry.get("kind") == "factor_trial" and entry.get("family") == name
                        for entry in ledger.all_entries(root, "factor_trials")
                    ),
                    "remaining_budget": accounting.remaining_budget if accounting is not None else None,
                    "state": "sealed" if seal is not None else "not_sealed",
                    "opened": False,
                }
            )
    except (ProjectError, ledger.LedgerError, OSError, TypeError, ValueError, KeyError) as exc:
        return _factor_holdout_error(command, exc)
    return Envelope(
        command=command,
        summary=f"{len(rows)} families; holdouts have not been opened",
        data={"families": rows},
    )


def _load_evaluation_membership(
    root: Path,
    paths: Sequence[Path],
    dataset_manifest: dict[str, Any],
) -> tuple[UniverseAt, ...]:
    """Load only explicitly selected, hash-valid and replayable universe builds."""
    if not paths:
        raise LibraryError("UNIVERSE_INPUT_UNAVAILABLE", "select at least one universe manifest")
    build_dir = (root / "data" / "universe" / "builds").resolve()
    memberships: list[UniverseAt] = []
    seen_sessions: set[date] = set()
    for requested in paths:
        path = requested if requested.is_absolute() else root / requested
        path = path.resolve(strict=True)
        if not path.is_relative_to(build_dir):
            raise LibraryError(
                "DATA_MANIFEST_INVALID", "universe manifest must be under data/universe/builds"
            )
        raw_value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_value, dict):
            raise LibraryError("DATA_MANIFEST_INVALID", "universe manifest must be an object")
        raw = cast(dict[str, Any], raw_value)
        verify_universe_manifest(raw)
        replay_universe_build(root, raw)
        if (
            raw.get("dataset_id") != dataset_manifest.get("dataset_id")
            or raw.get("dataset_identity") != dataset_manifest.get("dataset_identity")
            or raw.get("dataset_manifest_hash") != dataset_manifest.get("manifest_hash")
        ):
            raise LibraryError("DATA_MANIFEST_INVALID", "universe manifest does not match selected dataset")
        try:
            session_value = raw["decision_session"]
            known_at_value = raw["known_at"]
            if not isinstance(session_value, str) or not isinstance(known_at_value, str):
                raise ValueError("manifest timestamps must be strings")
            session = date.fromisoformat(session_value)
            observed_at = datetime.fromisoformat(known_at_value.replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError) as exc:
            raise LibraryError("DATA_MANIFEST_INVALID", "universe manifest session or cutoff") from exc
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise LibraryError("DATA_MANIFEST_INVALID", "universe cutoff must be timezone-aware")
        if session in seen_sessions:
            raise LibraryError("FACTOR_EVALUATION_SESSIONS_INVALID", "duplicate universe decision session")
        seen_sessions.add(session)
        members = raw.get("member_symbols")
        identity = raw.get("manifest_hash")
        if not isinstance(members, list):
            raise LibraryError("DATA_MANIFEST_INVALID", "universe members")
        member_values = cast(list[object], members)
        if not all(isinstance(symbol, str) for symbol in member_values):
            raise LibraryError("DATA_MANIFEST_INVALID", "universe members")
        member_symbols = cast(list[str], member_values)
        if not isinstance(identity, str):
            raise LibraryError("DATA_MANIFEST_INVALID", "universe manifest identity")
        memberships.append(
            UniverseAt(
                session=session,
                observed_at=observed_at,
                decision_cutoff=observed_at,
                symbols=tuple(member_symbols),
                identity=identity,
            )
        )
    return tuple(sorted(memberships, key=lambda item: item.session))


def factor_evaluate(
    factor_id: str,
    dataset_id: str,
    *,
    universe_manifests: Sequence[Path],
    project: Path | None = None,
) -> Envelope:
    """Return descriptive factor metrics with no trial, grade, or holdout authority."""
    envelope = Envelope(command="factor evaluate")
    loaded = _load(project, envelope.command)
    if isinstance(loaded, Envelope):
        return loaded
    factors, root = loaded
    item = factors.get(factor_id)
    if item is None:
        return Envelope(
            command=envelope.command,
            status="invalid",
            reason_codes=["FACTOR_NOT_FOUND"],
            summary=factor_id,
            data={"factors": sorted(factors)},
        )
    checks = [import_policy(item.code_dir, item.module.__name__.split(".")[0], sdk_only=True)]
    if checks[0].ok:
        checks.extend(run_factor_checks(item.definition, item.params))
    if not all(check.ok for check in checks):
        return Envelope(
            command=envelope.command,
            status="blocked",
            reason_codes=["CONFORMANCE_FAILED"],
            summary=f"factor {factor_id} must pass synthetic conformance before evaluation",
            data={"factor": factor_id, "checks": [check.as_dict() for check in checks]},
        )
    try:
        library = library_for(root)
        manifests = [entry for entry in library.manifests() if entry.get("dataset_id") == dataset_id]
        if len(manifests) != 1:
            raise LibraryError("DATA_MANIFEST_INVALID", f"expected one manifest for dataset {dataset_id}")
        dataset_manifest = manifests[0]
        dataset = dataset_from_manifest(library, dataset_manifest)
        memberships = _load_evaluation_membership(root, universe_manifests, dataset_manifest)
        panel_store = PanelStore(library.cache_dir)
        panel_store.build(dataset)
        selected_symbols = tuple(
            sorted({symbol for membership in memberships for symbol in membership.symbols})
        )
        panel = panel_store.load(dataset.identity(), symbols=selected_symbols)
        sessions = tuple(member.session for member in memberships)
        labels = derive_forward_return_labels(
            dataset,
            decision_sessions=sessions,
            symbols=panel.symbols,
            horizons=item.spec.evaluation.horizons,
        )
        scores = score_factor(item.definition, item.params, panel, memberships)
        report = rank_ic(
            scores,
            labels,
            blocks=item.spec.evaluation.chronological_blocks,
            cost_bps=float(item.spec.evaluation.cost_bps),
            capital=float(item.spec.evaluation.capital),
        )
    except (LibraryError, OSError, TypeError, ValueError, KeyError) as exc:
        code = getattr(exc, "code", None) or str(exc).split(":", 1)[0]
        if code not in REASON_CODES:
            code = "DATA_MANIFEST_INVALID"
        status = "unavailable" if code in {"UNIVERSE_INPUT_UNAVAILABLE", "DATA_PAGE_MISSING"} else "invalid"
        return Envelope(
            command=envelope.command,
            status=status,
            reason_codes=[code],
            summary=str(exc)[:500],
        )

    return Envelope(
        command=envelope.command,
        summary=(
            f"unverified descriptive diagnostics for {factor_id}; no trial recorded, "
            "evidence grade assigned, or holdout accessed"
        ),
        data={
            "factor": factor_id,
            "family": item.spec.family,
            "configuration_hash": item.configuration_hash,
            "code_tree_hash": item.code_tree_hash,
            "dataset_id": dataset_id,
            "dataset_manifest_hash": dataset_manifest["manifest_hash"],
            "dataset_identity": report.dataset_identity,
            "universe_identity": report.universe_identity,
            "universe_manifests": [member.identity for member in memberships],
            "decision_sessions": [session.isoformat() for session in scores.sessions],
            "label_identity": report.label_identity,
            "scope": report.scope,
            "authority": {
                "provenance_verified": False,
                "trial_recorded": False,
                "evidence_grade": None,
                "holdout_accessed": False,
            },
            "evaluation": {
                "horizons": list(item.spec.evaluation.horizons),
                "chronological_blocks": item.spec.evaluation.chronological_blocks,
                "cost_bps": str(item.spec.evaluation.cost_bps),
                "capital": str(item.spec.evaluation.capital),
            },
            "conformance_checks": [check.as_dict() for check in checks],
            "horizons": [
                {
                    "horizon": horizon.horizon,
                    "observations": horizon.observations,
                    "scored_pairs": horizon.scored_pairs,
                    "eligible_pairs": horizon.eligible_pairs,
                    "mean_ic": horizon.mean_ic,
                    "icir": horizon.icir,
                    "chronological_blocks": list(horizon.chronological_blocks),
                    "daily_ic": list(horizon.daily_ic),
                    "quintile_returns": list(horizon.quintile_returns),
                    "quintile_monotonicity": horizon.quintile_monotonicity,
                    "top_quintile_net_return": horizon.top_quintile_net_return,
                    "long_short_spread": horizon.long_short_spread,
                    "top_quintile_turnover": horizon.top_quintile_turnover,
                    "max_share_of_adv": horizon.max_share_of_adv,
                }
                for horizon in report.horizons
            ],
        },
    )

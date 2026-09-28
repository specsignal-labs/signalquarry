# SPDX-License-Identifier: Apache-2.0
"""Project factor discovery and synthetic conformance; no real-data evaluator yet."""

from __future__ import annotations

from pathlib import Path

from signalquarry._internal.project.factors import LoadedFactor, load_factors
from signalquarry._internal.project.project import ProjectError, find_root, load_config
from signalquarry._internal.validation.conformance import import_policy
from signalquarry._internal.validation.factor_conformance import run_factor_checks
from signalquarry.api.envelope import Envelope


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
    results = [import_policy(item.package_dir, item.module.__name__.split(".")[0], sdk_only=True)]
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

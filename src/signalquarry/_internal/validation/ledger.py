# SPDX-License-Identifier: Apache-2.0
"""Append-only, hash-chained records under ``<project>/evidence/``.

* ``trials.jsonl`` — one line per distinct (configuration, dataset) evaluated on
  real data. Written by the engine-facing commands only. The project-wide count
  of distinct configurations is the ``N`` of the deflated Sharpe ratio.
* ``factor_trials.jsonl`` — one line per distinct factor evaluation identity,
  separate from strategy trials and counted by factor configuration hash.
* ``freezes.jsonl`` — ``sqy spec freeze`` records (hypothesis, gates, holdout seal) and
  ``sqy holdout seal`` records (``kind: seal``: a family's seal without a freeze).
* ``holdouts.jsonl`` — the single permitted opening of each family's holdout.

Layouts (``signalquarry.toml`` → ``[evidence] layout``):

* ``project`` (default): the logs live in ``<project>/evidence/``.
* ``per_family``: each family's logs live in ``<family_root>/evidence/`` (default
  ``families/{family}``) so a family can be transferred with its own verifiable
  history. Every family append is mirrored in ``evidence/project_index.jsonl``
  (family, log, head, entries), and the deflated Sharpe ratio still counts every
  family's trials. A family log that no longer matches the index is corrupt.

Every line carries ``seq``, ``prev`` (hash of the previous line) and ``hash``.
A shortened or edited file fails verification.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from signalquarry._internal.canonical import canonical_hash, canonical_json, to_canonical


class LedgerError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class ChainedLog:
    path: Path
    schema: str

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        entries, previous = [], None
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError as exc:
                raise LedgerError("EVIDENCE_LOG_CORRUPT", f"{self.path.name}:{number}") from exc
            if not isinstance(entry, dict):
                raise LedgerError("EVIDENCE_LOG_CORRUPT", f"{self.path.name}:{number}")
            body = {k: v for k, v in entry.items() if k != "hash"}
            sequence = entry.get("seq")
            if (
                (self.schema != "any" and entry.get("schema") != self.schema)
                or type(sequence) is not int
                or sequence != len(entries) + 1
                or entry.get("prev") != previous
                or canonical_hash(body) != entry.get("hash")
            ):
                raise LedgerError("EVIDENCE_LOG_CORRUPT", f"{self.path.name}:{number}")
            entries.append(entry)
            previous = entry["hash"]
        return entries

    def head(self) -> str | None:
        entries = self.entries()
        return entries[-1]["hash"] if entries else None

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        entries = self.entries()
        body = to_canonical(
            {
                "schema": self.schema,
                **record,
                "seq": len(entries) + 1,
                "prev": entries[-1]["hash"] if entries else None,
            }
        )
        entry = {**body, "hash": canonical_hash(body)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(entry) + "\n")
        return entry


KINDS = ("trials", "factor_trials", "freezes", "holdouts")
SCHEMAS = {
    "trials": "signalquarry.trial/v1",
    "factor_trials": "signalquarry.factor-trial/v1",
    "freezes": "signalquarry.freeze/v1",
    "holdouts": "signalquarry.holdout-opening/v1",
}


@dataclass(frozen=True)
class Layout:
    per_family: bool
    family_root: str = "families/{family}"


def layout(root: Path) -> Layout:
    try:
        document = tomllib.loads((root / "signalquarry.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return Layout(per_family=False)
    evidence = document.get("evidence", {})
    kind = evidence.get("layout", "project")
    if kind not in ("project", "per_family"):
        raise LedgerError("PROJECT_CONFIG_INVALID", f"evidence.layout={kind}")
    family_root = str(evidence.get("family_root", "families/{family}"))
    if kind == "per_family" and family_root.count("{family}") != 1:
        raise LedgerError("PROJECT_CONFIG_INVALID", "evidence.family_root")
    if kind == "per_family":
        try:
            family_root.format(family="__signalquarry_family__")
        except (KeyError, ValueError) as exc:
            raise LedgerError("PROJECT_CONFIG_INVALID", "evidence.family_root") from exc
    return Layout(per_family=kind == "per_family", family_root=family_root)


def _log(root: Path, kind: str, family: str | None) -> ChainedLog:
    current = layout(root)
    if not current.per_family:
        return ChainedLog(root / "evidence" / f"{kind}.jsonl", SCHEMAS[kind])
    if family is None:
        raise LedgerError("EVIDENCE_FAMILY_REQUIRED", kind)
    return ChainedLog(
        root / current.family_root.format(family=family) / "evidence" / f"{kind}.jsonl", SCHEMAS[kind]
    )


def logs(root: Path, kind: str) -> list[ChainedLog]:
    """Every log of ``kind`` in the project (one per family in the per-family layout)."""
    current = layout(root)
    if not current.per_family:
        return [ChainedLog(root / "evidence" / f"{kind}.jsonl", SCHEMAS[kind])]
    pattern = current.family_root.format(family="*") + f"/evidence/{kind}.jsonl"
    return [ChainedLog(path, SCHEMAS[kind]) for path in sorted(root.glob(pattern))]


def project_index(root: Path) -> ChainedLog:
    return ChainedLog(root / "evidence" / "project_index.jsonl", "signalquarry.project-index/v1")


def trials(root: Path, family: str | None = None) -> ChainedLog:
    return _log(root, "trials", family)


def factor_trials(root: Path, family: str | None = None) -> ChainedLog:
    return _log(root, "factor_trials", family)


def freezes(root: Path, family: str | None = None) -> ChainedLog:
    return _log(root, "freezes", family)


def holdouts(root: Path, family: str | None = None) -> ChainedLog:
    return _log(root, "holdouts", family)


def append(root: Path, kind: str, family: str, record: dict[str, Any]) -> dict[str, Any]:
    """Append to a family's log; in the per-family layout, also chain its new head into the index."""
    log = _log(root, kind, family)
    entry = log.append(record)
    if layout(root).per_family:
        project_index(root).append(
            {"family": family, "log": kind, "head": entry["hash"], "entries": entry["seq"]}
        )
    return entry


def all_entries(root: Path, kind: str) -> list[dict[str, Any]]:
    rows = [entry for log in logs(root, kind) for entry in log.entries()]
    return sorted(rows, key=lambda e: (str(e.get("at", "")), e["seq"])) if layout(root).per_family else rows


def verify_index(root: Path) -> list[str]:
    """Per-family layout: every family log must end exactly where the index last saw it."""
    current = layout(root)
    if not current.per_family:
        return []
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in project_index(root).entries():
        latest[(entry["family"], entry["log"])] = entry
    problems = []
    seen = set()
    family_base = root / current.family_root.partition("{family}")[0]
    for kind in KINDS:
        for log in logs(root, kind):
            try:
                family = log.path.relative_to(family_base).parts[0]
            except (ValueError, IndexError) as exc:
                raise LedgerError("PROJECT_CONFIG_INVALID", "evidence.family_root") from exc
            seen.add((family, kind))
            entries = log.entries()
            expected = latest.get((family, kind))
            if (
                expected is None
                or not entries
                or entries[-1]["hash"] != expected.get("head")
                or type(expected.get("entries")) is not int
                or expected.get("entries") != len(entries)
            ):
                problems.append(f"EVIDENCE_INDEX_MISMATCH:{family}/{kind}")
    problems += [f"EVIDENCE_INDEX_MISMATCH:{f}/{k}" for (f, k) in latest if (f, k) not in seen]
    return problems


def record_trial(root: Path, trial: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Append a trial unless the same configuration was already evaluated on the same dataset."""
    for entry in trials(root, trial["family"]).entries():
        if (
            entry.get("kind") == "trial"
            and entry["configuration_hash"] == trial["configuration_hash"]
            and entry["dataset_identity"] == trial["dataset_identity"]
        ):
            return entry, False
    return append(root, "trials", trial["family"], trial), True


def trial_summary(root: Path, family: str | None = None) -> dict[str, Any]:
    entries = all_entries(root, "trials")
    runs = [e for e in entries if e.get("kind") == "trial"]
    configurations = {e["configuration_hash"]: e for e in runs}
    family_configs = {e["configuration_hash"] for e in runs if family is None or e["family"] == family}
    return {
        "project_count": len(configurations),
        "family_count": len(family_configs),
        "sharpes": [float(e["sharpe"]) for e in configurations.values() if e.get("sharpe") is not None],
        "extensions": sum(
            1
            for e in entries
            if e.get("kind") == "budget_extension" and (family is None or e["family"] == family)
        ),
        "head": _head(root, entries),
    }


def factor_trial_summary(root: Path, family: str | None = None) -> dict[str, Any]:
    """Count unique factor configurations without mixing strategy trials."""
    entries = all_entries(root, "factor_trials")
    runs = [entry for entry in entries if entry.get("kind") == "factor_trial"]
    project_configs = {entry["factor_configuration_hash"] for entry in runs}
    family_configs = {
        entry["factor_configuration_hash"]
        for entry in runs
        if family is None or entry.get("family") == family
    }
    selected = [entry for entry in runs if family is None or entry.get("family") == family]
    return {
        "project_count": len(project_configs),
        "family_count": len(family_configs),
        "trial_count": len(selected),
        "head": _head(root, entries),
    }


def _head(root: Path, entries: list[dict[str, Any]]) -> str | None:
    if layout(root).per_family:
        index = project_index(root).entries()
        return index[-1]["hash"] if index else None
    return entries[-1]["hash"] if entries else None


def latest_freeze(root: Path, strategy_id: str) -> dict[str, Any] | None:
    matches = [
        e for e in all_entries(root, "freezes") if e["strategy_id"] == strategy_id and e.get("kind") != "seal"
    ]
    return matches[-1] if matches else None


def holdout_opened(root: Path, family: str) -> dict[str, Any] | None:
    return next((e for e in holdouts(root, family).entries() if e["family"] == family), None)

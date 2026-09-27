# SPDX-License-Identifier: Apache-2.0
"""Build an EvidenceBundleV1 for one strategy family.

The ``public`` tier follows the family's publication policy (default-deny):

* ``strategies/<id>/profile.json`` — always; rule text only at the ``rules`` tier.
* ``results/<run>.json`` — the latest backtest per strategy, only for non-commercial
  families that allow backtest results.
* ``forward/<id>.json`` — weekly (or monthly) paper returns from the deployment's
  journal, only for commercial families with a forward policy, only for periods that
  ended at least ``lag_days`` ago. Positions, fills and exposure are always withheld.
* ``commitments/`` — commitment records and proofs, when present.

The ``nda`` tier adds the full evidence for a data room: every run result and equity
series of the family's strategies, the trial ledger and freezes, and the paper journals.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from signalquarry._internal.canonical import canonical_hash, to_canonical
from signalquarry._internal.contracts.publication import PublicationV1
from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.engine.run import is_options
from signalquarry._internal.validation.ledger import ChainedLog, all_entries

MEDIA_TYPES = {
    ".json": "application/json",
    ".md": "text/markdown",
    ".svg": "image/svg+xml",
    ".ots": "application/vnd.opentimestamps",
    ".tsr": "application/timestamp-reply",
}


class ExportError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class StrategyEvidence:
    spec: StrategySpecV1
    configuration_hash: str


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_canonical(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _runs(root: Path, name: str, strategy_id: str) -> list[tuple[Path, dict[str, Any]]]:
    found = []
    for path in sorted((root / ".signalquarry" / "runs").glob(f"*/{name}")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if document.get("strategy_id") == strategy_id:
            found.append((path.parent, document))
    return found


def _week_end(day: date) -> date:
    return day + timedelta(days=(4 - day.weekday()) % 7)  # the Friday of that week


def _month_end(day: date) -> date:
    following = date(day.year + (day.month == 12), day.month % 12 + 1, 1)
    return following - timedelta(days=1)


def forward_periods(journal: Path, *, resolution: str, lag_days: int, today: date) -> list[dict[str, Any]]:
    """Period returns from pre-open equity observations in a paper journal (no positions or fills)."""
    if not journal.is_file():
        return []
    points: dict[date, float] = {}
    for entry in ChainedLog(journal, "any").entries():
        if entry.get("kind") == "session_started":
            points[date.fromisoformat(entry["session"])] = float(entry["equity"])
    if len(points) < 2:
        return []
    close_of = _week_end if resolution == "weekly" else _month_end
    period_last: dict[date, float] = {}
    first = points[min(points)]
    for day in sorted(points):
        period_last[close_of(day)] = points[day]
    cutoff = today - timedelta(days=lag_days)
    rows, previous, peak = [], first, first
    for ending in sorted(period_last):
        if ending > cutoff:
            break
        value = period_last[ending]
        peak = max(peak, value)
        rows.append(
            {
                "week_ending": ending.isoformat(),
                "return": round(value / previous - 1.0, 8),
                "cumulative_return": round(value / first - 1.0, 8),
                "drawdown": round(1.0 - value / peak, 8),
            }
        )
        previous = value
    return rows


def _claim(root: Path, strategy_id: str, configuration_hash: str) -> tuple[str, str]:
    claim, grade = _backtest_claim(root, strategy_id, configuration_hash)
    passed = any(
        document.get("configuration_hash") == configuration_hash and (document.get("g5") or {}).get("ok")
        for _, document in _runs(root, "drift.json", strategy_id)
    )
    if passed and claim in ("walk_forward", "holdout_passed"):
        return "paper_forward", "paper"
    return claim, grade


def _backtest_claim(root: Path, strategy_id: str, configuration_hash: str) -> tuple[str, str]:
    for _, document in reversed(_runs(root, "evaluation.json", strategy_id)):
        if document.get("configuration_hash") == configuration_hash:
            evidence = document.get("evidence") or {}
            return evidence.get("claim_level", "none"), evidence.get("grade", "none")
    for _, document in reversed(_runs(root, "result.json", strategy_id)):
        if document.get("configuration_hash") == configuration_hash:
            evidence = document.get("evidence") or {}
            return evidence.get("claim_level", "none"), evidence.get("grade", "none")
    return "none", "none"


def _profile(
    publication: PublicationV1, item: Any, root: Path, evidence: StrategyEvidence, forward: bool
) -> dict[str, Any]:
    claim, grade = _claim(root, item.id, evidence.configuration_hash)
    profile: dict[str, Any] = {
        "schema": "signalquarry.showcase.strategy-profile/v1",
        "id": item.id,
        "title": item.title,
        "family": publication.family_label,
        "kind": "research",
        # The spec decides, so an options strategy can never be published as equity.
        "asset_class": "options" if is_options(evidence.spec) else "equity",
        "assets": item.assets,
        "horizon": item.horizon,
        "summary": item.summary,
        "disclosure_tier": publication.tier,
        "commercial": publication.commercial,
        "status": "forward_paper" if forward else "research",
        "evidence_grade": "paper"
        if forward
        else (
            {
                "synthetic": "synthetic",
                "historical": "historical",
                "low_evidence_options": "low_evidence_options",
            }.get(grade, "none")
        ),
        "claim_level": claim,
        "primary_result": None,
        "children": [],
    }
    if item.capacity_note:
        profile["capacity_note"] = item.capacity_note
    if item.rules is not None:
        profile["rules"] = item.rules.model_dump(mode="json")
    return profile


# Showcase study-result evidence: synthetic runs are diagnostics, modelled option prices a proxy.
RESULT_EVIDENCE = {
    "synthetic": "diagnostic",
    "low_evidence_options": "option_proxy",
    "historical": "historical",
}


def _result(root: Path, strategy_id: str, spec: StrategySpecV1) -> dict[str, Any] | None:
    runs = _runs(root, "result.json", strategy_id)
    if not runs:
        return None
    directory, document = runs[-1]
    metrics = document.get("metrics", {})
    evidence = document.get("evidence") or {}
    costs = spec.execution.costs
    return {
        "schema": "signalquarry.showcase.study-result/v1",
        "id": f"{strategy_id}-{document['run_id']}".lower(),
        "strategy_id": strategy_id,
        "variant": document["configuration_hash"].removeprefix("sha256:")[:12],
        "title": f"{spec.id} {spec.version} backtest",
        "test_start": metrics.get("start"),
        "test_end": metrics.get("end"),
        "published_at": document["created_at"][:10],
        "evidence": RESULT_EVIDENCE.get(evidence.get("grade", ""), "historical"),
        "status": f"Claim level {evidence.get('claim_level', 'none')}",
        "conclusion": f"Backtest of {spec.id} {spec.version} on {evidence.get('grade', 'unknown')} data; see the claim level for what it establishes.",
        "assumptions": (
            f"{spec.execution.model} execution, {spec.execution.sizing} sizing, {spec.account.model} account, "
            + (
                f"modelled option prices, {spec.options.per_contract_fee}/contract fee, "
                f"{spec.options.spread_haircut} of the spread paid on every fill."
                if spec.options is not None
                else f"costs {costs.bps} bps + {costs.per_share}/share."
            )
        ),
        "limitations": [
            f"Evidence grade: {evidence.get('grade', 'unknown')}",
            "Hypothetical simulated results",
        ],
        "source": {
            "path": (directory / "result.json").relative_to(root).as_posix(),
            "sha256": hashlib.sha256((directory / "result.json").read_bytes()).hexdigest(),
        },
        "hypothetical": True,
        "commercial": False,
        "metrics": {
            "total_return": metrics.get("total_return"),
            "annualized_return": metrics.get("cagr"),
            "max_drawdown": metrics.get("max_drawdown"),
            "sharpe": metrics.get("sharpe"),
            "trades": metrics.get("fills"),
        },
    }


def _nda_files(root: Path, family: str, strategy_ids: set[str], out: Path) -> None:
    for strategy_id in sorted(strategy_ids):
        for name in ("result.json", "evaluation.json"):
            for directory, _ in _runs(root, name, strategy_id):
                _write(
                    out / "nda" / "runs" / directory.name / name, json.loads((directory / name).read_text())
                )
                series = directory / "equity.csv"
                if series.is_file():
                    with series.open(encoding="utf-8") as handle:
                        _write(
                            out / "nda" / "runs" / directory.name / "equity.json",
                            list(csv.DictReader(handle)),
                        )
    for kind in ("trials", "freezes", "holdouts"):
        rows = [e for e in all_entries(root, kind) if e.get("family") == family]
        if rows:
            _write(out / "nda" / "evidence" / f"{kind}.json", rows)


WITHHELD_JOURNAL_KEYS = ("order_id",)  # broker identifiers stay out of shared artifacts


def _withhold(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: "withheld" if k in WITHHELD_JOURNAL_KEYS and v is not None else _withhold(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_withhold(item) for item in value]
    return value


def _nda_paper(root: Path, deployments: set[str], out: Path) -> None:
    """Paper journals (broker order ids withheld; chain head and length kept) and captures."""
    for alias in sorted(deployments):
        directory = root / "paper" / alias
        journal = directory / "journal.jsonl"
        if journal.is_file():
            entries = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line]
            _write(
                out / "nda" / "paper" / alias / "journal.json",
                {
                    "deployment": alias,
                    "count": len(entries),
                    "head": entries[-1]["hash"] if entries else None,
                    "withheld": list(WITHHELD_JOURNAL_KEYS),
                    "entries": [_withhold(e) for e in entries],
                },
            )
        captures = directory / "captures"
        if captures.is_dir():
            for path in sorted(captures.glob("*.json")):
                _write(out / "nda" / "paper" / alias / "captures" / path.name, json.loads(path.read_text()))


def build_bundle(
    root: Path,
    publication: PublicationV1,
    strategies: dict[str, StrategyEvidence],
    *,
    tier: str,
    out: Path,
    produced_at: str,
    today: date,
    framework_version: str,
    project: str,
    supersedes: str | None,
) -> dict[str, Any]:
    missing = [item.id for item in publication.strategies if item.id not in strategies]
    if missing:
        raise ExportError("PUBLICATION_STRATEGY_UNKNOWN", ",".join(missing))
    wrong_family = [
        item.id for item in publication.strategies if strategies[item.id].spec.family != publication.family
    ]
    if wrong_family:
        raise ExportError("PUBLICATION_FAMILY_MISMATCH", ",".join(wrong_family))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for item in publication.strategies:
        evidence = strategies[item.id]
        weeks: list[dict[str, Any]] = []
        if publication.forward is not None and item.deployment is not None:
            weeks = forward_periods(
                root / "paper" / item.deployment / "journal.jsonl",
                resolution=publication.forward.resolution,
                lag_days=publication.forward.lag_days,
                today=today,
            )
            _write(
                out / "forward" / f"{item.id}.json",
                {
                    "schema": "signalquarry.showcase.forward-record/v1",
                    "strategy_id": item.id,
                    "account_kind": "paper",
                    "resolution": publication.forward.resolution,
                    "lag_days": publication.forward.lag_days,
                    "weeks": weeks,
                    "spy_correlation": None,
                    "spy_beta": None,
                    "positions": "withheld",
                    "fills": "withheld",
                    "exposure": "withheld",
                },
            )
        _write(
            out / "strategies" / item.id / "profile.json",
            _profile(publication, item, root, evidence, bool(weeks)),
        )
        if not publication.commercial and publication.backtest_results == "allow":
            result = _result(root, item.id, evidence.spec)
            if result is not None:
                _write(out / "results" / f"{result['id']}.json", result)
    commitments = root / "evidence" / "commitments"
    if commitments.is_dir():
        for path in sorted(commitments.rglob("*")):
            if (
                path.is_file()
                and path.suffix in MEDIA_TYPES
                and publication.family in path.relative_to(commitments).parts[:1]
            ):
                target = out / "commitments" / path.relative_to(commitments)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    if tier == "nda":
        _nda_files(root, publication.family, {item.id for item in publication.strategies}, out)
        _nda_paper(root, {item.deployment for item in publication.strategies if item.deployment}, out)
    files = []
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        relative = path.relative_to(out).as_posix()
        media = MEDIA_TYPES.get(path.suffix)
        if media is None:
            raise ExportError("EXPORT_FILE_TYPE_NOT_ALLOWED", relative)
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "media_type": media,
                "tier": "nda" if relative.startswith("nda/") else "public",
            }
        )
    manifest: dict[str, Any] = {
        "schema": "signalquarry-evidence-bundle/v1",
        "bundle_id": f"{publication.family}-{tier}-{produced_at[:10]}",
        "produced_at": produced_at,
        "canonical": "v2",
        "tier": tier,
        "producer": {"framework_version": framework_version, "project": project[:120]},
        "files": files,
    }
    if supersedes:
        manifest["supersedes"] = supersedes
    manifest["bundle_hash"] = canonical_hash(manifest)
    _write(out / "bundle.json", manifest)
    return manifest

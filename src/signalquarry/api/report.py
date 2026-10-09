# SPDX-License-Identifier: Apache-2.0
"""``sqy report``: render the latest backtest and evaluation of a strategy as Markdown + SVG."""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from signalquarry._internal.canonical import file_sha256
from signalquarry._internal.evidence.report import equity_svg, render_report
from signalquarry._internal.project.project import ProjectError, find_root, load_config, load_strategies
from signalquarry._internal.validation.ledger import LedgerError, trial_summary
from signalquarry.api.envelope import Envelope
from signalquarry.plugins import ReportContext, discover


def _runs(root: Path, name: str, strategy_id: str) -> list[tuple[Path, dict[str, Any]]]:
    found: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((root / ".signalquarry" / "runs").glob(f"*/{name}")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if document.get("strategy_id") == strategy_id:
            found.append((path.parent, document))
    return found


def _benchmark_symbol(result: dict[str, Any] | None) -> str:
    block = (result or {}).get("benchmark")
    symbol = cast(dict[str, Any], block).get("symbol") if isinstance(block, dict) else None
    return symbol if isinstance(symbol, str) else "benchmark"


def _curve(path: Path) -> list[tuple[str, Decimal]]:
    with path.open(encoding="utf-8") as handle:
        return [(row["session"], Decimal(row["equity"])) for row in csv.DictReader(handle)]


def report(strategy_id: str, *, run_id: str | None = None, project: Path | None = None) -> Envelope:
    envelope = Envelope(command="report")
    try:
        root = find_root(project)
        strategies = load_strategies(load_config(root))
    except ProjectError as exc:
        envelope.status, envelope.reason_codes, envelope.summary = (
            "invalid",
            [exc.code],
            exc.detail or exc.code,
        )
        return envelope
    strategy = strategies.get(strategy_id)
    if strategy is None:
        envelope.status, envelope.reason_codes, envelope.summary = (
            "invalid",
            ["STRATEGY_NOT_FOUND"],
            strategy_id,
        )
        return envelope
    backtests = _runs(root, "result.json", strategy_id)
    if run_id is not None:
        backtests = [item for item in backtests if item[0].name == run_id]
    evaluations = _runs(root, "evaluation.json", strategy_id)
    if not backtests and not evaluations:
        envelope.status, envelope.reason_codes = "invalid", ["REPORT_NO_RUNS"]
        envelope.summary = f"no backtest or evaluation runs for {strategy_id}" + (
            f" with run id {run_id}" if run_id else ""
        )
        envelope.next_actions = [
            {"command": f"sqy backtest --strategy {strategy_id}", "why": "Create a run to report on."}
        ]
        return envelope
    result_dir, result = backtests[-1] if backtests else (None, None)
    configuration = (result or evaluations[-1][1])["configuration_hash"]
    matching = [item for item in evaluations if item[1]["configuration_hash"] == configuration]
    evaluation = matching[-1][1] if matching else None
    if evaluation is None:
        envelope.next_actions.append(
            {
                "command": f"sqy evaluate --strategy {strategy_id}",
                "why": "Gates decide what the report may claim.",
            }
        )
    if configuration != strategy.configuration_hash:
        envelope.warnings.append("REPORT_CONFIGURATION_CHANGED")
    try:
        trials = trial_summary(root, strategy.spec.family)
    except LedgerError as exc:
        envelope.status, envelope.reason_codes, envelope.summary = "blocked", [exc.code], str(exc)
        return envelope
    chart = None
    if result_dir is not None and (result_dir / "equity.csv").is_file():
        chart = (
            equity_svg(
                _curve(result_dir / "equity.csv"),
                benchmark=_curve(result_dir / "benchmark.csv")
                if (result_dir / "benchmark.csv").is_file()
                else None,
                benchmark_label=_benchmark_symbol(result),
            )
            or None
        )
    text = render_report(
        spec=strategy.spec.model_dump(mode="json", by_alias=True),
        result=result,
        evaluation=evaluation,
        trials=trials,
        chart=chart,
    )
    text += _plugin_sections(
        strategy.spec.model_dump(mode="json", by_alias=True), result, evaluation, envelope
    )
    label = (
        result_dir.name
        if result_dir is not None
        else matching[-1][0].name
        if matching
        else evaluations[-1][0].name
    )
    out = root / ".signalquarry" / "reports" / strategy_id / label
    out.mkdir(parents=True, exist_ok=True)
    files = {"report.md": text, **({"equity.svg": chart} if chart else {})}
    for name, content in files.items():
        (out / name).write_text(content, encoding="utf-8")
    envelope.artifacts = [
        {
            "path": str((out / name).relative_to(root)),
            "sha256": file_sha256(out / name),
            "kind": name.split(".")[0],
        }
        for name in files
    ]
    source: dict[str, Any] = evaluation or result or {}
    evidence: dict[str, Any] | None = source.get("evidence")
    envelope.evidence = evidence
    envelope.summary = f"report for {strategy_id} at {out.relative_to(root)}/report.md"
    envelope.data = {
        "backtest_run": result["run_id"] if result else None,
        "evaluation_run": evaluation["run_id"] if evaluation else None,
        "claim_level": evidence["claim_level"] if evidence and "claim_level" in evidence else None,
    }
    return envelope


def _plugin_sections(
    spec: dict[str, Any],
    result: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
    envelope: Envelope,
) -> str:
    """Installed report-section plugins, appended after the framework's sections."""
    discovery = discover("report_sections")
    envelope.warnings += [f"PLUGIN_ERROR:{error}" for error in discovery.errors]
    context = ReportContext(spec=spec, result=result, evaluation=evaluation)
    parts: list[str] = []
    for loaded in discovery.plugins:
        try:
            body = loaded.plugin.render(context)
            if not isinstance(body, str):
                raise TypeError("render must return Markdown text")
        except Exception as exc:  # noqa: BLE001 - a broken section is left out, with a warning
            envelope.warnings.append(f"PLUGIN_SECTION_ERROR:{loaded.name}:{type(exc).__name__}")
            continue
        parts += [
            "",
            f"## {loaded.plugin.title}",
            "",
            f"*Section from plugin `{loaded.source}`.*",
            "",
            body.strip(),
            "",
        ]
    return "\n".join(parts)

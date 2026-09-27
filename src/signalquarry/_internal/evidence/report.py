# SPDX-License-Identifier: Apache-2.0
"""``report.md`` and ``equity.svg`` rendered from run artifacts only. Numbers are never recomputed here.

The report states the evidence grade and claim level first and uses only the
wording the claim ladder permits for that level.
"""

from __future__ import annotations

from decimal import Decimal
from html import escape
from typing import Any

CLAIM_WORDING = {
    "none": "The strategy runs. Nothing may be said about its performance.",
    "in_sample": "It performed as shown on the data it was developed on. This is not predictive.",
    "walk_forward": "It held up out of sample in walk-forward tests.",
    "holdout_passed": "It passed the sealed holdout once.",
    "paper_forward": "It traded forward on a paper account for the stated number of sessions.",
}
DISCLAIMER = (
    "Hypothetical, simulated results. They do not represent actual trading, do not reflect all "
    "costs or market conditions, and are not investment advice. Past or simulated performance "
    "does not predict future results."
)


def _percent(value: Any) -> str:
    return "–" if value is None else f"{float(value):.2%}"


def _number(value: Any, digits: int = 2) -> str:
    return "–" if value is None else f"{float(value):.{digits}f}"


def _gate_detail(gate: dict[str, Any]) -> str:
    parts = []
    for key, value in gate.items():
        if key in ("ok", "scenarios"):
            continue
        parts.append(f"{key}={_number(value, 4) if isinstance(value, float) else value}")
    for name, scenario in (gate.get("scenarios") or {}).items():
        parts.append(f"{name}:{'pass' if scenario.get('ok') else 'fail'}")
    return ", ".join(parts)


def equity_svg(points: list[tuple[str, Decimal]], *, width: int = 720, height: int = 240) -> str:
    """A dependency-free line chart of equity; dates on the first and last tick only."""
    if len(points) < 2:
        return ""
    values = [float(v) for _, v in points]
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    pad = 32
    step = (width - 2 * pad) / (len(values) - 1)
    coords = " ".join(
        f"{pad + i * step:.1f},{height - pad - (v - low) / span * (height - 2 * pad):.1f}"
        for i, v in enumerate(values)
    )
    first, last = escape(points[0][0]), escape(points[-1][0])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Equity from {first} to {last}">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f'<polyline fill="none" stroke="#1f5fa8" stroke-width="1.5" points="{coords}"/>'
        f'<text x="{pad}" y="{height - 8}" font-size="11" font-family="sans-serif">{first}</text>'
        f'<text x="{width - pad}" y="{height - 8}" font-size="11" font-family="sans-serif" '
        f'text-anchor="end">{last}</text>'
        f'<text x="4" y="{pad}" font-size="11" font-family="sans-serif">{high:,.0f}</text>'
        f'<text x="4" y="{height - pad}" font-size="11" font-family="sans-serif">{low:,.0f}</text>'
        "</svg>\n"
    )


def render_report(
    *,
    spec: dict[str, Any],
    result: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
    trials: dict[str, Any],
    chart: str | None,
) -> str:
    evidence = (evaluation or result or {}).get("evidence") or {}
    grade = evidence.get("grade", "unknown")
    claim = evidence.get("claim_level", "none")
    lines = [
        f"# {spec['id']} {spec['version']}",
        "",
        f"> **Evidence:** grade `{grade}`, claim level `{claim}`. {CLAIM_WORDING.get(claim, '')}",
    ]
    if grade == "synthetic":
        lines.append("> Synthetic data: these numbers say nothing about real markets.")
    lines += [
        "",
        "## Hypothesis",
        "",
        spec["hypothesis"]["statement"],
        "",
        f"*Falsified if:* {spec['hypothesis']['falsification']}",
        "",
        "## Trials",
        "",
        f"- Family `{spec['family']}`: {trials['family_count']} configuration(s) evaluated on real data.",
        f"- Project: {trials['project_count']} configuration(s); the deflated Sharpe ratio uses this count.",
        f"- Holdout: {evidence.get('holdout', 'unknown')}.",
        "",
    ]
    if result is not None:
        m = result.get("metrics", {})
        lines += [
            "## Backtest",
            "",
            f"Run `{result['run_id']}` on dataset `{result.get('dataset_id')}`.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Sessions | {m.get('sessions', '–')} |",
            f"| Total return | {_percent(m.get('total_return'))} |",
            f"| CAGR | {_percent(m.get('cagr'))} |",
            f"| Annual volatility | {_percent(m.get('annual_volatility'))} |",
            f"| Sharpe (annualized, rf=0) | {_number(m.get('sharpe'))} |",
            f"| Max drawdown | {_percent(m.get('max_drawdown'))} |",
            f"| Fills | {m.get('fills', '–')} |",
            "",
        ]
        if chart:
            lines += ["![Equity](equity.svg)", ""]
    if evaluation is not None:
        oos = evaluation.get("oos", {})
        lines += [
            "## Evaluation",
            "",
            f"Out of sample across {len(evaluation.get('folds', []))} walk-forward fold(s): "
            f"total return {_percent(oos.get('total_return'))}, Sharpe {_number(oos.get('sharpe_annual'))}, "
            f"max drawdown {_percent(oos.get('max_drawdown'))}, PSR {_number(oos.get('psr'), 3)}, "
            f"DSR {_number(oos.get('dsr'), 3)}."
            + (
                f" Bootstrap 90% interval for the annual Sharpe: {_number(oos['sharpe_annual_90'][0])} to "
                f"{_number(oos['sharpe_annual_90'][1])}."
                if oos.get("sharpe_annual_90")
                else ""
            ),
            "",
            "| Gate | Result | Detail |",
            "|---|---|---|",
        ]
        for name, gate in evaluation.get("gates", {}).items():
            lines.append(f"| {name} | {'pass' if gate.get('ok') else 'FAIL'} | {_gate_detail(gate)} |")
        lines.append("")
    lines += ["## Identity", ""]
    for label, source, key in (
        ("Configuration", result or evaluation, "configuration_hash"),
        ("Dataset", result, "dataset_identity"),
        ("Ledger", result, "ledger_hash"),
        ("Backtest result", result, "result_hash"),
        ("Evaluation result", evaluation, "result_hash"),
    ):
        if source and source.get(key):
            lines.append(f"- {label}: `{source[key]}`")
    lines += ["", "---", "", f"*{DISCLAIMER}*", ""]
    return "\n".join(lines)

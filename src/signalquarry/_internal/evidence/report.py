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


def equity_svg(
    points: list[tuple[str, Decimal]],
    *,
    width: int = 720,
    height: int = 240,
    benchmark: list[tuple[str, Decimal]] | None = None,
    benchmark_label: str = "benchmark",
) -> str:
    """A dependency-free line chart of equity; dates on the first and last tick only.

    ``benchmark`` adds a second, dashed line on the same scale. It is drawn only when it has
    one point per equity point.
    """
    if len(points) < 2:
        return ""
    values = [float(v) for _, v in points]
    other = [float(v) for _, v in benchmark] if benchmark and len(benchmark) == len(points) else []
    low, high = min(values + other), max(values + other)
    span = (high - low) or 1.0
    pad = 32
    step = (width - 2 * pad) / (len(values) - 1)

    def line(series: list[float]) -> str:
        return " ".join(
            f"{pad + i * step:.1f},{height - pad - (v - low) / span * (height - 2 * pad):.1f}"
            for i, v in enumerate(series)
        )

    first, last = escape(points[0][0]), escape(points[-1][0])
    label = escape(benchmark_label)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Equity from {first} to {last}{f" against {label}" if other else ""}">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        + (
            f'<polyline fill="none" stroke="#8a8a8a" stroke-width="1.2" stroke-dasharray="4 3" '
            f'points="{line(other)}"/>'
            f'<text x="{width - pad}" y="14" font-size="11" font-family="sans-serif" text-anchor="end" '
            f'fill="#8a8a8a">- - {label}</text>'
            f'<text x="{width - pad}" y="28" font-size="11" font-family="sans-serif" text-anchor="end" '
            f'fill="#1f5fa8">— strategy</text>'
            if other
            else ""
        )
        + f'<polyline fill="none" stroke="#1f5fa8" stroke-width="1.5" points="{line(values)}"/>'
        f'<text x="{pad}" y="{height - 8}" font-size="11" font-family="sans-serif">{first}</text>'
        f'<text x="{width - pad}" y="{height - 8}" font-size="11" font-family="sans-serif" '
        f'text-anchor="end">{last}</text>'
        f'<text x="4" y="{pad}" font-size="11" font-family="sans-serif">{high:,.0f}</text>'
        f'<text x="4" y="{height - pad}" font-size="11" font-family="sans-serif">{low:,.0f}</text>'
        "</svg>\n"
    )


def _difference(left: Any, right: Any, kind: str) -> str:
    if left is None or right is None:
        return "–"
    gap = float(left) - float(right)
    return f"{gap:+.2%}" if kind == "percent" else f"{gap:+.2f}"


def _benchmark_lines(metrics: dict[str, Any], block: dict[str, Any]) -> list[str]:
    """Strategy against the buy-and-hold benchmark over the same sessions."""
    symbol, other, relative = block["symbol"], block.get("metrics", {}), block.get("relative", {})
    lines = [
        f"### Versus {symbol} buy-and-hold",
        "",
        f"| Metric | Strategy | {symbol} | Difference |",
        "|---|---|---|---|",
    ]
    for label, key, kind in (
        ("Total return", "total_return", "percent"),
        ("CAGR", "cagr", "percent"),
        ("Annual volatility", "annual_volatility", "percent"),
        ("Sharpe (annualized, rf=0)", "sharpe", "number"),
        ("Max drawdown", "max_drawdown", "percent"),
    ):
        show = _percent if kind == "percent" else _number
        lines.append(
            f"| {label} | {show(metrics.get(key))} | {show(other.get(key))} | "
            f"{_difference(metrics.get(key), other.get(key), kind)} |"
        )
    lines.append("")
    if relative.get("status") == "ok":
        lines.append(
            f"Beta {_number(relative.get('beta'))}, alpha {_percent(relative.get('alpha'))} a year, "
            f"correlation {_number(relative.get('correlation'))}, tracking error "
            f"{_percent(relative.get('tracking_error'))}, information ratio "
            f"{_number(relative.get('information_ratio'))}, up capture {_number(relative.get('up_capture'))}, "
            f"down capture {_number(relative.get('down_capture'))} "
            f"({relative.get('observations')} sessions)."
        )
    else:
        lines.append(
            f"Too few sessions ({relative.get('observations', 0)}) for statistics relative to the benchmark."
        )
    lines.append(
        f"The benchmark is one position in {symbol} under the same account and cost settings, "
        + (
            f"invested from {block['invested_from']}."
            if block.get("invested_from")
            else "which could not be bought in this window."
        )
    )
    lines.append("")
    return lines


def _drawdown_lines(episodes: list[dict[str, Any]]) -> list[str]:
    lines = [
        "### Drawdowns",
        "",
        "| Peak | Trough | Recovered | Depth | Sessions to trough | Sessions to recover |",
        "|---|---|---|---|---|---|",
    ]
    for item in episodes:
        lines.append(
            f"| {item.get('peak') or 'start'} | {item['trough']} | {item.get('recovery') or 'not yet'} | "
            f"{_percent(item['depth'])} | {item['sessions_to_trough']} | "
            f"{'–' if item.get('sessions_to_recovery') is None else item['sessions_to_recovery']} |"
        )
    lines.append("")
    return lines


def _activity_lines(activity: dict[str, Any], fees: Any) -> list[str]:
    holding = activity.get("implied_holding_sessions")
    return [
        "### Trading",
        "",
        f"One-way turnover {_number(activity.get('turnover'))} times a year"
        + ("" if holding is None else f" (about {holding:,.0f} sessions per holding)")
        + f"; fees {fees} in total, {_percent(activity.get('fee_drag'))} of mean equity a year.",
        "",
    ]


def _fold_lines(folds: list[dict[str, Any]], oos: dict[str, Any]) -> list[str]:
    versus = (oos.get("benchmark") or {}).get("symbol")
    lines = [
        "| Fold | Sessions | Return | Sharpe | Max drawdown |"
        + (f" {versus} return | Excess |" if versus else ""),
        "|---|---|---|---|---|" + ("---|---|" if versus else ""),
    ]
    for fold in folds:
        lines.append(
            f"| {fold['start']} to {fold['end']} | {fold.get('sessions', '–')} | "
            f"{_percent(fold.get('total_return'))} | {_number(fold.get('sharpe_annual'))} | "
            f"{_percent(fold.get('max_drawdown'))} |"
            + (
                f" {_percent(fold.get('benchmark_total_return'))} | "
                f"{_difference(fold.get('total_return'), fold.get('benchmark_total_return'), 'percent')} |"
                if versus
                else ""
            )
        )
    lines.append("")
    if versus:
        benchmark, relative = oos["benchmark"], oos.get("relative", {})
        lines.append(
            f"Ahead of {versus} buy-and-hold in {oos.get('folds_ahead_of_benchmark', 0)} of {len(folds)} "
            f"fold(s). Out of sample {versus} returned {_percent(benchmark.get('total_return'))} with a max "
            f"drawdown of {_percent(benchmark.get('max_drawdown'))}"
            + (
                f"; beta {_number(relative.get('beta'))}, information ratio "
                f"{_number(relative.get('information_ratio'))}."
                if relative.get("status") == "ok"
                else "."
            )
        )
        lines.append("")
    return lines


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
        if result.get("benchmark"):
            lines += _benchmark_lines(m, result["benchmark"])
        if result.get("drawdowns"):
            lines += _drawdown_lines(result["drawdowns"])
        if (result.get("activity") or {}).get("status") == "ok":
            lines += _activity_lines(result["activity"], m.get("fees", "–"))
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
        if evaluation.get("folds"):
            lines += _fold_lines(evaluation["folds"], oos)
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

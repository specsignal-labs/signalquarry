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


def _factor_text(value: Any) -> str:
    return "–" if value is None else escape(str(value))


def _factor_ic(value: Any) -> str:
    """An information coefficient to four decimals; the exact value stays in the envelope."""
    return "–" if value is None else f"{float(value):.4f}"


def render_factor_report(*, spec: dict[str, Any], diagnostics: dict[str, Any]) -> str:
    """Render descriptive factor diagnostics without recomputing or grading them."""
    scope = diagnostics["scope"]
    statement = (
        "synthetic data: these numbers say nothing about real markets"
        if scope == "synthetic"
        else "descriptive diagnostics on recorded data whose provenance is not verified"
    )
    hypothesis = spec.get("hypothesis") or {}
    lines = [
        f"# Factor report: {_factor_text(spec.get('id'))}",
        "",
        f"> **Scope: `{scope}`** — {statement}.",
        "> No trial was recorded, no evidence grade assigned and no holdout accessed.",
        "",
        "## Hypothesis",
        "",
        _factor_text(hypothesis.get("statement")),
        "",
        "**Falsification:** " + _factor_text(hypothesis.get("falsification")),
        "",
        "## Diagnostics by horizon",
        "",
        "Horizons are in sessions. ICIR is unannualized. Returns are mean horizon outcomes, "
        "not compounded portfolio returns. The long-short spread is a statistic, "
        "not an executable portfolio. Turnover is one-way; capacity uses predecision ADV.",
        "",
        "| Horizon | Observations | Mean IC | ICIR | Top-quintile net return | "
        "Long-short spread (statistic) | Top-quintile turnover | Maximum share of ADV | "
        "Quintile monotonicity |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    horizons = diagnostics.get("horizons") or []
    for row in horizons:
        cells = [
            _factor_text(row.get("horizon")),
            _factor_text(row.get("observations")),
            _factor_ic(row.get("mean_ic")),
            _number(row.get("icir"), 4),
            _percent(row.get("top_quintile_net_return")),
            _percent(row.get("long_short_spread")),
            _percent(row.get("top_quintile_turnover")),
            _percent(row.get("max_share_of_adv")),
            _number(row.get("quintile_monotonicity"), 4),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Chronological-block IC", "", "Blocks are listed from earliest to latest."]
    for row in horizons:
        lines += [
            "",
            f"### Horizon {_factor_text(row.get('horizon'))}",
            "",
            "| Block | Mean IC |",
            "|---|---|",
        ]
        blocks = row.get("chronological_blocks") or [None]
        lines += [f"| {index} | {_factor_ic(value)} |" for index, value in enumerate(blocks, 1)]
    lines += [
        "",
        "## Charts",
        "",
        "![Mean rank IC by horizon](ic.svg)",
        "",
        "![Mean outcome by quintile for each horizon](quintiles.svg)",
        "",
        "## Identity",
        "",
    ]
    for label, key in (
        ("Factor configuration hash", "configuration_hash"),
        ("Dataset identity", "dataset_identity"),
        ("Universe identity", "universe_identity"),
        ("Label identity", "label_identity"),
    ):
        lines += [f"**{label}:** {_factor_text(diagnostics.get(key))}", ""]
    if scope == "synthetic":
        lines += ["**Synthetic panel arguments:**", ""]
        for key, value in (diagnostics.get("synthetic") or {}).items():
            lines.append(f"- {_factor_text(key)}: {_factor_text(value)}")
        lines.append("")
    lines += [DISCLAIMER, ""]
    return "\n".join(lines)


def _factor_bars_svg(
    *, title: str, labels: list[str], series: list[tuple[str, list[Any]]], percent: bool
) -> str:
    """Grouped bars with a shared zero line; missing values have labels but no bars."""
    width, top, bottom, left, right = 720, 32, 196, 76, 704
    height = 240 + 24 * ((len(series) + 3) // 4)
    values = [float(value) for _, points in series for value in points if value is not None]
    low, high = min([0.0, *values]), max([0.0, *values])
    padding = (high - low) * 0.1 or 1.0
    low, high = low - padding, high + padding

    def y(value: float) -> float:
        return bottom - (value - low) / (high - low) * (bottom - top)

    def tick(value: Any) -> str:
        return _percent(value) if percent else _number(value, 3)

    def label(x: float, ypos: float, value: str, anchor: str = "middle") -> str:
        return (
            f'<text x="{x:.2f}" y="{ypos:.2f}" text-anchor="{anchor}" '
            f'font-size="11" font-family="sans-serif">{escape(value)}</text>'
        )

    zero = y(0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{escape(title)}">',
        f"<title>{escape(title)}</title>",
        f'<rect width="{width}" height="{height}" fill="white"/>',
        label(left, 16, title, "start"),
    ]
    for value in (high, 0.0, low):
        ypos = y(value)
        parts += [
            f'<line x1="{left}" x2="{right}" y1="{ypos:.2f}" y2="{ypos:.2f}" '
            f'stroke="{"#444" if value == 0 else "#ddd"}" '
            f'aria-label="{"zero line" if value == 0 else "grid line"}"/>',
            label(left - 8, ypos + 4, tick(value), "end"),
        ]
    colors = ("#1f5fa8", "#b45309", "#047857", "#7c3aed", "#be185d", "#475569")
    step = (right - left) / max(len(labels), 1)
    bar_width = step * 0.75 / max(len(series), 1)
    for index, name in enumerate(labels):
        center = left + (index + 0.5) * step
        parts.append(label(center, bottom + 18, name))
        for number, (series_name, points) in enumerate(series):
            value = points[index] if index < len(points) else None
            x = center - step * 0.375 + number * bar_width
            if value is None:
                parts.append(label(x + bar_width / 2, zero - 4, "–"))
                continue
            ypos = y(float(value))
            parts.append(
                f'<rect x="{x:.2f}" y="{min(zero, ypos):.2f}" '
                f'width="{bar_width * 0.9:.2f}" height="{abs(zero - ypos):.2f}" '
                f'fill="{colors[number % len(colors)]}">'
                f"<title>{escape(series_name)}; {escape(name)}: {escape(str(value))}</title></rect>"
            )
    for number, (name, _) in enumerate(series):
        x, ypos = left + (number % 4) * 156, 238 + (number // 4) * 24
        parts += [
            f'<rect x="{x}" y="{ypos - 9}" width="10" height="10" fill="{colors[number % len(colors)]}"/>',
            label(x + 15, ypos, name, "start"),
        ]
    parts.append("</svg>\n")
    return "".join(parts)


def factor_ic_svg(horizons: list[dict[str, Any]]) -> str:
    """Mean rank IC by horizon, including negative, zero and unavailable values."""
    return _factor_bars_svg(
        title="Mean rank IC by horizon (sessions)",
        labels=[str(row["horizon"]) for row in horizons],
        series=[("Mean IC", [row.get("mean_ic") for row in horizons])],
        percent=False,
    )


def factor_quintiles_svg(horizons: list[dict[str, Any]]) -> str:
    """Mean outcome by quintile and horizon, without constructing a portfolio."""
    return _factor_bars_svg(
        title="Mean horizon outcome by quintile",
        labels=[f"Q{number}" for number in range(1, 6)],
        series=[(f"Horizon {row['horizon']}", row.get("quintile_returns") or []) for row in horizons],
        percent=True,
    )

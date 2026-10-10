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
    diagnostics: dict[str, Any] | None = None,
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
    if diagnostics is not None:
        lines += diagnostic_lines(diagnostics)
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


_SERIES_COLORS = ("#1f5fa8", "#c4572f", "#3f8f4f", "#8a4fb0", "#a8891f", "#4f9aa8", "#b04f7d", "#6b6b6b")


def growth_svg(
    series: list[tuple[str, list[tuple[str, Decimal]]]], *, width: int = 720, height: int = 260
) -> str:
    """Several equity curves on one chart, each as growth of 1 from its first point.

    Only curves that have one point per point of the first curve are drawn.
    """
    drawn = [
        (label, [float(value) for _, value in points])
        for label, points in series
        if len(points) >= 2 and len(points) == len(series[0][1]) and float(points[0][1]) > 0
    ]
    if not drawn:
        return ""
    curves = [(label, [value / values[0] for value in values]) for label, values in drawn]
    low = min(min(values) for _, values in curves)
    high = max(max(values) for _, values in curves)
    span = (high - low) or 1.0
    pad = 32
    count = len(curves[0][1])
    step = (width - 2 * pad) / (count - 1)
    first, last = escape(series[0][1][0][0]), escape(series[0][1][-1][0])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Growth of 1 from {first} to {last} for {len(curves)} runs">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
    ]
    for index, (label, values) in enumerate(curves):
        color = _SERIES_COLORS[index % len(_SERIES_COLORS)]
        points = " ".join(
            f"{pad + i * step:.1f},{height - pad - (v - low) / span * (height - 2 * pad):.1f}"
            for i, v in enumerate(values)
        )
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.4" points="{points}"/>')
        parts.append(
            f'<text x="{width - pad}" y="{14 + 13 * index}" font-size="11" font-family="sans-serif" '
            f'text-anchor="end" fill="{color}">{escape(label)}</text>'
        )
    parts.append(
        f'<text x="{pad}" y="{height - 8}" font-size="11" font-family="sans-serif">{first}</text>'
        f'<text x="{width - pad}" y="{height - 8}" font-size="11" font-family="sans-serif" '
        f'text-anchor="end">{last}</text>'
        f'<text x="4" y="{pad}" font-size="11" font-family="sans-serif">{high:.2f}</text>'
        f'<text x="4" y="{height - pad}" font-size="11" font-family="sans-serif">{low:.2f}</text>'
        "</svg>\n"
    )
    return "".join(parts)


def _change(values: list[Any]) -> str:
    return f"`{values[0]}` → `{values[1]}`"


def render_comparison(comparison: dict[str, Any], *, labels: dict[str, str], chart: bool) -> str:
    """``comparison.md`` for a result of ``validation.compare.compare_runs``.

    ``labels`` gives each run id the short name used in the tables.
    """
    runs = comparison["runs"]
    reference = comparison["reference"]
    grades = sorted({str(run["grade"]) for run in runs})
    lines = [
        f"# Comparison of {len(runs)} runs",
        "",
        f"> **Evidence:** grade `{'`, `'.join(grades)}`. In-sample backtests compared with each other; "
        "nothing here raises a claim level.",
    ]
    if "synthetic" in grades:
        lines.append("> Synthetic data: these numbers say nothing about real markets.")
    if not comparison["comparable"]:
        lines += [
            ">",
            f"> **Not comparable** ({', '.join(comparison['reasons'])}). The runs are listed side by side, "
            "but nothing is differenced or ranked.",
        ]
    lines += [
        "",
        "| Run | Strategy | Total return | CAGR | Annual volatility | Sharpe | Max drawdown | Fills |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        m = run["metrics"]
        name = labels.get(run["run_id"], run["run_id"])
        lines.append(
            f"| {name}{' (reference)' if run['run_id'] == reference else ''} | {run['strategy_id']} | "
            f"{_percent(m.get('total_return'))} | {_percent(m.get('cagr'))} | "
            f"{_percent(m.get('annual_volatility'))} | {_number(m.get('sharpe'))} | "
            f"{_percent(m.get('max_drawdown'))} | {m.get('fills', '–')} |"
        )
    lines.append("")
    if chart:
        lines += ["![Growth of 1](equity.svg)", ""]
    lines += [f"## Differences from the reference `{labels.get(reference, reference)}`", ""]
    for run_id, parts in comparison["differences"].items():
        changes = [f"parameter `{path}`: {_change(values)}" for path, values in parts["params"].items()]
        changes += [f"`{path}`: {_change(values)}" for path, values in parts["spec"].items()]
        lines.append(
            f"- **{labels.get(run_id, run_id)}**: " + ("; ".join(changes) or "no recorded difference")
        )
    lines.append("")
    if comparison["pairs"]:
        lines += [
            "## Against the reference",
            "",
            "| Run | Return correlation | Sharpe difference | 90% interval | Total return | Max drawdown |",
            "|---|---|---|---|---|---|",
        ]
        for run_id, pair in comparison["pairs"].items():
            interval = pair.get("sharpe_difference_90")
            lines.append(
                f"| {labels.get(run_id, run_id)} | {_number(pair.get('correlation'))} | "
                f"{'–' if pair.get('sharpe_difference') is None else format(pair['sharpe_difference'], '+.2f')} | "
                f"{'–' if interval is None else f'{interval[0]:+.2f} to {interval[1]:+.2f}'} | "
                f"{pair['total_return_difference']:+.2%} | {pair['max_drawdown_difference']:+.2%} |"
            )
        lines += [
            "",
            "Differences are the run minus the reference. The interval is a paired moving-block "
            "bootstrap of the annualized Sharpe difference; when it contains zero, this sample cannot "
            "tell the two runs apart.",
            "",
        ]
    lines += ["## Identity", ""]
    for run in runs:
        lines.append(
            f"- {labels.get(run['run_id'], run['run_id'])}: run `{run['run_id']}`, configuration "
            f"`{run['configuration_hash']}`, dataset `{run['dataset_identity']}`"
        )
    lines += ["", "---", "", f"*{DISCLAIMER}*", ""]
    return "\n".join(lines)


_VERDICT_WORDING = {
    "supported": "Supported",
    "not_supported": "Not supported",
    "insufficient": "Insufficient evidence",
}
_PERCENT_METRICS = ("total_return", "cagr", "max_drawdown", "annual_volatility")


def _metric_text(metric: str, value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "–"
    if metric in _PERCENT_METRICS:
        return f"{float(value):+.2%}" if signed else f"{float(value):.2%}"
    return f"{float(value):+.2f}" if signed else f"{float(value):.2f}"


def render_study(result: dict[str, Any], *, chart: bool) -> str:
    """``comparison.md`` for one run of a study (``signalquarry.study-result/v1``)."""
    study, outcome, comparison = result["study"], result["verdict"], result["comparison"]
    rule = study["compare"]
    grade = result["evidence"]["grade"]
    lines = [
        f"# Study {result['study_id']}",
        "",
        f"> **Evidence:** grade `{grade}`, claim level `{result['evidence']['claim_level']}`. A study compares "
        "in-sample backtests by a rule declared beforehand. It does not raise a claim level.",
    ]
    if grade == "synthetic":
        lines.append("> Synthetic data: these numbers say nothing about real markets.")
    lines += [
        "",
        "## Hypothesis",
        "",
        study["hypothesis"]["statement"],
        "",
        f"*Falsified if:* {study['hypothesis']['falsification']}",
        "",
        "## Verdict",
        "",
        f"**{_VERDICT_WORDING.get(outcome['outcome'], outcome['outcome'])}.** Declared rule: `{rule['metric']}` "
        f"of the subject `{result['base']}` is {rule['direction']} than that of `{rule['versus']}`.",
        "",
        f"- Subject {_metric_text(rule['metric'], outcome.get('subject'))}, baseline "
        f"{_metric_text(rule['metric'], outcome.get('baseline'))}, difference "
        f"{_metric_text(rule['metric'], outcome.get('difference'), signed=True)}.",
    ]
    interval = outcome.get("interval_90")
    lines.append(
        "- Paired 90% interval of the difference: "
        + (
            "not available."
            if interval is None
            else f"{_metric_text(rule['metric'], interval[0], signed=True)} to "
            f"{_metric_text(rule['metric'], interval[1], signed=True)}."
        )
    )
    lines.append(f"- Why: {outcome['reason']}.")
    if not comparison["comparable"]:
        lines.append(f"- The arms are not all comparable ({', '.join(comparison['reasons'])}).")
    lines += [
        "",
        f"## Arms over {result['sessions']} sessions",
        "",
        "| Arm | Role | Trial | Total return | CAGR | Annual volatility | Sharpe | Max drawdown |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in result["arms"]:
        m = arm["metrics"]
        lines.append(
            f"| {arm['id']} | {arm['role']} | {'yes' if arm['counts'] else 'no'} | "
            f"{_percent(m.get('total_return'))} | {_percent(m.get('cagr'))} | "
            f"{_percent(m.get('annual_volatility'))} | {_number(m.get('sharpe'))} | "
            f"{_percent(m.get('max_drawdown'))} |"
        )
    lines.append("")
    if chart:
        lines += ["![Growth of 1](equity.svg)", ""]
    changed = [arm for arm in result["arms"] if "changes" in arm]
    if changed:
        lines += ["## What each variant changes", ""]
        for arm in changed:
            parts = [
                f"parameter `{path}`: {_change(values)}" for path, values in arm["changes"]["params"].items()
            ]
            parts += [f"`{path}`: {_change(values)}" for path, values in arm["changes"]["spec"].items()]
            note = f" ({arm['note']})" if arm.get("note") else ""
            lines.append(
                f"- **{arm['id']}**, {arm['role']}{note}: " + ("; ".join(parts) or "no recorded difference")
            )
        lines.append("")
    if comparison["pairs"]:
        lines += [
            f"## Against `{rule['versus']}`",
            "",
            "| Arm | Return correlation | Sharpe difference | 90% interval | Total return | Max drawdown |",
            "|---|---|---|---|---|---|",
        ]
        for arm_id, pair in comparison["pairs"].items():
            band = pair.get("sharpe_difference_90")
            lines.append(
                f"| {arm_id} | {_number(pair.get('correlation'))} | "
                f"{'–' if pair.get('sharpe_difference') is None else format(pair['sharpe_difference'], '+.2f')} | "
                f"{'–' if band is None else f'{band[0]:+.2f} to {band[1]:+.2f}'} | "
                f"{pair['total_return_difference']:+.2%} | {pair['max_drawdown_difference']:+.2%} |"
            )
        lines += [
            "",
            "Differences are the arm minus the baseline. Variants are shown beside the subject, not as a "
            "ranking to pick from: choosing the best row afterwards is what the trial count prices.",
            "",
        ]
    pbo = result.get("pbo")
    if pbo and pbo.get("splits"):
        lines += [
            "## Overfitting",
            "",
            f"Probability of backtest overfitting across the {pbo['configurations']} candidate "
            f"configurations: {pbo['pbo']:.2f} over {pbo['splits']} splits. Near 0.5, the best of them is "
            "mostly luck.",
            "",
        ]
    if result.get("trials"):
        lines += ["## Trials", ""]
        for family, row in result["trials"].items():
            lines.append(
                f"- Family `{family}`: this study added {row['new']} new configuration(s); "
                f"{row['used']} of {row['budget']} were already used before it."
            )
        lines.append("")
    lines += [
        "## Identity",
        "",
        f"- Study: `{result['study_hash']}`",
        f"- Dataset: `{result['dataset_identity']}` (`{result['dataset_id']}`)",
        f"- Window: {result['window']['start'] or 'first session'} to {result['window']['end'] or 'last session'}"
        + (" (clipped at the sealed holdout)" if result.get("holdout_clipped") else ""),
    ]
    for arm in result["arms"]:
        if arm.get("run_id"):
            lines.append(f"- {arm['id']}: run `{arm['run_id']}`, configuration `{arm['configuration_hash']}`")
    lines += ["", "---", "", f"*{DISCLAIMER}*", ""]
    return "\n".join(lines)


def diagnostic_lines(document: dict[str, Any]) -> list[str]:
    """Render recorded diagnostics; do no calculations or simulations here."""
    lines: list[str] = []
    unavailable: list[str] = []
    regimes = document["regimes"]
    if regimes["status"] == "ok":
        lines += ["## Regimes", "", "Descriptive only. " + regimes["note"], ""]
        for kind in ("calendar", "trend", "volatility"):
            block = regimes[kind]
            if block["status"] != "ok":
                unavailable.append(f"{kind}: {block['reason']}")
                continue
            lines += [
                f"### {kind.title()}",
                "",
                "| Regime | Sessions | Share | Status | Return | Sharpe | Max drawdown | Benchmark return | Excess |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for row in block["rows"]:
                lines.append(
                    f"| {row['regime']} | {row['sessions']} | {_percent(row['share'])} | {row['status']} | "
                    f"{_percent(row['total_return'])} | {_number(row['sharpe'])} | "
                    f"{_percent(row['max_drawdown'])} | {_percent(row['benchmark_total_return'])} | "
                    f"{_percent(row['excess_return'])} |"
                )
            lines.append("")
    else:
        unavailable.append(f"regimes: {regimes['reason']}")
    costs = document["costs"]
    if costs["status"] == "ok":
        lines += [
            "## Cost sensitivity",
            "",
            costs["note"],
            "",
            "| Cost multiplier | Return | Sharpe | Max drawdown |",
            "|---|---|---|---|",
        ]
        for point in costs["points"]:
            lines.append(
                f"| {point['multiplier']}× | {_percent(point['total_return'])} | "
                f"{_number(point['sharpe'])} | {_percent(point['max_drawdown'])} |"
            )
        lines += [
            "",
            f"Break-even at {_number(costs['break_even'], 2)}× costs (linear interpolation)."
            if costs["break_even"] is not None
            else "The return is not positive even at zero cost, so there is no break-even."
            if (costs["points"][0].get("total_return") or 0) <= 0
            else "No break-even within 4× costs: the return stays positive.",
            "",
        ]
    else:
        unavailable.append(f"costs: {costs['reason']}")
    folds = document["folds"]
    if folds["status"] == "ok":
        summary = folds["consistency"]
        lines += [
            "## Fold consistency",
            "",
            "Descriptive only; this does not change any evaluation gate or claim.",
            "",
            f"{summary['positive']} of {summary['folds']} folds positive "
            f"({_percent(summary['share_positive'])}); ahead of benchmark in "
            f"{summary['ahead_of_benchmark'] if summary['ahead_of_benchmark'] is not None else 'unavailable'} folds. "
            f"Best return {_percent(summary['best'])}, worst {_percent(summary['worst'])}, "
            f"sample dispersion {_percent(summary['dispersion'])}. Status: {summary['status']}.",
            "",
        ]
    else:
        unavailable.append(f"folds: {folds['reason']}")
    parameters = document["parameters"]
    if parameters["status"] == "ok":
        best = parameters["best"]
        lines += [
            "## Parameter sensitivity",
            "",
            "Descriptive only; these recorded sweep results do not change a claim.",
            "",
            f"Best point: `{best['params'] if best else 'unavailable'}` "
            f"({_number(best['value'] if best else None)} {parameters['metric']}); "
            f"neighbour median {_number(parameters['neighbour_median'])}; plateau {_number(parameters['plateau'], 6)}.",
            "",
        ]
    else:
        unavailable.append(f"parameters: {parameters['reason']}")
    exposure = document.get("exposure")
    if exposure is not None:
        if exposure["status"] == "ok":
            regression = exposure["regression"]
            lines += [
                "## Exposure to reference series",
                "",
                "| Reference | Beta | t | Annual contribution |",
                "|---|---|---|---|",
            ]
            for reference in regression["references"]:
                lines.append(
                    f"| {reference['name']} | {_number(reference.get('beta'), 6)} | "
                    f"{_number(reference.get('t'))} | {_percent(reference.get('contribution_annual'))} |"
                )
            lines += [
                "",
                f"Annual alpha {_percent(regression.get('alpha_annual'))} "
                f"(t {_number(regression.get('alpha_t'))}); R² {_number(regression.get('r_squared'), 6)}; "
                f"residual volatility {_percent(regression.get('residual_volatility'))}.",
                "",
            ]
            rolling = exposure.get("rolling", {}).get("rows", [])
            for reference in regression["references"]:
                name = reference["name"]
                betas = [
                    row["betas"][name]
                    for row in rolling
                    if row["status"] == "ok" and row.get("betas", {}).get(name) is not None
                ]
                lines.append(
                    f"Rolling beta {name}: {_number(min(betas) if betas else None, 6)} "
                    f"to {_number(max(betas) if betas else None, 6)}."
                )
            lines += ["", exposure.get("note") or "–", ""]
        else:
            reason = exposure.get("reason") or {
                "insufficient": "Too few sessions.",
                "collinear": "The references move together too closely to separate.",
            }.get(exposure["status"], "–")
            unavailable.append(f"exposure: {reason}")
    if unavailable:
        lines += ["Unavailable diagnostics: " + " ".join(unavailable), ""]
    return lines

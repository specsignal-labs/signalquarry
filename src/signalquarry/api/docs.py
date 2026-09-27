# SPDX-License-Identifier: Apache-2.0
"""Documentation for people and agents, generated from packaged guides and live catalogs.

``sqy docs --llms`` returns ``llms.txt`` (an index) and ``--full`` returns
``llms-full.txt`` (every guide, the CLI reference and all reason codes). The
docs site is generated from the same functions (``tools/gen_docs.py``), so the
offline text and the site never drift.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

from signalquarry import __version__
from signalquarry._internal.contracts.reason_codes import REASON_CODES
from signalquarry.api.envelope import Envelope

GUIDES: tuple[tuple[str, str], ...] = (
    ("index", "What SignalQuarry is and why its results can be trusted."),
    ("quickstart", "Install, create a demo project and run the golden path."),
    ("authoring", "Write decide(ctx, params) and strategy.yaml; rules sqy check enforces."),
    ("evidence", "Claim ladder, trial ledger, freeze and sealed holdout, gates G1–G4, reports."),
    ("data", "Alpaca credentials, fetching and verifying data, point-in-time adjustment."),
    ("paper", "Paper deployments, the arm step, run-once, journal, guards and scheduling."),
    ("agents", "The JSON envelope, exit codes and human-only actions for coding agents."),
)


def guide(name: str) -> str:
    return (resources.files("signalquarry") / "guides" / f"{name}.md").read_text(encoding="utf-8")


def cli_markdown(catalog: list[dict[str, Any]]) -> str:
    lines = [
        "# CLI reference",
        "",
        "Generated from the command registry. Every command accepts `--json`.",
        "",
    ]

    def options(item: dict[str, Any]) -> list[str]:
        rows: list[str] = []
        for option in item["options"]:
            flags = ", ".join(f"`{flag}`" for flag in option["flags"])
            extra: list[str] = []
            if option["required"]:
                extra.append("required")
            if option["choices"]:
                extra.append("one of " + ", ".join(f"`{c}`" for c in option["choices"]))
            detail = option["help"] + (f" ({'; '.join(extra)})" if extra else "")
            rows.append(f"| {flags} | {detail.strip()} |")
        return ["| Option | Meaning |", "|---|---|", *rows, ""] if rows else []

    for command in catalog:
        lines += [f"## `sqy {command['name']}`", "", command["help"], ""]
        lines += options(command)
        for sub in command["subcommands"]:
            lines += [f"### `sqy {command['name']} {sub['name']}`", "", sub["help"], ""]
            lines += options(sub)
    return "\n".join(lines).rstrip() + "\n"


def reason_codes_markdown() -> str:
    lines = [
        "# Reason codes",
        "",
        "Codes are append-only: never renamed or removed. `sqy explain <CODE>` prints one.",
        "",
        "| Code | Category | Meaning | What to do |",
        "|---|---|---|---|",
    ]
    for code, item in sorted(REASON_CODES.items()):
        lines.append(f"| `{code}` | {item.category.value} | {item.description} | {item.remedy} |")
    return "\n".join(lines) + "\n"


def llms_index(catalog: list[dict[str, Any]]) -> str:
    lines = [
        "# SignalQuarry",
        "",
        "> Agent-first strategy research: honest backtests on Alpaca data and Alpaca paper forward "
        "tests. The framework owns time, data, fills and the evidence record; strategies are one pure "
        "`decide(ctx, params)` function plus `strategy.yaml`. No live-trading path exists.",
        "",
        f"Framework version {__version__}. Run `sqy docs --llms --full` for the complete text offline.",
        "",
        "## Guides",
        "",
        *[f"- [{name}]({name}.md): {summary}" for name, summary in GUIDES],
        "",
        "## Reference",
        "",
        "- [CLI reference](reference/cli.md): every command and option.",
        "- [Reason codes](reference/reason-codes.md): every status code and its remedy.",
        "",
        "## Commands",
        "",
        *[f"- `sqy {c['name']}`: {c['help']}" for c in catalog],
        "",
    ]
    return "\n".join(lines)


def llms_full(catalog: list[dict[str, Any]]) -> str:
    parts = [llms_index(catalog)]
    parts += [guide(name) for name, _ in GUIDES]
    parts += [cli_markdown(catalog), reason_codes_markdown()]
    return "\n\n".join(part.rstrip() for part in parts) + "\n"


def docs(catalog: list[dict[str, Any]], *, full: bool) -> Envelope:
    text = llms_full(catalog) if full else llms_index(catalog)
    return Envelope(
        command="docs",
        summary=f"{'llms-full.txt' if full else 'llms.txt'}: {len(text.encode('utf-8'))} bytes in data.text",
        data={"format": "llms-full.txt" if full else "llms.txt", "text": text},
    )

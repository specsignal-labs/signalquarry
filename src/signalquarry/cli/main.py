# SPDX-License-Identifier: Apache-2.0
"""Argument parsing and output. All logic lives in :mod:`signalquarry.api`."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, NoReturn

from signalquarry import api
from signalquarry._internal.contracts import progress
from signalquarry._internal.data.credentials import redact
from signalquarry.api.envelope import Envelope, cap_data


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:  # argparse would exit 2; usage errors are 64
        raise _UsageError(message)


def _absolute_command(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("--notify-command requires an absolute executable path")
    return path


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    handler: Callable[[argparse.Namespace], Envelope]
    configure: Callable[[argparse.ArgumentParser], object] = lambda parser: None


def _commands_envelope(_: argparse.Namespace) -> Envelope:
    return Envelope(
        command="commands",
        summary=f"{len(COMMANDS)} commands",
        data={
            "commands": command_catalog(),
            "reason_codes": api.reason_code_catalog(),
            "exit_codes": {
                "ok": 0,
                "error": 1,
                "blocked": 2,
                "usage": 64,
                "invalid": 65,
                "unavailable": 69,
                "busy": 75,
                "disabled": 78,
            },
        },
    )


def _configure_init(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", nargs="?", help="new project directory")
    parser.add_argument(
        "--upgrade-agents-md",
        action="store_true",
        help="refresh the framework-owned blocks of an existing project's AGENTS.md",
    )
    parser.add_argument("--project", type=Path, help="project directory for --upgrade-agents-md")
    parser.add_argument("--demo", action="store_true", help="use synthetic data (no credentials needed)")
    parser.add_argument("--lab", action="store_true", help="a multi-family strategy lab (per-family ledgers)")
    parser.add_argument(
        "--kind",
        choices=["equity", "options"],
        default="equity",
        help="starter strategy: equity trend or options wheel",
    )
    parser.add_argument(
        "--package", help="Python package name for strategies (default: from the directory name)"
    )


def _usage(command: str, summary: str) -> Envelope:
    return Envelope(command=command, status="usage", reason_codes=["USAGE_INVALID"], summary=summary)


def _init(args: argparse.Namespace) -> Envelope:
    if args.upgrade_agents_md:
        if args.path:
            return _usage("init", "--upgrade-agents-md takes --project, not a path")
        return api.upgrade_agents_md(project=args.project)
    if not args.path:
        return _usage("init", "a new project directory is required")
    return api.init(Path(args.path), demo=args.demo, package=args.package, lab=args.lab, kind=args.kind)


def _configure_check(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strategy", help="check only this strategy id")
    parser.add_argument("--factor", help="check one project module defining a @factor function")
    parser.add_argument("--factor-id", help="check a factor id registered in signalquarry.toml")
    parser.add_argument("--params-json", default="{}", help="JSON object of factor parameters")
    parser.add_argument(
        "--parity",
        action="store_true",
        help="also replay 60 sessions through the paper kernel and a fake venue; must match the backtest",
    )
    parser.add_argument("--project", type=Path, help="project directory (default: search upwards from cwd)")


def _check(args: argparse.Namespace) -> Envelope:
    if args.factor_id:
        if args.factor or args.strategy or args.parity or args.params_json != "{}":
            return _usage("check", "--factor-id cannot be combined with other check selectors")
        return api.check_registered_factor(args.factor_id, project=args.project)
    if args.factor:
        if args.strategy or args.parity:
            return _usage("check", "--factor cannot be combined with --strategy or --parity")
        return api.check_factor(args.factor, project=args.project, params_json=args.params_json)
    if args.params_json != "{}":
        return _usage("check", "--params-json requires --factor")
    return api.check(args.strategy, project=args.project, parity=args.parity)


def _configure_factor(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="action", required=True, parser_class=_Parser)
    listing = actions.add_parser("ls", help="List explicitly registered project factors and hashes.")
    listing.add_argument("--project", type=Path, help="project directory (default: search upwards)")
    listing.add_argument("--json", action="store_true", default=argparse.SUPPRESS)


def _factor(args: argparse.Namespace) -> Envelope:
    return api.factor_ls(project=args.project)


def _configure_backtest(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strategy", required=True, help="strategy id from strategy.yaml")
    parser.add_argument("--start", type=date.fromisoformat, help="first session (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, help="last session (YYYY-MM-DD)")
    parser.add_argument("--project", type=Path, help="project directory (default: search upwards from cwd)")


def _configure_report(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--run", help="backtest run id (default: the latest)")
    parser.add_argument("--project", type=Path)


def _configure_data(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="action", required=True, parser_class=_Parser)
    fetch = actions.add_parser(
        "fetch", help="Download daily bars and corporate actions and record a manifest."
    )
    fetch.add_argument("--strategy", help="fetch the symbols (and benchmark) of this strategy")
    fetch.add_argument("--symbols", default="", help="comma-separated extra symbols")
    fetch.add_argument("--start", type=date.fromisoformat, help="first date (default 2016-01-01)")
    fetch.add_argument("--end", type=date.fromisoformat, help="last date (default yesterday)")
    fetch.add_argument("--feed", choices=("sip", "iex"), help="default: the strategy's feed, else sip")
    probe = actions.add_parser(
        "probe", help="Check provider coverage: expired option contracts and their bars."
    )
    probe.add_argument("what", choices=["options-coverage"], help="what to probe")
    probe.add_argument("--underlying", default="QQQ", help="underlying symbol")
    probe.add_argument("--month", required=True, help="expiration month to probe, YYYY-MM")
    record = actions.add_parser(
        "record", help="Record today's option chain for an underlying (raw pages cached, hash-only record)."
    )
    record.add_argument("what", choices=["options"], help="what to record")
    record.add_argument("--underlying", required=True, help="underlying symbol, e.g. QQQ")
    record.add_argument("--max-dte", type=int, default=60, help="latest expiration, in days from today")
    record.add_argument("--width", type=float, default=0.2, help="strike window around spot, as a fraction")
    for name in ("verify", "ls"):
        actions.add_parser(
            name,
            help={"verify": "Re-hash cached pages against every manifest.", "ls": "List recorded datasets."}[
                name
            ],
        )
    for child in actions.choices.values():
        child.add_argument(
            "--project", type=Path, help="project directory (default: search upwards from cwd)"
        )
        child.add_argument("--json", action="store_true", default=argparse.SUPPRESS)


def _data(args: argparse.Namespace) -> Envelope:
    if args.action == "fetch":
        symbols = tuple(item.strip().upper() for item in args.symbols.split(",") if item.strip())
        return api.data_fetch(
            strategy_id=args.strategy,
            symbols=symbols,
            start=args.start,
            end=args.end,
            feed=args.feed,
            project=args.project,
        )
    if args.action == "record":
        return api.data_record_options(
            args.underlying, max_dte=args.max_dte, width=args.width, project=args.project
        )
    if args.action == "probe":
        return api.data_probe_options(args.underlying, args.month)
    if args.action == "verify":
        return api.data_verify(project=args.project)
    return api.data_ls(project=args.project)


def _configure_universe(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="action", required=True, parser_class=_Parser)
    actions.add_parser("snapshot", help="Capture the full current Alpaca US-equity asset list.")
    actions.add_parser("verify", help="Verify hashed asset snapshots against cached raw pages.")
    as_of = actions.add_parser("as-of", help="Find a verified snapshot known by a UTC cutoff.")
    as_of.add_argument(
        "--known-at",
        type=datetime.fromisoformat,
        required=True,
        help="timezone-aware decision cutoff (ISO 8601); older than 31 days is unavailable",
    )
    for child in actions.choices.values():
        child.add_argument(
            "--project", type=Path, help="project directory (default: search upwards from cwd)"
        )
        child.add_argument("--json", action="store_true", default=argparse.SUPPRESS)


def _universe(args: argparse.Namespace) -> Envelope:
    if args.action == "snapshot":
        return api.universe_snapshot(project=args.project)
    if args.action == "verify":
        return api.universe_verify(project=args.project)
    return api.universe_as_of(known_at=args.known_at, project=args.project)


def _configure_spec(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="action", required=True, parser_class=_Parser)
    freeze = actions.add_parser("freeze", help="Freeze the configuration and seal the family's holdout.")
    freeze.add_argument("--strategy", required=True)
    freeze.add_argument("--project", type=Path)
    freeze.add_argument("--json", action="store_true", default=argparse.SUPPRESS)


def _configure_evaluate(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--no-stress", action="store_true", help="skip the cost and delay stress reruns")
    parser.add_argument(
        "--holdout", action="store_true", help="open the sealed holdout (once per family; needs G1-G3)"
    )
    parser.add_argument("--project", type=Path)


def _configure_trials(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="action", required=True, parser_class=_Parser)
    listing = actions.add_parser("ls", help="List recorded trials.")
    listing.add_argument("--family")
    show = actions.add_parser("show", help="Show entries by sequence number or configuration-hash prefix.")
    show.add_argument("key", help="sequence number or configuration-hash prefix")
    extend = actions.add_parser("extend", help="HUMAN ONLY: add 10 trials to a family's budget.")
    extend.add_argument("--family", required=True)
    extend.add_argument("--reason", required=True)
    for child in actions.choices.values():
        child.add_argument("--project", type=Path)
        child.add_argument("--json", action="store_true", default=argparse.SUPPRESS)


def _configure_holdout(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="action", required=True, parser_class=_Parser)
    status = actions.add_parser("status", help="Show each family's holdout seal and opening.")
    seal = actions.add_parser("seal", help="Seal the strategy family's holdout now, before exploring.")
    seal.add_argument("--strategy", required=True)
    for child in (status, seal):
        child.add_argument("--project", type=Path)
        child.add_argument("--json", action="store_true", default=argparse.SUPPRESS)


_PAPER_ACTIONS = {
    "preflight": "Read-only readiness checks for a deployment.",
    "dry-run": "Decide and size the next session without submitting anything.",
    "arm": "HUMAN ONLY: allow a deployment to submit paper orders (interactive confirmation).",
    "run-once": "Reconcile, decide and submit the current session's orders (idempotent).",
    "status": "Show the journal state: armed/halted, sessions, pending orders.",
    "reconcile": "Finalize journaled orders and check positions against the broker.",
    "halt": "Stop a deployment and cancel its open orders; a human re-arms.",
    "drift": "Replay sessions against the engine, measure fill slippage, apply model costs; gate G5.",
    "run": "Run run-once until done, retrying while busy or unavailable (for a service manager).",
    "backup": "Write a tar.gz of the journal and arm token with a hash manifest.",
    "verify-continuity": "Check the live journal extends a backup (no rollback on the host).",
    "schedule": "Write scheduler templates (systemd, launchd, cron, github-actions) for review.",
}


def _configure_paper(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="action", required=True, parser_class=_Parser)
    for name, text in _PAPER_ACTIONS.items():
        child = actions.add_parser(name, help=text, description=text)
        child.add_argument(
            "--alias", required=True, help="deployment alias (the ALIAS in paper/ALIAS.paper.yaml)"
        )
        child.add_argument("--project", type=Path)
        child.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
        if name in ("arm", "halt"):
            child.add_argument("--reason", required=True, help="recorded in the journal")
        if name == "run":
            child.add_argument("--interval", type=float, default=60.0, help="seconds between retries")
            child.add_argument("--max-minutes", type=float, default=30.0, help="give up after this long")
        if name == "backup":
            child.add_argument("--out", type=Path, help="directory (default: .signalquarry/backups/ALIAS)")
        if name == "verify-continuity":
            child.add_argument("--backup", type=Path, required=True, help="a tar.gz written by paper backup")
        if name == "schedule":
            child.add_argument(
                "--target",
                required=True,
                choices=["systemd", "launchd", "cron", "github-actions"],
                help="scheduler to write templates for (github-actions is demo-only)",
            )
        if name in ("run-once", "schedule"):
            child.add_argument(
                "--notify-command",
                type=_absolute_command,
                help="absolute executable path; called on a non-zero run-once exit with no credential environment",
            )


def _paper(args: argparse.Namespace) -> Envelope:
    alias, project = args.alias, args.project
    if args.action == "arm":
        return api.paper_arm(alias, args.reason, project=project, confirm=input)
    if args.action == "halt":
        return api.paper_halt(alias, args.reason, project=project)
    if args.action == "schedule":
        return api.paper_schedule(alias, args.target, project=project, notify_command=args.notify_command)
    if args.action == "run":
        return api.paper_run(alias, project=project, interval=args.interval, max_minutes=args.max_minutes)
    if args.action == "backup":
        return api.paper_backup(alias, out=args.out, project=project)
    if args.action == "verify-continuity":
        return api.paper_verify_continuity(alias, args.backup, project=project)
    handlers = {
        "preflight": api.paper_preflight,
        "dry-run": api.paper_dry_run,
        "run-once": api.paper_run_once,
        "status": api.paper_status,
        "reconcile": api.paper_reconcile,
        "drift": api.paper_drift,
    }
    return handlers[args.action](alias, project=project)


def _configure_evidence(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="action", required=True, parser_class=_Parser)
    verify = actions.add_parser(
        "verify", help="Verify evidence ledgers and paper journals, or an exported bundle (--bundle)."
    )
    verify.add_argument("--base", help="git ref; fail if any log changed other than by appending since it")
    verify.add_argument("--bundle", type=Path, help="verify this exported evidence bundle directory instead")
    verify.add_argument(
        "--max-tier", choices=["public", "nda"], default="public", help="highest tier the bundle may carry"
    )
    export = actions.add_parser(
        "export", help="Export a family's evidence bundle under its publication policy."
    )
    export.add_argument("--family", required=True, help="strategy family")
    export.add_argument(
        "--tier", choices=["public", "nda"], default="public", help="public (showcase) or nda (data room)"
    )
    export.add_argument("--out", type=Path, help="output directory (default: .signalquarry/exports/...)")
    for child in (verify, export):
        child.add_argument("--project", type=Path)
        child.add_argument("--json", action="store_true", default=argparse.SUPPRESS)


def _evidence(args: argparse.Namespace) -> Envelope:
    if args.action == "export":
        return api.evidence_export(args.family, tier=args.tier, out=args.out, project=args.project)
    if args.bundle is not None:
        return api.bundle_verify(args.bundle, max_tier=args.max_tier)
    return api.evidence_verify(base=args.base, project=args.project)


def _configure_commit(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="action", required=True, parser_class=_Parser)
    create = actions.add_parser(
        "create", help="Commit to a frozen strategy (salted) or to a paper journal head."
    )
    create.add_argument("--strategy", required=True)
    create.add_argument("--alias", help="commit to this paper deployment's journal head instead")
    reveal = actions.add_parser("reveal", help="Write the private opening of a spec commitment (NDA only).")
    reveal.add_argument("--id", required=True, help="commitment id")
    reveal.add_argument("--out", type=Path, required=True, help="where to write the opening (mode 0600)")
    verify = actions.add_parser("verify", help="Check an opening against a commitment record.")
    verify.add_argument("--record", type=Path, required=True, help="commitment record JSON")
    verify.add_argument("--reveal", type=Path, required=True, help="opening JSON from `sqy commit reveal`")
    for child in (create, reveal, verify):
        child.add_argument("--project", type=Path)
        child.add_argument("--json", action="store_true", default=argparse.SUPPRESS)


def _commit(args: argparse.Namespace) -> Envelope:
    if args.action == "create":
        return api.commit_create(args.strategy, alias=args.alias, project=args.project)
    if args.action == "reveal":
        return api.commit_reveal(args.id, out=args.out, project=args.project)
    return api.commit_verify(args.record, args.reveal)


def _configure_perf(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="action", required=True, parser_class=_Parser)
    publish = actions.add_parser(
        "publish", help="Capture and write ALIAS/latest.json (signalquarry-public-performance/v1)."
    )
    publish.add_argument("--alias", required=True, help="paper deployment alias")
    publish.add_argument(
        "--out", type=Path, required=True, help="feed directory, e.g. site/public/feeds/paper"
    )
    publish.add_argument("--project", type=Path)
    publish.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    capture = actions.add_parser(
        "capture", help="Record a private, hash-chained snapshot under paper/ALIAS/captures/."
    )
    capture.add_argument("--alias", required=True, help="paper deployment alias")
    capture.add_argument("--project", type=Path)
    capture.add_argument("--json", action="store_true", default=argparse.SUPPRESS)


def _trials(args: argparse.Namespace) -> Envelope:
    if args.action == "extend":
        return api.trials_extend(args.family, args.reason, project=args.project)
    if args.action == "show":
        return api.trials_show(args.key, project=args.project)
    return api.trials_ls(family=args.family, project=args.project)


COMMANDS: tuple[Command, ...] = (
    Command("version", "Show framework, Python and platform versions.", lambda _: api.version()),
    Command("doctor", "Check the environment, dependencies, cache and credentials.", lambda _: api.doctor()),
    Command("commands", "List commands, reason codes and exit codes.", _commands_envelope),
    Command(
        "schema",
        "List bundled JSON schemas, or print one.",
        lambda args: api.schema(args.name),
        lambda parser: parser.add_argument("name", nargs="?"),
    ),
    Command(
        "explain",
        "Explain a reason code and how to resolve it.",
        lambda args: api.explain(args.code),
        lambda parser: parser.add_argument("code"),
    ),
    Command(
        "init",
        "Create a new strategy project.",
        lambda args: _init(args),
        _configure_init,
    ),
    Command(
        "check",
        "Run conformance, determinism and look-ahead checks.",
        _check,
        _configure_check,
    ),
    Command("factor", "Inspect project-registered research factors.", _factor, _configure_factor),
    Command("data", "Fetch, verify and list recorded market data.", _data, _configure_data),
    Command(
        "universe", "Capture and verify prospective asset-list snapshots.", _universe, _configure_universe
    ),
    Command(
        "spec",
        "Freeze a strategy configuration (spec freeze).",
        lambda args: api.spec_freeze(args.strategy, project=args.project),
        _configure_spec,
    ),
    Command(
        "evaluate",
        "Walk-forward, stress and holdout gates; sets the claim level.",
        lambda args: api.evaluate_command(
            args.strategy, stress=not args.no_stress, open_holdout=args.holdout, project=args.project
        ),
        _configure_evaluate,
    ),
    Command("trials", "List trials or (humans only) extend a family's budget.", _trials, _configure_trials),
    Command(
        "holdout",
        "Seal a family's holdout early, or show seals and openings.",
        lambda args: (
            api.holdout_seal(args.strategy, project=args.project)
            if args.action == "seal"
            else api.holdout_status(project=args.project)
        ),
        _configure_holdout,
    ),
    Command(
        "sweep",
        "Backtest a parameter grid (each point on real data is a trial); reports PBO.",
        lambda args: api.sweep(args.strategy, args.param, project=args.project),
        lambda parser: (
            parser.add_argument("--strategy", required=True),
            parser.add_argument("--param", action="append", required=True, help="NAME=V1,V2,… (repeatable)"),
            parser.add_argument("--project", type=Path),
        ),
    ),
    Command(
        "backtest",
        "Backtest a strategy and write run artifacts.",
        lambda args: api.backtest(args.strategy, start=args.start, end=args.end, project=args.project),
        _configure_backtest,
    ),
    Command(
        "evidence",
        "Verify evidence ledgers, journals and bundles; export a family's evidence bundle.",
        _evidence,
        _configure_evidence,
    ),
    Command(
        "commit",
        "Salted commitments to frozen strategies and journal heads (OpenTimestamps).",
        _commit,
        _configure_commit,
    ),
    Command(
        "perf",
        "Paper performance snapshots: private captures, or a public feed (live_feed: allow).",
        lambda args: (
            api.perf_capture(args.alias, project=args.project)
            if args.action == "capture"
            else api.perf_publish(args.alias, out=args.out, project=args.project)
        ),
        _configure_perf,
    ),
    Command(
        "docs",
        "Print llms.txt (or --full: every guide, the CLI reference and reason codes) for agents.",
        lambda args: api.docs(command_catalog(), full=args.full),
        lambda parser: (
            parser.add_argument("--llms", action="store_true", help="llms.txt format (the default)"),
            parser.add_argument("--full", action="store_true", help="llms-full.txt: all documentation"),
        ),
    ),
    Command(
        "report",
        "Render the latest backtest and evaluation as report.md and equity.svg.",
        lambda args: api.report(args.strategy, run_id=args.run, project=args.project),
        _configure_report,
    ),
    Command(
        "paper",
        "Alpaca paper forward tests: preflight, dry-run, run-once, status, halt.",
        _paper,
        _configure_paper,
    ),
)


_DEFAULT_HELP = {
    "project": "project directory (default: search upwards from the current directory)",
    "strategy": "strategy id from strategy.yaml",
    "name": "schema name (omit to list them)",
    "code": "reason code, e.g. HOLDOUT_REUSED",
    "family": "strategy family",
    "reason": "why; recorded in the evidence log",
}


def _options(parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    rows = []
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)) or action.dest == "json":
            continue
        rows.append(
            {
                "flags": list(action.option_strings) or [action.dest],
                "required": bool(action.required) and bool(action.option_strings),
                "choices": list(action.choices) if action.choices else [],
                "help": action.help or _DEFAULT_HELP.get(action.dest, ""),
            }
        )
    return rows


def command_catalog() -> list[dict[str, Any]]:
    """Every command, subcommand and option, for `sqy commands` and the generated docs."""
    parser = build_parser()
    commands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    catalog = []
    for command in COMMANDS:
        child = commands.choices[command.name]
        subparsers = next((a for a in child._actions if isinstance(a, argparse._SubParsersAction)), None)
        helps = {c.dest: c.help for c in subparsers._choices_actions} if subparsers else {}
        catalog.append(
            {
                "name": command.name,
                "help": command.help,
                "options": _options(child),
                "subcommands": [
                    {"name": name, "help": helps.get(name) or "", "options": _options(sub)}
                    for name, sub in (subparsers.choices.items() if subparsers else [])
                ],
            }
        )
    return catalog


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="sqy", description="SignalQuarry: honest strategy research and paper forward tests."
    )
    parser.add_argument("--json", action="store_true", help="always print the JSON envelope")
    parser.add_argument(
        "--detail",
        choices=["summary", "full"],
        help="full: keep data over 64 KB in the envelope instead of moving it to an artifact",
    )
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)
    for command in COMMANDS:
        child = sub.add_parser(command.name, help=command.help, description=command.help)
        child.add_argument(
            "--json", action="store_true", default=argparse.SUPPRESS, help="always print the JSON envelope"
        )
        command.configure(child)
        child.set_defaults(_handler=command.handler)
    return parser


def _render_text(payload: dict[str, Any]) -> str:
    lines = [f"{payload['command']}: {payload['status']} — {payload['summary']}".rstrip(" —")]
    lines += [f"  reason: {code}" for code in payload["reason_codes"]]
    lines += [f"  warning: {code}" for code in payload["warnings"]]
    lines += [f"  next: {item['command']}  ({item['why']})" for item in payload["next_actions"]]
    return "\n".join(lines)


def _internal_error(command: str, exc: Exception) -> Envelope:
    """An unexpected failure: one error envelope on stdout, the traceback on stderr."""
    trace_id = uuid.uuid4().hex[:16]
    sys.stderr.write(redact(f"trace {trace_id}\n{''.join(traceback.format_exception(exc))}"))
    if os.environ.get("SIGNALQUARRY_DEBUG") == "1":
        raise exc
    return Envelope(
        command=command,
        status="error",
        reason_codes=["INTERNAL_ERROR"],
        summary=redact(f"{type(exc).__name__}: {exc}")[:500],
        data={"trace_id": trace_id},
        next_actions=[
            {
                "command": "sqy doctor",
                "why": f"Check the environment; report the bug with trace {trace_id} from stderr.",
            }
        ],
    )


def _pop_detail(raw: list[str]) -> bool:
    """``--detail full`` (anywhere on the line) lifts the 64 KB cap on ``data``."""
    full = False
    for token in ("--detail=full", "--detail=summary"):
        while token in raw:
            raw.remove(token)
            full = token.endswith("full")
    while "--detail" in raw:
        index = raw.index("--detail")
        value = raw[index + 1] if index + 1 < len(raw) else ""
        if value not in ("full", "summary"):
            raise _UsageError("--detail takes full or summary")
        del raw[index : index + 2]
        full = value == "full"
    return full


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in raw or not sys.stdout.isatty()
    full_detail = False
    started = time.monotonic()
    args = None
    progress.configure(sys.stderr if as_json or os.environ.get("SIGNALQUARRY_PROGRESS") == "1" else None)
    try:
        full_detail = _pop_detail(raw)
        args = build_parser().parse_args(raw)
        envelope = args._handler(args)
    except _UsageError as exc:
        envelope = Envelope(
            command=raw[0] if raw else "", status="usage", reason_codes=["USAGE_INVALID"], summary=str(exc)
        )
    except SystemExit as exc:  # --help
        return int(exc.code or 0)
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 - stdout must still carry exactly one envelope
        envelope = _internal_error(raw[0] if raw else "", exc)
    finally:
        progress.configure(None)
    envelope.started = started  # duration_ms covers the whole command, not just envelope assembly
    if not full_detail:
        envelope = cap_data(envelope, api.cache_dir() / "envelopes")
    payload = envelope.as_dict()
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True) if as_json else _render_text(payload)
    print(redact(text))  # credential values never reach the output, whatever a handler put there
    if (
        args is not None
        and getattr(args, "command", None) == "paper"
        and args.action == "run-once"
        and args.notify_command is not None
        and envelope.exit_code != 0
    ):
        _notify_on_failure(args.notify_command, args.alias, envelope.exit_code)
    return envelope.exit_code


def _notify_on_failure(command: Path, alias: str, exit_code: int) -> None:
    """Run an explicit local notifier without passing broker or data credentials."""
    env = {key: os.environ[key] for key in ("HOME", "PATH", "LANG") if key in os.environ}
    env.update(SIGNALQUARRY_EXIT_CODE=str(exit_code), SIGNALQUARRY_ALIAS=alias)
    try:
        subprocess.run(
            [str(command)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        print("paper notification command failed", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

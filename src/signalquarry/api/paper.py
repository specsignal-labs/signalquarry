# SPDX-License-Identifier: Apache-2.0
"""``sqy paper …``: preflight, dry-run, arm, run-once, status, reconcile, halt, schedule.

Deployments live in ``paper/<alias>.paper.yaml``; their journal and arm token in
``paper/<alias>/``. Only the Alpaca **paper** origin exists. Arming needs a human
at an interactive terminal; agents may run everything else.
"""

from __future__ import annotations

import sys
import time as _time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from signalquarry._internal.canonical import file_sha256, to_canonical
from signalquarry._internal.contracts.paper import load_paper_config
from signalquarry._internal.data.alpaca import AlpacaDataClient, ProviderError
from signalquarry._internal.data.credentials import load_data_credentials, load_paper_credentials
from signalquarry._internal.data.dataset import Dataset, truncated, with_pending_session
from signalquarry._internal.data.library import LibraryError, build_dataset
from signalquarry._internal.evidence.runs import result_document, unique_run_id, write_run
from signalquarry._internal.paper.arm import sha256_hex
from signalquarry._internal.paper.brokers.alpaca_options import AlpacaOptionsVenue
from signalquarry._internal.paper.brokers.alpaca_paper import AlpacaPaperBroker
from signalquarry._internal.paper.brokers.fake import FakeBroker, exchange_time
from signalquarry._internal.paper.brokers.fake_options import FakeOptionsVenue
from signalquarry._internal.paper.brokers.fake_options import at as options_at
from signalquarry._internal.paper.models import PaperError
from signalquarry._internal.paper.options_runner import OptionsKernel
from signalquarry._internal.paper.runner import Deployment, Outcome, PaperKernel
from signalquarry._internal.paper.schedule import TARGETS, write_schedule
from signalquarry._internal.project.project import ProjectError, find_root, load_config, load_strategies
from signalquarry._internal.validation.ledger import LedgerError, latest_freeze
from signalquarry.api.envelope import Envelope
from signalquarry.api.resolve import Resolved, library_for, resolve


@dataclass(frozen=True)
class _Context:
    root: Path
    resolved: Resolved | None
    deployment: Deployment
    freeze_stale: bool


def _envelope(command: str, outcome: Outcome) -> Envelope:
    return Envelope(
        command=command,
        summary=outcome.summary,
        data=to_canonical(outcome.data),
        warnings=outcome.warnings,
        reason_codes=outcome.reason_codes,
    )


def failure_envelope(command: str, alias: str, exc: PaperError) -> Envelope:
    envelope = Envelope(command=command, status=exc.status, reason_codes=[exc.code], summary=str(exc))
    remedies = {
        "PAPER_NOT_ARMED": (
            "sqy paper preflight --alias {a}",
            "Review readiness; a human then runs `sqy paper arm`.",
        ),
        "PAPER_HALTED": ("sqy paper status --alias {a}", "Read the halt reason; only a human re-arms."),
        "PAPER_PENDING_ORDERS": ("sqy paper reconcile --alias {a}", "Finalize journaled orders."),
        "FREEZE_REQUIRED": ("sqy spec freeze --strategy <id>", "Freeze the configuration before arming."),
    }
    if exc.code in remedies:
        command_text, why = remedies[exc.code]
        envelope.next_actions = [{"command": command_text.format(a=alias), "why": why}]
    return envelope


def deployment_context(command: str, alias: str, project: Path | None) -> _Context | Envelope:
    try:
        root = find_root(project)
    except ProjectError as exc:
        return Envelope(
            command=command, status="invalid", reason_codes=[exc.code], summary=exc.detail or exc.code
        )
    path = root / "paper" / f"{alias}.paper.yaml"
    if not path.is_file():
        return Envelope(
            command=command,
            status="invalid",
            reason_codes=["PAPER_CONFIG_NOT_FOUND"],
            summary=str(path.relative_to(root)),
            data={
                "deployments": sorted(
                    p.name.removesuffix(".paper.yaml") for p in (root / "paper").glob("*.paper.yaml")
                )
            },
        )
    try:
        config = load_paper_config(path)
    except (ValidationError, ValueError) as exc:
        return Envelope(
            command=command, status="invalid", reason_codes=["PAPER_CONFIG_INVALID"], summary=str(exc)[:2000]
        )
    resolved = resolve(command, config.strategy, root)
    if isinstance(resolved, Envelope):
        if config.broker != "simulated" and resolved.reason_codes == ["PROVIDER_UNAVAILABLE"]:
            resolved = None  # paper sessions fetch their own data; no recorded dataset is needed
        else:
            return resolved
    if resolved is None:
        try:
            strategy = load_strategies(load_config(root))[config.strategy]
        except (ProjectError, KeyError) as exc:
            return Envelope(
                command=command, status="invalid", reason_codes=["STRATEGY_NOT_FOUND"], summary=str(exc)
            )
    else:
        strategy = resolved.strategy
    try:
        freeze = latest_freeze(root, strategy.spec.id)
    except LedgerError as exc:
        return Envelope(command=command, status="blocked", reason_codes=[exc.code], summary=str(exc))
    configuration_hash = strategy.configuration_hash
    current = freeze is not None and freeze["configuration_hash"] == configuration_hash
    deployment = Deployment(
        config=config,
        spec=strategy.spec,
        definition=strategy.definition,
        params=strategy.params,
        configuration_hash=configuration_hash,
        freeze_hash=freeze["freeze_hash"] if freeze and current else None,
        state_dir=root / "paper" / alias,
    )
    return _Context(root, resolved, deployment, freeze is not None and not current)


class _HasCalendar(Protocol):
    def calendar(self, start: date, end: date) -> list[date]: ...


def _alpaca_loader(
    root: Path, deployment: Deployment, broker: _HasCalendar, key: tuple[str, str]
) -> Callable[[date], Dataset]:
    spec = deployment.spec

    def load(session: date) -> Dataset:
        if spec.data.feed == "synthetic":
            raise PaperError(
                "PAPER_CONFIG_INVALID",
                "invalid",
                "a synthetic-feed strategy cannot run on a real paper account",
            )
        lookback = int(deployment.definition.lookback(deployment.params))
        start = session - timedelta(days=int(lookback * 1.6) + 30)
        calendar = broker.calendar(start, session)
        previous = [s for s in calendar if s < session]
        if session not in calendar or not previous:
            raise PaperError("PAPER_NO_SESSION_TODAY", "busy", session.isoformat())
        data_keys = load_data_credentials()
        client = AlpacaDataClient(*(data_keys.key_id, data_keys.secret_key) if data_keys else key)
        try:
            bars = client.daily_bars(spec.data.symbols, start, previous[-1], spec.data.feed)
            actions = client.corporate_actions(spec.data.symbols, start, session)
            dataset = build_dataset(bars, actions, source=f"alpaca:{spec.data.feed}")
        except (ProviderError, LibraryError) as exc:
            raise PaperError("PAPER_DATA_UNAVAILABLE", "unavailable", exc.code) from exc
        library = library_for(root)
        for page in [*bars, *actions]:
            library.store_page(page)
        if dataset.sessions[-1] != previous[-1]:
            raise PaperError("PAPER_DATA_STALE", "busy", f"latest bars {dataset.sessions[-1].isoformat()}")
        return with_pending_session(dataset, session)

    return load


def kernel_for(context: _Context) -> PaperKernel:
    if context.deployment.config.broker.startswith("plugin:"):
        if context.deployment.spec.kind == "options_single_leg":
            raise PaperError(
                "PAPER_KIND_UNSUPPORTED", "invalid", "broker plugins trade equity deployments only"
            )
        return _plugin_kernel(context)
    if context.deployment.spec.kind == "options_single_leg":
        return _options_kernel(context)  # type: ignore[return-value]
    deployment = context.deployment
    config = deployment.config
    if config.broker == "simulated":
        assert context.resolved is not None
        dataset = context.resolved.dataset
        session = dataset.sessions[-1]
        broker = FakeBroker(
            dataset, deployment.spec.account.initial_cash, account_id=f"simulated-{config.alias}"
        )
        broker.pre_open(session)
        return PaperKernel(
            deployment,
            broker,
            lambda s: with_pending_session(truncated(dataset, dataset.index_of(s)), s),
            now=lambda: exchange_time(session, time(9, 10)),
        )
    try:
        keys = load_paper_credentials(config.profile)
    except PermissionError as exc:
        raise PaperError(
            "CREDENTIALS_FILE_PERMISSIONS_TOO_OPEN", "blocked", "chmod 600 credentials.toml"
        ) from exc
    if keys is None:
        raise PaperError("PAPER_CREDENTIALS_MISSING", "unavailable", config.profile)
    if sha256_hex(keys.key_id) in config.denied_key_id_sha256:
        raise PaperError("PAPER_KEY_DENIED", "blocked", keys.source)
    broker = AlpacaPaperBroker(keys.key_id, keys.secret_key)
    return PaperKernel(
        deployment, broker, _alpaca_loader(context.root, deployment, broker, (keys.key_id, keys.secret_key))
    )


BROKER_METHODS = (
    "account",
    "clock",
    "positions",
    "open_orders",
    "order_by_client_id",
    "submit",
    "cancel",
    "calendar",
)


def _plugin_kernel(context: _Context) -> PaperKernel:
    """A paper broker from a ``signalquarry.brokers`` plugin; market data stays the project's."""
    from signalquarry.plugins import discover

    deployment = context.deployment
    config = deployment.config
    name = config.broker.removeprefix("plugin:")
    found = {loaded.name: loaded for loaded in discover("brokers").plugins}
    if name not in found:
        raise PaperError("PAPER_BROKER_PLUGIN_NOT_FOUND", "invalid", name)
    try:
        keys = load_paper_credentials(config.profile)
    except PermissionError as exc:
        raise PaperError(
            "CREDENTIALS_FILE_PERMISSIONS_TOO_OPEN", "blocked", "chmod 600 credentials.toml"
        ) from exc
    if keys is not None and sha256_hex(keys.key_id) in config.denied_key_id_sha256:
        raise PaperError("PAPER_KEY_DENIED", "blocked", keys.source)
    broker = found[name].plugin.create(
        config.alias, keys.key_id if keys else None, keys.secret_key if keys else None
    )
    if getattr(broker, "paper_only", False) is not True:
        raise PaperError("BROKER_NOT_PAPER_ONLY", "blocked", f"plugin {name} must return a paper_only broker")
    missing = [m for m in BROKER_METHODS if not callable(getattr(broker, m, None))]
    if missing:
        raise PaperError(
            "BROKER_PLUGIN_INVALID", "invalid", f"plugin {name} broker lacks {', '.join(missing)}"
        )
    data_keys = load_data_credentials()
    fallback = (data_keys.key_id, data_keys.secret_key) if data_keys else ("", "")
    return PaperKernel(deployment, broker, _alpaca_loader(context.root, deployment, broker, fallback))


def _options_kernel(context: _Context) -> OptionsKernel:
    deployment = context.deployment
    config = deployment.config
    if config.broker == "simulated":
        assert context.resolved is not None
        dataset = context.resolved.dataset
        session = dataset.sessions[-1]
        venue = FakeOptionsVenue(
            dataset, deployment.spec.account.initial_cash, account_id=f"simulated-{config.alias}"
        )
        venue.now = options_at(session, time(9, 35))
        return OptionsKernel(
            deployment,
            venue,
            lambda s: with_pending_session(truncated(dataset, dataset.index_of(s)), s),
            now=lambda: venue.now,
        )
    try:
        keys = load_paper_credentials(config.profile)
    except PermissionError as exc:
        raise PaperError(
            "CREDENTIALS_FILE_PERMISSIONS_TOO_OPEN", "blocked", "chmod 600 credentials.toml"
        ) from exc
    if keys is None:
        raise PaperError("PAPER_CREDENTIALS_MISSING", "unavailable", config.profile)
    if sha256_hex(keys.key_id) in config.denied_key_id_sha256:
        raise PaperError("PAPER_KEY_DENIED", "blocked", keys.source)
    broker = AlpacaPaperBroker(keys.key_id, keys.secret_key)
    data_keys = load_data_credentials()
    data = AlpacaDataClient(
        *((data_keys.key_id, data_keys.secret_key) if data_keys else (keys.key_id, keys.secret_key))
    )
    return OptionsKernel(
        deployment,
        AlpacaOptionsVenue(broker, data),
        _alpaca_loader(context.root, deployment, broker, (keys.key_id, keys.secret_key)),
    )


def _run(
    command: str,
    alias: str,
    project: Path | None,
    action: Callable[[PaperKernel, _Context], Outcome],
    *,
    live_only: bool = False,
) -> Envelope:
    context = deployment_context(command, alias, project)
    if isinstance(context, Envelope):
        return context
    try:
        if live_only and context.deployment.config.broker == "simulated":
            raise PaperError("PAPER_BROKER_SIMULATED", "disabled", alias)
        envelope = _envelope(command, action(kernel_for(context), context))
    except PaperError as exc:
        return failure_envelope(command, alias, exc)
    envelope.evidence = {
        "grade": "simulated" if context.deployment.config.broker == "simulated" else "paper",
        "claim_level": "none",
        "freeze": context.deployment.freeze_hash,
        "freeze_stale": context.freeze_stale,
    }
    return envelope


def paper_preflight(alias: str, *, project: Path | None = None) -> Envelope:
    envelope = _run("paper preflight", alias, project, lambda kernel, _: kernel.preflight())
    if envelope.status == "ok" and not envelope.data.get("ready"):
        envelope.status = "blocked"
    return envelope


def paper_dry_run(alias: str, *, project: Path | None = None) -> Envelope:
    return _run("paper dry-run", alias, project, lambda kernel, _: kernel.dry_run())


def paper_status(alias: str, *, project: Path | None = None) -> Envelope:
    return _run("paper status", alias, project, lambda kernel, _: kernel.status())


def paper_run_once(alias: str, *, project: Path | None = None) -> Envelope:
    return _run("paper run-once", alias, project, lambda kernel, _: kernel.run_once(), live_only=True)


def paper_reconcile(alias: str, *, project: Path | None = None) -> Envelope:
    return _run("paper reconcile", alias, project, lambda kernel, _: kernel.reconcile(), live_only=True)


def paper_halt(alias: str, reason: str, *, project: Path | None = None) -> Envelope:
    return _run("paper halt", alias, project, lambda kernel, _: kernel.halt(reason), live_only=True)


def paper_drift(alias: str, *, project: Path | None = None) -> Envelope:
    """Replay the journal against the engine, measure fills, apply model costs and dividends; record G5."""

    def action(kernel: PaperKernel, context: _Context) -> Outcome:
        outcome = kernel.drift()
        deployment = context.deployment
        run_id = unique_run_id(context.root, f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-drift-{alias}")
        artifacts = write_run(
            context.root,
            run_id,
            {
                "drift.json": result_document(
                    kind="paper_drift",
                    run_id=run_id,
                    strategy_id=deployment.spec.id,
                    alias=alias,
                    configuration_hash=deployment.configuration_hash,
                    freeze_hash=deployment.freeze_hash,
                    g5=outcome.data["g5"],
                    shadow=outcome.data["shadow"],
                    sessions=len(outcome.data["sessions"]),
                )
            },
        )
        outcome.data["run_id"] = run_id
        outcome.data["artifact"] = artifacts[0]["path"]
        return outcome

    envelope = _run("paper drift", alias, project, action)
    if envelope.status == "ok" and envelope.reason_codes:
        envelope.status = "blocked"
    return envelope


def paper_backup(alias: str, *, out: Path | None = None, project: Path | None = None) -> Envelope:
    return _run(
        "paper backup",
        alias,
        project,
        lambda kernel, context: kernel.backup(out or context.root / ".signalquarry" / "backups" / alias),
    )


def paper_verify_continuity(alias: str, backup: Path, *, project: Path | None = None) -> Envelope:
    return _run("paper verify-continuity", alias, project, lambda kernel, _: kernel.verify_continuity(backup))


def paper_run(
    alias: str,
    *,
    project: Path | None = None,
    interval: float = 60.0,
    max_minutes: float = 30.0,
    sleep: Callable[[float], None] = _time.sleep,
    clock: Callable[[], float] = _time.monotonic,
) -> Envelope:
    """Call run-once until it is done: retry while busy (75) or unavailable (69); stop on anything else."""
    deadline = clock() + max_minutes * 60
    attempts = 0
    while True:
        attempts += 1
        envelope = paper_run_once(alias, project=project)
        envelope.command = "paper run"
        envelope.data = {**envelope.data, "attempts": attempts}
        if envelope.status not in ("busy", "unavailable") or clock() + interval > deadline:
            if envelope.status in ("busy", "unavailable"):
                envelope.warnings.append("PAPER_RUN_GAVE_UP")
            return envelope
        sleep(interval if envelope.status == "busy" else min(interval * 2 ** min(attempts, 4), 900))


def paper_arm(
    alias: str,
    reason: str,
    *,
    project: Path | None = None,
    confirm: Callable[[str], str] | None = None,
    interactive: bool | None = None,
) -> Envelope:
    """HUMAN ONLY. ``confirm`` asks the human to type the alias; the CLI passes ``input``."""

    def action(kernel: PaperKernel, context: _Context) -> Outcome:
        if context.freeze_stale:
            raise PaperError("FREEZE_STALE", "blocked", "the configuration changed since `sqy spec freeze`")
        tty = sys.stdin.isatty() and sys.stdout.isatty() if interactive is None else interactive
        if not tty or confirm is None:
            raise PaperError(
                "PAPER_ARM_REQUIRES_HUMAN", "disabled", "run `sqy paper arm` yourself in a terminal"
            )
        typed = confirm(
            f"Arm paper submission for '{alias}' ({context.deployment.spec.id}, "
            f"freeze {str(context.deployment.freeze_hash)[:19]}). Type the alias to confirm: "
        )
        if typed.strip() != alias:
            raise PaperError("PAPER_ARM_NOT_CONFIRMED", "disabled", "confirmation did not match")
        return kernel.arm(reason=reason)

    return _run("paper arm", alias, project, action, live_only=True)


def paper_schedule(alias: str, target: str, *, project: Path | None = None) -> Envelope:
    context = deployment_context("paper schedule", alias, project)
    if isinstance(context, Envelope):
        return context
    if target not in TARGETS:
        return Envelope(
            command="paper schedule",
            status="usage",
            reason_codes=["USAGE_INVALID"],
            summary=f"target must be one of {', '.join(TARGETS)}",
        )
    options = context.deployment.spec.kind == "options_single_leg"
    if options and target == "github-actions":
        return Envelope(
            command="paper schedule",
            status="usage",
            reason_codes=["USAGE_INVALID"],
            summary="options deployments poll every minute; use systemd, launchd or cron",
        )
    written = write_schedule(context.root, context.deployment.config, target, poll=options)
    envelope = Envelope(
        command="paper schedule",
        summary=f"wrote {len(written)} {target} file(s); review and install them yourself",
        data={
            "target": target,
            "files": [str(p.relative_to(context.root)) for p in written],
            "demo_only": target == "github-actions",
        },
        artifacts=[
            {"path": str(p.relative_to(context.root)), "sha256": file_sha256(p), "kind": "schedule"}
            for p in written
        ],
    )
    if target == "github-actions":
        envelope.warnings.append("SCHEDULE_GITHUB_ACTIONS_DEMO_ONLY")
    return envelope


__all__: list[str] = [
    "paper_arm",
    "paper_backup",
    "paper_drift",
    "paper_dry_run",
    "paper_halt",
    "paper_preflight",
    "paper_reconcile",
    "paper_run",
    "paper_run_once",
    "paper_schedule",
    "paper_status",
    "paper_verify_continuity",
]

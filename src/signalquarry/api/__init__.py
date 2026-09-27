# SPDX-License-Identifier: Apache-2.0
"""Command facade. Every function returns an :class:`Envelope`; the CLI only prints it."""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from importlib import resources
from pathlib import Path

from signalquarry import __version__
from signalquarry._internal.canonical import CANONICAL_VERSION
from signalquarry._internal.contracts.reason_codes import REASON_CODES, lookup
from signalquarry.api.envelope import Envelope

RUNTIME_DEPENDENCIES = ("pydantic", "numpy", "pyarrow", "yaml")


def version() -> Envelope:
    return Envelope(
        command="version",
        summary=f"signalquarry {__version__}",
        data={
            "framework_version": __version__,
            "python": platform.python_version(),
            "platform": f"{sys.platform}-{platform.machine()}",
            "canonical": CANONICAL_VERSION,
        },
    )


def _rosetta() -> bool:
    if sys.platform != "darwin" or platform.machine() != "x86_64":
        return False
    try:
        result = subprocess.run(
            ["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.stdout.strip() == "1"


def cache_dir() -> Path:
    return Path(
        os.environ.get("SIGNALQUARRY_CACHE_DIR") or Path.home() / ".cache" / "signalquarry"
    ).expanduser()


def doctor() -> Envelope:
    envelope = Envelope(command="doctor")
    checks: dict[str, dict[str, object]] = {}
    blocking: list[str] = []

    ok = sys.version_info >= (3, 12)
    checks["python"] = {"ok": ok, "version": platform.python_version()}
    if not ok:
        blocking.append("PYTHON_VERSION_UNSUPPORTED")
    emulated = _rosetta()
    checks["architecture"] = {"ok": not emulated, "machine": platform.machine()}
    if emulated:
        envelope.warnings.append("PYTHON_ARCH_EMULATED")

    missing: list[str] = []
    for name in RUNTIME_DEPENDENCIES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    checks["dependencies"] = {"ok": not missing, "missing": missing}
    if missing:
        blocking.append("DEPENDENCY_MISSING")

    from signalquarry.plugins import GROUPS, discover

    plugins: dict[str, list[str]] = {}
    plugin_errors: list[str] = []
    for group in GROUPS:
        found = discover(group)
        plugins[group] = [f"{p.name} ({p.source})" for p in found.plugins]
        plugin_errors += list(found.errors)
    checks["plugins"] = {"ok": not plugin_errors, **plugins, "errors": plugin_errors}
    if plugin_errors:
        envelope.warnings.append("PLUGIN_ERROR")

    import shutil

    ots = shutil.which("ots")
    # Optional: commitments work without it and stay "pending" until stamped.
    checks["opentimestamps"] = {"ok": True, "client": ots, "installed": ots is not None}
    if ots is None:
        envelope.warnings.append("OTS_CLIENT_MISSING")

    path = cache_dir()
    probe = path if path.exists() else next((p for p in path.parents if p.exists()), path)
    writable = os.access(probe, os.W_OK)
    checks["cache_dir"] = {"ok": writable, "path": str(path)}
    if not writable:
        envelope.warnings.append("CACHE_DIR_NOT_WRITABLE")

    from signalquarry._internal.data.credentials import load_data_credentials

    try:
        credentials = load_data_credentials()
        source = credentials.source if credentials else None
    except PermissionError:
        source = None
        envelope.warnings.append("CREDENTIALS_FILE_PERMISSIONS_TOO_OPEN")
    checks["data_credentials"] = {"ok": source is not None, "source": source}
    if source is None:
        envelope.warnings.append("DATA_CREDENTIALS_MISSING")
        envelope.next_actions.append(
            {"command": "sqy init --demo", "why": "The synthetic provider needs no credentials."}
        )

    envelope.data = {"checks": checks}
    envelope.reason_codes = blocking
    envelope.status = "blocked" if blocking else "ok"
    envelope.summary = "environment ready" if not blocking else "environment not ready"
    return envelope


def _schema_names() -> list[str]:
    root = resources.files("signalquarry") / "schemas" / "v1"
    names = [item.name.removesuffix(".json") for item in root.iterdir() if item.name.endswith(".json")]
    names += [
        f"showcase/{item.name.removesuffix('.json')}"
        for item in (root / "showcase").iterdir()
        if item.name.endswith(".json")
    ]
    return sorted(names)


def schema(name: str | None = None) -> Envelope:
    names = _schema_names()
    if name is None:
        return Envelope(command="schema", summary=f"{len(names)} schemas", data={"schemas": names})
    if name not in names:
        return Envelope(
            command="schema",
            status="invalid",
            reason_codes=["SCHEMA_UNKNOWN"],
            summary=f"unknown schema {name!r}",
            data={"schemas": names},
        )
    text = (resources.files("signalquarry") / "schemas" / "v1" / f"{name}.json").read_text(encoding="utf-8")
    return Envelope(command="schema", summary=name, data={"name": name, "schema": json.loads(text)})


def explain(code: str) -> Envelope:
    item = lookup(code.strip().upper())
    if item is None:
        return Envelope(
            command="explain",
            status="invalid",
            reason_codes=["REASON_CODE_UNKNOWN"],
            summary=f"unknown code {code!r}",
        )
    return Envelope(
        command="explain",
        summary=item.description,
        data={
            "code": item.code,
            "category": item.category.value,
            "description": item.description,
            "remedy": item.remedy,
        },
    )


def reason_code_catalog() -> dict[str, dict[str, str]]:
    return {
        code: {"category": item.category.value, "description": item.description}
        for code, item in sorted(REASON_CODES.items())
    }


from signalquarry.api.commit import commit_create, commit_reveal, commit_verify  # noqa: E402
from signalquarry.api.data import (  # noqa: E402
    data_fetch,
    data_ls,
    data_probe_options,
    data_record_options,
    data_verify,
)
from signalquarry.api.docs import docs  # noqa: E402
from signalquarry.api.evidence import (  # noqa: E402
    evaluate_command,
    evidence_verify,
    holdout_seal,
    holdout_status,
    spec_freeze,
    trials_extend,
    trials_ls,
    trials_show,
)
from signalquarry.api.paper import (  # noqa: E402
    paper_arm,
    paper_backup,
    paper_drift,
    paper_dry_run,
    paper_halt,
    paper_preflight,
    paper_reconcile,
    paper_run,
    paper_run_once,
    paper_schedule,
    paper_status,
    paper_verify_continuity,
)
from signalquarry.api.perf import perf_capture, perf_publish  # noqa: E402
from signalquarry.api.project import backtest, check, init, upgrade_agents_md  # noqa: E402
from signalquarry.api.publish import bundle_verify, evidence_export  # noqa: E402
from signalquarry.api.report import report  # noqa: E402
from signalquarry.api.sweep import sweep  # noqa: E402

__all__ = [
    "Envelope",
    "backtest",
    "bundle_verify",
    "cache_dir",
    "check",
    "commit_create",
    "commit_reveal",
    "commit_verify",
    "data_fetch",
    "data_ls",
    "data_probe_options",
    "data_record_options",
    "data_verify",
    "docs",
    "doctor",
    "evaluate_command",
    "evidence_export",
    "evidence_verify",
    "holdout_seal",
    "holdout_status",
    "spec_freeze",
    "sweep",
    "trials_extend",
    "trials_ls",
    "upgrade_agents_md",
    "trials_show",
    "explain",
    "init",
    "paper_arm",
    "perf_capture",
    "perf_publish",
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
    "reason_code_catalog",
    "report",
    "schema",
    "version",
]

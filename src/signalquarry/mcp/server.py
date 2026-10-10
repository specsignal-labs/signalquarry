# SPDX-License-Identifier: Apache-2.0
"""Local stdio MCP tools. Every operation returns the ordinary CLI envelope.

Install ``signalquarry[mcp]`` and run ``sqy-mcp`` from an MCP host. Authored
strategy code is imported by some tools; run projects in a trusted environment.
Human-only paper arming and trial-budget extension are intentionally absent.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from signalquarry import __version__, api
from signalquarry.api.envelope import Envelope, cap_data

mcp = MCPServer(
    "SignalQuarry",
    version=__version__,
    description="Local strategy research, evidence, and Alpaca paper tools with structured envelopes.",
)


def _path(value: str | None) -> Path | None:
    return Path(value).expanduser() if value is not None else None


def _result(envelope: Envelope) -> dict[str, Any]:
    return cap_data(envelope, api.cache_dir() / "envelopes").as_dict()


def _date(value: str | None, command: str) -> date | None | Envelope:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return Envelope(
            command=command,
            status="usage",
            reason_codes=["USAGE_INVALID"],
            summary=f"invalid ISO date: {value[:40]}",
        )


@mcp.tool()
def sqy_version() -> dict[str, Any]:
    """Report the framework, Python and canonical format versions."""
    return _result(api.version())


@mcp.tool()
def sqy_doctor() -> dict[str, Any]:
    """Check local runtime and optional integration health."""
    return _result(api.doctor())


@mcp.tool()
def sqy_explain(reason_code: str) -> dict[str, Any]:
    """Explain one SignalQuarry reason code and its recovery action."""
    return _result(api.explain(reason_code))


@mcp.tool()
def sqy_init(path: str, demo: bool = False, kind: str = "equity") -> dict[str, Any]:
    """Create a new strategy project. The destination must be empty."""
    return _result(api.init(Path(path).expanduser(), demo=demo, kind=kind))


@mcp.tool()
def sqy_check(strategy_id: str, project: str | None = None) -> dict[str, Any]:
    """Run synthetic conformance for a registered strategy."""
    return _result(api.check(strategy_id, project=_path(project)))


@mcp.tool()
def sqy_factor_ls(project: str | None = None) -> dict[str, Any]:
    """List registered research factors and their configuration identities."""
    return _result(api.factor_ls(project=_path(project)))


@mcp.tool()
def sqy_check_factor(factor_id: str, project: str | None = None) -> dict[str, Any]:
    """Check a registered factor against synthetic contract and lookahead probes."""
    return _result(api.check_registered_factor(factor_id, project=_path(project)))


@mcp.tool()
def sqy_data_fetch(strategy_id: str, project: str | None = None) -> dict[str, Any]:
    """Fetch the strategy's declared market data through the configured provider."""
    return _result(api.data_fetch(strategy_id=strategy_id, project=_path(project)))


@mcp.tool()
def sqy_data_ls(project: str | None = None) -> dict[str, Any]:
    """List locally recorded datasets and their identities."""
    return _result(api.data_ls(project=_path(project)))


@mcp.tool()
def sqy_data_verify(project: str | None = None) -> dict[str, Any]:
    """Verify local dataset manifests and cached source bytes."""
    return _result(api.data_verify(project=_path(project)))


@mcp.tool()
def sqy_backtest(
    strategy_id: str,
    project: str | None = None,
    start: str | None = None,
    end: str | None = None,
    params: list[str] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Run a strategy backtest with optional ISO session boundaries.

    ``params`` are NAME=VALUE overrides for this run only (strategy.yaml is not changed; on
    real data a new configuration is a trial). ``label`` is a short name stored with the run.
    """
    first = _date(start, "backtest")
    last = _date(end, "backtest")
    if isinstance(first, Envelope):
        return _result(first)
    if isinstance(last, Envelope):
        return _result(last)
    return _result(
        api.backtest(
            strategy_id, start=first, end=last, params=params or (), label=label, project=_path(project)
        )
    )


@mcp.tool()
def sqy_diagnose(
    strategy_id: str,
    project: str | None = None,
    run_id: str | None = None,
    exposures: list[str] | None = None,
) -> dict[str, Any]:
    """Write run diagnostics: regimes, costs, folds, parameters and optional passive exposures.

    Descriptive only: no trial is recorded and no gate or claim level changes.
    """
    return _result(
        api.diagnose(strategy_id, run_id=run_id, exposures=exposures or (), project=_path(project))
    )


@mcp.tool()
def sqy_runs_ls(
    project: str | None = None, strategy_id: str | None = None, limit: int = 20
) -> dict[str, Any]:
    """List recent backtest runs (sweep points and study arms included), newest first."""
    return _result(api.runs_ls(strategy_id=strategy_id, limit=limit, project=_path(project)))


@mcp.tool()
def sqy_runs_show(run_id: str, project: str | None = None) -> dict[str, Any]:
    """Show one run's verified result document and its artifacts."""
    return _result(api.runs_show(run_id, project=_path(project)))


@mcp.tool()
def sqy_runs_compare(run_ids: list[str], project: str | None = None) -> dict[str, Any]:
    """Compare runs against the first one; runs that are not comparable are not differenced."""
    return _result(api.runs_compare(run_ids, project=_path(project)))


@mcp.tool()
def sqy_study_init(strategy_id: str, study_id: str, project: str | None = None) -> dict[str, Any]:
    """Scaffold studies/ID/study.yaml for a strategy; nothing is run."""
    return _result(api.study_init(strategy_id, study_id, project=_path(project)))


@mcp.tool()
def sqy_study_check(study_id: str, project: str | None = None) -> dict[str, Any]:
    """Validate a study and show its arms and the trials it would record; runs nothing."""
    return _result(api.study_check(study_id, project=_path(project)))


@mcp.tool()
def sqy_study_run(
    study_id: str, project: str | None = None, rerun: bool = False, jobs: int = 1
) -> dict[str, Any]:
    """Run every arm of a study and judge it by its declared rule.

    Spends trial budget on real data and is refused beforehand when a budget would be
    exceeded. It never opens a holdout and never raises a claim level. `jobs` above 1
    simulates arms in worker processes; what is recorded does not depend on it.
    """
    return _result(api.study_run(study_id, rerun=rerun, jobs=jobs, project=_path(project)))


@mcp.tool()
def sqy_study_ls(project: str | None = None) -> dict[str, Any]:
    """List the project's studies and the verdict of each one's latest run."""
    return _result(api.study_ls(project=_path(project)))


@mcp.tool()
def sqy_study_show(study_id: str, project: str | None = None) -> dict[str, Any]:
    """Show the latest recorded result of a study."""
    return _result(api.study_show(study_id, project=_path(project)))


@mcp.tool()
def sqy_spec_freeze(strategy_id: str, project: str | None = None) -> dict[str, Any]:
    """Freeze the strategy specification and declared evaluation settings."""
    return _result(api.spec_freeze(strategy_id, project=_path(project)))


@mcp.tool()
def sqy_evaluate(strategy_id: str, project: str | None = None) -> dict[str, Any]:
    """Evaluate walk-forward and stress gates without opening a sealed holdout."""
    return _result(api.evaluate_command(strategy_id, project=_path(project)))


@mcp.tool()
def sqy_trials_ls(project: str | None = None, family: str | None = None) -> dict[str, Any]:
    """Read the append-only trial ledger and project-wide trial count."""
    return _result(api.trials_ls(family=family, project=_path(project)))


@mcp.tool()
def sqy_holdout_status(project: str | None = None) -> dict[str, Any]:
    """Read holdout seals and opening state; this tool does not open them."""
    return _result(api.holdout_status(project=_path(project)))


@mcp.tool()
def sqy_report(strategy_id: str, project: str | None = None, run_id: str | None = None) -> dict[str, Any]:
    """Render an evidence report for a recorded strategy run."""
    return _result(api.report(strategy_id, run_id=run_id, project=_path(project)))


@mcp.tool()
def sqy_paper_preflight(alias: str, project: str | None = None) -> dict[str, Any]:
    """Check a paper deployment's readiness without submitting orders."""
    return _result(api.paper_preflight(alias, project=_path(project)))


@mcp.tool()
def sqy_paper_dry_run(alias: str, project: str | None = None) -> dict[str, Any]:
    """Plan one paper session without submitting orders."""
    return _result(api.paper_dry_run(alias, project=_path(project)))


@mcp.tool()
def sqy_paper_status(alias: str, project: str | None = None) -> dict[str, Any]:
    """Read the paper kernel's lease, journal, arm and reconciliation state."""
    return _result(api.paper_status(alias, project=_path(project)))


@mcp.tool()
def sqy_paper_run_once(alias: str, project: str | None = None) -> dict[str, Any]:
    """Run one paper session only when explicitly requested by the user.

    The existing kernel still requires human arming, a lease, journal integrity
    and reconciliation; this tool cannot arm the deployment.
    """
    return _result(api.paper_run_once(alias, project=_path(project)))


@mcp.tool()
def sqy_paper_reconcile(alias: str, project: str | None = None) -> dict[str, Any]:
    """Reconcile the paper broker against the local journal."""
    return _result(api.paper_reconcile(alias, project=_path(project)))


@mcp.tool()
def sqy_paper_halt(alias: str, reason: str, project: str | None = None) -> dict[str, Any]:
    """Halt a paper deployment with a recorded reason."""
    return _result(api.paper_halt(alias, reason, project=_path(project)))


@mcp.tool()
def sqy_universe_verify(project: str | None = None) -> dict[str, Any]:
    """Verify locally captured asset snapshots."""
    return _result(api.universe_verify(project=_path(project)))


def main() -> None:
    """Serve the local MCP protocol on standard input and output."""
    mcp.run()


if __name__ == "__main__":
    main()

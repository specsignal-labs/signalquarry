# SPDX-License-Identifier: Apache-2.0
"""Read-only access to run results for notebooks and scripts (provisional in 0.x)."""

# PyArrow ships no type stubs; keep strict checking for the typed loader code.
# pyright: strict, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false

from __future__ import annotations

import csv
import importlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from signalquarry._internal.evidence.runs import iter_runs, read_curve, result_hash_ok
from signalquarry._internal.project.project import ProjectError, find_root


class ResearchError(Exception):
    """A project, artifact or optional dependency could not be loaded. ``code`` identifies it."""

    code: str

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        return MappingProxyType({key: _freeze(item) for key, item in mapping.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in cast(list[Any], value))
    return value


def _root(project: Path | str | None) -> Path:
    try:
        return find_root(Path(project) if project is not None else None)
    except ProjectError as exc:
        raise ResearchError("PROJECT_NOT_FOUND", exc.detail) from exc


def _document(
    root: Path, kind: str, identifier: str, filename: str, schema: str
) -> tuple[Path, dict[str, Any]]:
    directory = root / ".signalquarry" / kind / identifier
    if (
        not identifier
        or identifier in (".", "..")
        or "/" in identifier
        or "\\" in identifier
        or not directory.is_dir()
    ):
        raise ResearchError("RUN_NOT_FOUND", identifier)
    try:
        parsed: Any = json.loads((directory / filename).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResearchError(
            "RUN_ARTIFACT_INVALID", f"{identifier}: {filename} is missing or unreadable"
        ) from exc
    document = cast(dict[str, Any], parsed) if isinstance(parsed, dict) else {}
    if document.get("schema") != schema:
        raise ResearchError("RUN_ARTIFACT_INVALID", f"{identifier}: unexpected {filename} schema")
    return directory, document


def _immutable(values: NDArray[np.float64]) -> NDArray[np.float64]:
    # Immutable storage also prevents callers from switching the writeable flag back on.
    return np.frombuffer(values.tobytes(), dtype=np.float64)


_FILLS_SCHEMA = pa.schema(
    [
        ("session", pa.date32()),
        ("symbol", pa.string()),
        ("side", pa.string()),
        ("quantity", pa.float64()),
        ("price", pa.float64()),
        ("fee", pa.float64()),
        ("settle_session", pa.date32()),
    ]
)


@dataclass(frozen=True)
class Run:
    """A verified result document and its session-aligned equity curves."""

    run_id: str
    path: Path
    document: Mapping[str, Any]
    sessions: tuple[date, ...]
    equity: NDArray[np.float64]
    returns: NDArray[np.float64]
    benchmark: NDArray[np.float64] | None

    @property
    def strategy_id(self) -> str:
        return str(self.document["strategy_id"])

    @property
    def label(self) -> str | None:
        return cast(str | None, self.document.get("label"))

    @property
    def params(self) -> Mapping[str, Any]:
        return self.document.get("params") or MappingProxyType({})

    @property
    def metrics(self) -> Mapping[str, Any]:
        return self.document.get("metrics") or MappingProxyType({})

    @property
    def configuration_hash(self) -> str:
        return str(self.document["configuration_hash"])

    @property
    def grade(self) -> str | None:
        evidence: Mapping[str, Any] = self.document.get("evidence") or {}
        return cast(str | None, evidence.get("grade"))

    def fills(self) -> pa.Table:
        """Read fill rows; summary runs return an empty table with the same schema."""
        path = self.path / "fills.csv"
        rows: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    rows.append(
                        {
                            "session": date.fromisoformat(row["session"]),
                            "symbol": row["symbol"],
                            "side": row["side"],
                            **{key: float(row[key]) for key in ("quantity", "price", "fee")},
                            "settle_session": date.fromisoformat(row["settle_session"])
                            if row["settle_session"]
                            else None,
                        }
                    )
        except FileNotFoundError:
            pass
        except (OSError, KeyError, ValueError) as exc:
            raise ResearchError("RUN_ARTIFACT_INVALID", f"{self.run_id}: fills.csv is unreadable") from exc
        return pa.Table.from_pylist(rows, schema=_FILLS_SCHEMA)

    def decisions(self) -> Iterator[dict[str, Any]]:
        """Yield decision records; summary runs have none."""
        try:
            with (self.path / "decisions.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    parsed: Any = json.loads(line)
                    if not isinstance(parsed, dict):
                        raise ValueError("decision must be a JSON object")
                    yield cast(dict[str, Any], parsed)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            raise ResearchError(
                "RUN_ARTIFACT_INVALID", f"{self.run_id}: decisions.jsonl is unreadable"
            ) from exc

    def to_arrow(self) -> pa.Table:
        """Session, equity, simple daily returns and optional benchmark equity."""
        return pa.table(
            {
                "session": pa.array(self.sessions, type=pa.date32()),
                "equity": pa.array(self.equity, type=pa.float64()),
                "returns": pa.array(self.returns, type=pa.float64()),
                "benchmark": pa.array(
                    self.benchmark if self.benchmark is not None else [None] * len(self.sessions),
                    type=pa.float64(),
                ),
            }
        )

    def to_pandas(self) -> Any:
        """The Arrow table as a pandas DataFrame indexed by session (pandas is optional)."""
        try:
            pandas = importlib.import_module("pandas")
        except ImportError as exc:
            raise ResearchError("PANDAS_NOT_INSTALLED", "Install pandas to use Run.to_pandas()") from exc
        return pandas.DataFrame(self.to_arrow().to_pydict()).set_index("session")


@dataclass(frozen=True)
class Sweep:
    """A parameter grid, its recorded points and their run ids."""

    sweep_id: str
    path: Path
    document: Mapping[str, Any]

    @property
    def grid(self) -> Mapping[str, Any]:
        return self.document["grid"]

    @property
    def points(self) -> tuple[Mapping[str, Any], ...]:
        return self.document["points"]

    @property
    def pbo(self) -> Mapping[str, Any] | None:
        return self.document.get("pbo")

    def runs(self) -> tuple[Run, ...]:
        """Load one run per point, preserving point order."""
        return tuple(load_run(point["run_id"], self.path) for point in self.points)

    def to_arrow(self) -> pa.Table:
        """One row per point, with grid axes followed by run identity and metrics."""
        columns: dict[str, Any] = {
            axis: [point["params"][axis] for point in self.points] for axis in self.grid
        }
        columns.update(
            {
                key: pa.array([point.get(key) for point in self.points], type=kind)
                for key, kind in (
                    ("run_id", pa.string()),
                    ("configuration_hash", pa.string()),
                    ("total_return", pa.float64()),
                    ("sharpe", pa.float64()),
                    ("max_drawdown", pa.float64()),
                    ("fills", pa.int64()),
                )
            }
        )
        return pa.table(columns)


def list_runs(
    project: Path | str | None = None, *, strategy: str | None = None
) -> tuple[Mapping[str, Any], ...]:
    """Small run summaries, newest first, optionally filtered by strategy id."""
    rows: list[Mapping[str, Any]] = []
    for _, document in reversed(iter_runs(_root(project), strategy_id=strategy)):
        metrics: Mapping[str, Any] = document.get("metrics") or {}
        rows.append(
            _freeze(
                {
                    **{
                        key: document.get(key)
                        for key in (
                            "run_id",
                            "strategy_id",
                            "label",
                            "command",
                            "created_at",
                            "configuration_hash",
                            "params",
                        )
                    },
                    **{key: metrics.get(key) for key in ("total_return", "sharpe", "max_drawdown")},
                }
            )
        )
    return tuple(rows)


def load_run(run_id: str, project: Path | str | None = None) -> Run:
    """Load and verify a backtest or sweep-point result and its equity curves."""
    directory, document = _document(_root(project), "runs", run_id, "result.json", "signalquarry.result/v1")
    if not result_hash_ok(document):
        raise ResearchError("RUN_ARTIFACT_INVALID", f"{run_id}: result.json does not match its recorded hash")
    try:
        sessions, values = read_curve(directory / "equity.csv")
        metrics: Mapping[str, Any] = document.get("metrics") or {}
        if not sessions or len(sessions) != metrics.get("sessions"):
            raise ValueError("equity.csv does not match result.json")
        equity = np.array(values, dtype=np.float64)
        spec: Mapping[str, Any] = document.get("spec") or {}
        account: Mapping[str, Any] = spec.get("account") or {}
        initial = float(account.get("initial_cash", equity[0]))
        returns = equity / np.concatenate(([initial], equity[:-1])) - 1.0
        if "initial_cash" not in account:
            returns[0] = 0.0
        benchmark = None
        if (directory / "benchmark.csv").exists():
            benchmark_sessions, benchmark_values = read_curve(directory / "benchmark.csv")
            if benchmark_sessions != sessions:
                raise ValueError("benchmark.csv sessions do not match equity.csv")
            benchmark = _immutable(np.array(benchmark_values, dtype=np.float64))
    except (OSError, KeyError, ValueError, TypeError, ArithmeticError) as exc:
        raise ResearchError("RUN_ARTIFACT_INVALID", f"{run_id}: {exc}") from exc
    return Run(
        run_id,
        directory,
        _freeze(document),
        tuple(sessions),
        _immutable(equity),
        _immutable(returns),
        benchmark,
    )


def latest_run(strategy: str, project: Path | str | None = None) -> Run:
    """Load the most recently created run for a strategy."""
    root = _root(project)
    found = iter_runs(root, strategy_id=strategy)
    if not found:
        raise ResearchError("RUN_NOT_FOUND", strategy)
    return load_run(found[-1][0].name, root)


def list_sweeps(project: Path | str | None = None) -> tuple[Mapping[str, Any], ...]:
    """Recorded sweep documents, newest first."""
    return tuple(
        _freeze(document) for _, document in reversed(iter_runs(_root(project), "sweep.json", kind="sweeps"))
    )


def load_sweep(sweep_id: str, project: Path | str | None = None) -> Sweep:
    """Load a recorded sweep document."""
    directory, document = _document(_root(project), "sweeps", sweep_id, "sweep.json", "signalquarry.sweep/v1")
    return Sweep(sweep_id, directory, _freeze(document))


def load_comparison(comparison_id: str, project: Path | str | None = None) -> Mapping[str, Any]:
    """Load a recorded comparison document."""
    _, document = _document(
        _root(project), "comparisons", comparison_id, "comparison.json", "signalquarry.comparison/v1"
    )
    return _freeze(document)


def list_studies(project: Path | str | None = None) -> tuple[Mapping[str, Any], ...]:
    """Recorded study results, newest first: the run id, the study, its verdict and identity."""
    rows: list[Mapping[str, Any]] = []
    for directory, document in reversed(iter_runs(_root(project), "study.json", kind="studies")):
        outcome: Mapping[str, Any] = document.get("verdict") or {}
        rows.append(
            _freeze(
                {
                    "study_run_id": directory.name,
                    "study_id": document.get("study_id"),
                    "study_hash": document.get("study_hash"),
                    "created_at": document.get("created_at"),
                    "verdict": outcome.get("outcome"),
                }
            )
        )
    return tuple(rows)


def load_study(study_run_id: str, project: Path | str | None = None) -> Mapping[str, Any]:
    """Load one recorded study result (``study.json``), verified against its recorded hash."""
    _, document = _document(
        _root(project), "studies", study_run_id, "study.json", "signalquarry.study-result/v1"
    )
    if not result_hash_ok(document):
        raise ResearchError(
            "RUN_ARTIFACT_INVALID", f"{study_run_id}: study.json does not match its recorded hash"
        )
    return _freeze(document)


def latest_study(study_id: str, project: Path | str | None = None) -> Mapping[str, Any]:
    """The most recently recorded result of a study."""
    root = _root(project)
    for row in list_studies(root):
        if row["study_id"] == study_id:
            return load_study(str(row["study_run_id"]), root)
    raise ResearchError("RUN_NOT_FOUND", study_id)


__all__ = [
    "ResearchError",
    "Run",
    "Sweep",
    "latest_run",
    "latest_study",
    "list_runs",
    "list_studies",
    "list_sweeps",
    "load_comparison",
    "load_run",
    "load_study",
    "load_sweep",
]

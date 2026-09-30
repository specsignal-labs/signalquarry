# SPDX-License-Identifier: Apache-2.0
"""Disk-backed equity result rows consumed outside the pure backtest engine."""

from __future__ import annotations

import json
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from signalquarry._internal.canonical import canonical_json
from signalquarry._internal.engine.backtest import (
    BacktestEnd,
    BacktestTick,
    fill_ledger_row,
    ledger_hash_rows,
)


def _jsonl_rows(path: Path) -> Iterator[Any]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _file_chunks(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as stream:
        yield from stream


@dataclass(frozen=True)
class SpoolSummary:
    sessions: list[date]
    equity: list[Decimal]
    cash: list[Decimal]
    positions: dict[str, Decimal]
    warnings: list[str]
    dataset_identity: str
    ledger_hash: str


@dataclass(frozen=True)
class SpoolRun:
    result: SpoolSummary
    decisions_path: Path
    fills_path: Path
    fill_count: int
    fees: Decimal

    def decisions_chunks(self) -> Iterator[str]:
        return _file_chunks(self.decisions_path)

    def fills_csv_rows(self) -> Iterator[list[str]]:
        for row in _jsonl_rows(self.fills_path):
            yield [*row[:-1], row[-1] or ""]


def spool_equity_run(ticks: Generator[BacktestTick, None, BacktestEnd], scratch_dir: Path) -> SpoolRun:
    """Consume one pure backtest stream with bounded decision/fill memory."""
    decisions_path = scratch_dir / "decisions.jsonl"
    fills_path = scratch_dir / "fills.jsonl"
    sessions, equity, cash = [], [], []
    fill_count = 0
    fees = Decimal(0)
    try:
        with decisions_path.open("w", encoding="utf-8", newline="\n") as decisions_stream:
            with fills_path.open("w", encoding="utf-8", newline="\n") as fills_stream:
                while True:
                    try:
                        tick = next(ticks)
                    except StopIteration as completed:
                        final = completed.value
                        break
                    decisions_stream.write(canonical_json(tick.decision) + "\n")
                    for fill in tick.fills:
                        fills_stream.write(canonical_json(fill_ledger_row(fill)) + "\n")
                        fill_count += 1
                        fees += fill.fee
                    sessions.append(tick.session)
                    equity.append(tick.equity)
                    cash.append(tick.cash)
    except BaseException:
        ticks.close()
        decisions_path.unlink(missing_ok=True)
        fills_path.unlink(missing_ok=True)
        raise
    ledger_hash = ledger_hash_rows(
        final.dataset_identity,
        _jsonl_rows(decisions_path),
        ([s.isoformat(), e, c] for s, e, c in zip(sessions, equity, cash, strict=True)),
        _jsonl_rows(fills_path),
        final.positions,
    )
    result = SpoolSummary(
        sessions, equity, cash, final.positions, final.warnings, final.dataset_identity, ledger_hash
    )
    return SpoolRun(result, decisions_path, fills_path, fill_count, fees)

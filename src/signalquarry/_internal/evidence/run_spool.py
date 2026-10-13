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
from signalquarry._internal.evidence.runs import csv_chunks


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
    bought: Decimal = Decimal(0)  # total notional of buy fills
    sold: Decimal = Decimal(0)  # total notional of sell fills
    positions_path: Path | None = None
    positions_cash: list[list[Any]] | None = None

    def decisions_chunks(self) -> Iterator[str]:
        return _file_chunks(self.decisions_path)

    def fills_csv_rows(self) -> Iterator[list[str]]:
        for row in _jsonl_rows(self.fills_path):
            yield [*row[:-1], row[-1] or ""]

    def positions_chunks(self) -> Iterator[str]:
        assert self.positions_path is not None
        return _file_chunks(self.positions_path)


def spool_equity_run(ticks: Generator[BacktestTick, None, BacktestEnd], scratch_dir: Path) -> SpoolRun:
    """Consume one pure backtest stream with bounded decision/fill memory."""
    decisions_path = scratch_dir / "decisions.jsonl"
    fills_path = scratch_dir / "fills.jsonl"
    sessions, equity, cash = [], [], []
    fill_count = 0
    fees = Decimal(0)
    notional = {"buy": Decimal(0), "sell": Decimal(0)}
    positions_path: Path | None = None
    positions_cash: list[list[Any]] | None = None
    positions_stream = None
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
                    if tick.positions is not None:
                        if positions_stream is None:
                            positions_path = scratch_dir / "positions.csv"
                            positions_stream = positions_path.open("w", encoding="utf-8", newline="\n")
                            positions_stream.write("session,symbol,quantity,mark,value\n")
                            positions_cash = []
                        assert tick.positions_cash is not None and positions_cash is not None
                        positions_cash.append([tick.session.isoformat(), *tick.positions_cash])
                        chunks = csv_chunks(
                            [],
                            (
                                [tick.session.isoformat(), symbol, quantity, mark, quantity * mark]
                                for symbol, (quantity, mark) in sorted(tick.positions.items())
                            ),
                        )
                        next(chunks)  # the file's header was already written
                        positions_stream.writelines(chunks)
                    for fill in tick.fills:
                        fills_stream.write(canonical_json(fill_ledger_row(fill)) + "\n")
                        fill_count += 1
                        fees += fill.fee
                        notional[fill.side] += fill.quantity * fill.price
                    sessions.append(tick.session)
                    equity.append(tick.equity)
                    cash.append(tick.cash)
    except BaseException:
        ticks.close()
        decisions_path.unlink(missing_ok=True)
        fills_path.unlink(missing_ok=True)
        if positions_path is not None:
            positions_path.unlink(missing_ok=True)
        raise
    finally:
        if positions_stream is not None:
            positions_stream.close()
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
    return SpoolRun(
        result,
        decisions_path,
        fills_path,
        fill_count,
        fees,
        notional["buy"],
        notional["sell"],
        positions_path,
        positions_cash,
    )

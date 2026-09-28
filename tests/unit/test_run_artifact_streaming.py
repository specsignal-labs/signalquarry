# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from signalquarry._internal.canonical import file_sha256
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.backtest import (
    BacktestTick,
    EngineError,
    iter_equity_backtest,
    run_backtest,
)
from signalquarry._internal.engine.run import simulate_equity_ticks
from signalquarry._internal.evidence.run_spool import spool_equity_run
from signalquarry._internal.evidence.runs import csv_chunks, csv_text, jsonl_chunks, jsonl_text, write_run
from signalquarry.sdk import definition_of
from tests.helpers import spec
from tests.unit.test_engine import SmaP, sma_trend


def test_run_artifacts_stream_with_exact_csv_and_jsonl_bytes(tmp_path: Path) -> None:
    decisions = ({"session": session, "weight": Decimal("0.125")} for session in ("D1", "D2"))
    artifacts = write_run(
        tmp_path,
        "run-1",
        {
            "result.json": "{}\n",
            "equity.csv": csv_chunks(
                ["session", "note", "equity"],
                (("D1", "comma,name", Decimal("10.25")), ("D2", "plain", Decimal("11.00"))),
            ),
            "decisions.jsonl": jsonl_chunks(decisions),
            "raw.bin": b"\x00\xff",
        },
    )
    run_dir = tmp_path / ".signalquarry" / "runs" / "run-1"
    assert (run_dir / "equity.csv").read_bytes() == (
        b'session,note,equity\nD1,"comma,name",10.25\nD2,plain,11\n'
    )
    assert (run_dir / "decisions.jsonl").read_bytes() == (
        b'{"session":"D1","weight":"0.125"}\n{"session":"D2","weight":"0.125"}\n'
    )
    assert (run_dir / "raw.bin").read_bytes() == b"\x00\xff"
    assert [item["sha256"] for item in artifacts] == [
        file_sha256(run_dir / name) for name in ("result.json", "equity.csv", "decisions.jsonl", "raw.bin")
    ]


def test_failed_stream_does_not_leave_a_partial_run(tmp_path: Path) -> None:
    def broken() -> object:
        yield "first line\n"
        raise RuntimeError("interrupted producer")

    with pytest.raises(RuntimeError, match="interrupted producer"):
        write_run(tmp_path, "run-2", {"result.json": "{}\n", "decisions.jsonl": broken()})
    assert not (tmp_path / ".signalquarry" / "runs" / "run-2").exists()


def test_disk_spool_preserves_in_memory_equity_ledger(tmp_path: Path) -> None:
    dataset = synthetic_dataset(date(2022, 1, 3), date(2023, 12, 29), symbols=("SYNA",))
    strategy = spec(("SYNA",))
    definition = definition_of(sma_trend)
    params = SmaP()
    reference = run_backtest(strategy, definition, params, dataset)
    spool = spool_equity_run(iter_equity_backtest(strategy, definition, params, dataset), tmp_path)

    assert spool.result.ledger_hash == reference.ledger_hash
    assert spool.result.sessions == reference.sessions
    assert spool.result.equity == reference.equity
    assert spool.result.cash == reference.cash
    assert spool.result.positions == reference.positions
    assert spool.result.warnings == reference.warnings
    assert spool.fill_count == len(reference.fills)
    assert spool.fill_count > 0
    assert spool.fees == sum((fill.fee for fill in reference.fills), Decimal(0))
    assert spool.decisions_path.read_text(encoding="utf-8") == jsonl_text(reference.decisions)
    header = ["session", "symbol", "side", "quantity", "price", "fee", "settle_session"]
    assert "".join(csv_chunks(header, spool.fills_csv_rows())) == csv_text(
        header,
        [
            [
                fill.session.isoformat(),
                fill.symbol,
                fill.side,
                fill.quantity,
                fill.price,
                fill.fee,
                fill.settle_session.isoformat() if fill.settle_session else "",
            ]
            for fill in reference.fills
        ],
    )


def test_equity_stream_rejects_options_specs() -> None:
    options_spec = spec(("SYNA",)).model_copy(update={"kind": "options_single_leg"})
    dataset = synthetic_dataset(date(2022, 1, 3), date(2022, 1, 4), symbols=("SYNA",))
    with pytest.raises(EngineError, match="STRATEGY_KIND_MISMATCH"):
        simulate_equity_ticks(options_spec, definition_of(sma_trend), SmaP(), dataset)


def test_failed_engine_stream_removes_private_spool(tmp_path: Path) -> None:
    def broken_ticks() -> object:
        yield BacktestTick(date(2024, 1, 3), {"session": "2024-01-03"}, (), Decimal(1), Decimal(1))
        raise EngineError("BROKEN_TEST_STREAM")

    with pytest.raises(EngineError, match="BROKEN_TEST_STREAM"):
        spool_equity_run(broken_ticks(), tmp_path)
    assert list(tmp_path.iterdir()) == []

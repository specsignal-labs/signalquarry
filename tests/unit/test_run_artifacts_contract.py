# SPDX-License-Identifier: Apache-2.0
"""Run identity, on-disk layout, cleanup after failure, and the shape of result documents."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.evidence.runs import (
    csv_chunks,
    csv_text,
    jsonl_chunks,
    jsonl_text,
    new_run_id,
    result_document,
    unique_run_id,
    write_run,
)

HASH = "sha256:" + "0123456789abcdef" * 4


def test_a_run_id_is_the_utc_second_and_the_first_eight_hash_characters() -> None:
    assert new_run_id(HASH, datetime(2026, 9, 29, 14, 5, 9, tzinfo=UTC)) == "20260929T140509Z-01234567"
    assert new_run_id("ab" * 32, datetime(2026, 1, 2, 3, 4, 5)) == "20260102T030405Z-abababab"
    assert new_run_id(HASH, datetime(2026, 12, 31, 23, 59, 59, 999999)) == "20261231T235959Z-01234567"


def test_a_taken_run_id_gets_the_next_free_numeric_suffix(tmp_path: Path) -> None:
    runs = tmp_path / ".signalquarry" / "runs"
    assert unique_run_id(tmp_path, "run") == "run"
    (runs / "run").mkdir(parents=True)
    assert unique_run_id(tmp_path, "run") == "run-2"
    (runs / "run-2").mkdir()
    (runs / "run-3").mkdir()
    assert unique_run_id(tmp_path, "run") == "run-4"
    assert unique_run_id(tmp_path, "other") == "other"
    (runs / "run-5").mkdir()
    assert unique_run_id(tmp_path, "run") == "run-4"


def test_write_run_lays_out_files_and_describes_them_by_hash_and_kind(tmp_path: Path) -> None:
    artifacts = write_run(
        tmp_path,
        "r1",
        {
            "result.json": '{"a": 1}\n',
            "fills.csv": [b"head\n", "row\n"],
            "decisions.jsonl": iter(["x\n", b"y\n"]),
            "blob.bin": b"\x00\xff",
        },
    )
    run_dir = tmp_path / ".signalquarry" / "runs" / "r1"
    assert sorted(item.name for item in run_dir.iterdir()) == [
        "blob.bin",
        "decisions.jsonl",
        "fills.csv",
        "result.json",
    ]
    assert (run_dir / "fills.csv").read_bytes() == b"head\nrow\n"
    assert (run_dir / "decisions.jsonl").read_bytes() == b"x\ny\n"
    assert [item["kind"] for item in artifacts] == ["result", "fills", "decisions", "blob"]
    assert [item["path"] for item in artifacts] == [
        f".signalquarry/runs/r1/{name}"
        for name in ("result.json", "fills.csv", "decisions.jsonl", "blob.bin")
    ]
    for item in artifacts:
        assert set(item) == {"path", "sha256", "kind"}
        assert hashlib.sha256((tmp_path / item["path"]).read_bytes()).hexdigest() == item[
            "sha256"
        ].removeprefix("sha256:")


def test_text_chunks_are_written_as_utf8(tmp_path: Path) -> None:
    write_run(tmp_path, "r1", {"note.txt": "café ✓"})
    assert (tmp_path / ".signalquarry/runs/r1/note.txt").read_bytes() == "café ✓".encode()


def test_a_run_directory_cannot_be_reused(tmp_path: Path) -> None:
    write_run(tmp_path, "r1", {"a.json": "{}"})
    with pytest.raises(FileExistsError):
        write_run(tmp_path, "r1", {"b.json": "{}"})
    assert (tmp_path / ".signalquarry/runs/r1/a.json").exists()
    assert not (tmp_path / ".signalquarry/runs/r1/b.json").exists()


def test_a_failing_artifact_removes_the_whole_partial_run_and_reraises(tmp_path: Path) -> None:
    def broken():  # type: ignore[no-untyped-def]
        yield "first\n"
        raise RuntimeError("mid-stream failure")

    with pytest.raises(RuntimeError, match="mid-stream failure"):
        write_run(tmp_path, "r1", {"ok.json": "{}", "broken.csv": broken()})
    assert not (tmp_path / ".signalquarry/runs/r1").exists()
    assert (tmp_path / ".signalquarry/runs").is_dir()
    assert write_run(tmp_path, "r1", {"ok.json": "{}"})[0]["kind"] == "ok"


def test_a_failing_first_artifact_also_cleans_up(tmp_path: Path) -> None:
    with pytest.raises(AttributeError):
        write_run(tmp_path, "r2", {"bad.json": [123]})  # type: ignore[list-item]
    assert not (tmp_path / ".signalquarry/runs/r2").exists()


def test_csv_rows_use_unix_newlines_quote_text_and_canonicalise_values() -> None:
    text = csv_text(["a", "b"], [["x,y", 'q"t'], ["plain", 1.5], ["", None]])
    assert text == 'a,b\n"x,y","q""t"\nplain,1.5\n,\n'
    chunks = list(csv_chunks(["h"], [["one"], ["two"]]))
    assert chunks == ["h\n", "one\n", "two\n"]
    assert list(csv_chunks(["h"], [])) == ["h\n"]
    assert csv_text(["h"], []) == "h\n"
    from datetime import date
    from decimal import Decimal

    typed = csv_text(["d", "n", "t"], [[date(2026, 1, 2), Decimal("1.50"), True]])
    assert typed == "d,n,t\n2026-01-02,1.5,True\n"


def test_jsonl_rows_are_canonical_one_per_line() -> None:
    rows = [{"b": 1, "a": [2, 3]}, {"z": None}]
    assert jsonl_text(rows) == '{"a":[2,3],"b":1}\n{"z":null}\n'
    assert list(jsonl_chunks(iter(rows))) == ['{"a":[2,3],"b":1}\n', '{"z":null}\n']
    assert jsonl_text([]) == ""


def test_a_result_document_hashes_everything_except_its_timestamp() -> None:
    first = json.loads(result_document(kind="backtest", metrics={"sharpe": 1.25}))
    second = json.loads(result_document(kind="backtest", metrics={"sharpe": 1.25}))
    assert first["schema"] == "signalquarry.result/v1"
    assert first["kind"] == "backtest" and first["metrics"] == {"sharpe": 1.25}
    assert first["created_at"].endswith("Z") and datetime.fromisoformat(
        first["created_at"].replace("Z", "+00:00")
    )
    assert first["result_hash"] == second["result_hash"]
    expected = canonical_hash(
        {"schema": "signalquarry.result/v1", "kind": "backtest", "metrics": {"sharpe": 1.25}}
    )
    assert first["result_hash"] == expected
    assert json.loads(result_document(kind="other", metrics={"sharpe": 1.25}))["result_hash"] != expected


def test_a_result_document_is_indented_sorted_utf8_with_a_trailing_newline() -> None:
    text = result_document(name="café", z=1, a=2)
    assert text.endswith("}\n") and "café" in text and "\\u" not in text
    assert text.startswith('{\n  "a": 2,\n  "created_at"')
    keys = list(json.loads(text))
    assert keys == sorted(keys)


def test_a_caller_field_may_not_silently_change_the_schema_label() -> None:
    document = json.loads(result_document(schema="signalquarry.result/v9"))
    assert document["schema"] == "signalquarry.result/v9"
    assert document["result_hash"] == canonical_hash({"schema": "signalquarry.result/v9"})

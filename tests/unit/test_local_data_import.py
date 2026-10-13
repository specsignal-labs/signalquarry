# SPDX-License-Identifier: Apache-2.0
"""Strict local CSV refusals, replay and compatibility without provider calls."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from signalquarry import api
from signalquarry._internal.canonical import hash_without
from signalquarry._internal.data import local
from signalquarry._internal.data.alpaca import RawPage
from signalquarry._internal.data.library import (
    Library,
    LibraryError,
    build_dataset,
    dataset_from_manifest,
    select_manifest,
)
from signalquarry._internal.evidence.report import render_report
from signalquarry._internal.validation.evaluate import Evaluation, claim_level

HEADER = ",".join(local.BARS_HEADER) + "\n"
ACTION_HEADER = ",".join(local.ACTIONS_HEADER) + "\n"
BAR = "2024-01-02,SPY,123.456789,125,122,124,1000000\n"


def snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def page(body: bytes, endpoint: str = local.BARS_ENDPOINT) -> RawPage:
    return RawPage(endpoint, {}, body, hashlib.sha256(body).hexdigest(), {})


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SIGNALQUARRY_OFFLINE", "1")
    project = tmp_path / "project"
    assert api.init(project, package="import_validation").status == "ok"
    return project


@pytest.mark.parametrize(
    "role,body,code,line",
    [
        ("bars", b"", "EMPTY", 1),
        ("bars", HEADER.encode(), "EMPTY", 1),
        ("bars", BAR.encode(), "HEADER_INVALID", 1),
        ("bars", (HEADER.replace("session", "date") + BAR).encode(), "HEADER_INVALID", 1),
        ("bars", (HEADER + BAR.replace("124,", "oops,")).encode(), "VALUE_INVALID", 2),
        ("bars", (HEADER + BAR.replace("123.456789", "123.4567890")).encode(), "PRECISION", 2),
        ("bars", (HEADER + BAR.replace("123.456789", "0")).encode(), "PRICE_NONPOSITIVE", 2),
        ("bars", (HEADER + BAR.replace("123.456789", "-1")).encode(), "PRICE_NONPOSITIVE", 2),
        ("bars", (HEADER + BAR.replace("125,122", "121,122")).encode(), "OHLC_INVALID", 2),
        ("bars", (HEADER + BAR.replace("125,122", "123,122")).encode(), "OHLC_INVALID", 2),
        ("bars", (HEADER + BAR.replace("125,122", "125,124")).encode(), "OHLC_INVALID", 2),
        ("bars", (HEADER + BAR.replace("124,", "126,")).encode(), "OHLC_INVALID", 2),
        ("bars", (HEADER + BAR + BAR).encode(), "DUPLICATE", 3),
        ("bars", (HEADER + BAR.replace("2024-01-02", "2024-01-01")).encode(), "SESSION_INVALID", 2),
        ("bars", (HEADER + BAR.replace("2024-01-02", "2024-01-06")).encode(), "SESSION_INVALID", 2),
        ("bars", (HEADER + BAR.replace("2024-01-02", "2015-01-02")).encode(), "SESSION_INVALID", 2),
        ("bars", (HEADER + BAR.replace("2024-01-02", "2036-01-02")).encode(), "SESSION_INVALID", 2),
        ("bars", (HEADER + BAR.replace("2024-01-02", "2024-02-30")).encode(), "VALUE_INVALID", 2),
        ("bars", (HEADER + BAR.replace("2024-01-02", "20240102")).encode(), "VALUE_INVALID", 2),
        ("bars", (HEADER + BAR.replace("SPY", "spy")).encode(), "VALUE_INVALID", 2),
        ("bars", (HEADER + BAR.replace("1000000", "-1")).encode(), "VALUE_INVALID", 2),
        ("bars", (HEADER + BAR.replace("1000000", "NaN")).encode(), "VALUE_INVALID", 2),
        ("bars", (HEADER + BAR.replace("1000000", "abc")).encode(), "VALUE_INVALID", 2),
        ("bars", (HEADER + BAR.replace("1000000", "-1e-400")).encode(), "VALUE_INVALID", 2),
        ("bars", (HEADER + BAR.replace("1000000", "9" * 400)).encode(), "VALUE_INVALID", 2),
        ("bars", (HEADER + BAR.replace("123.456789", "99999999999999")).encode(), "VALUE_INVALID", 2),
        ("bars", (HEADER + BAR.replace(",1000000", "")).encode(), "VALUE_INVALID", 2),
        ("bars", HEADER.encode() + b"\xff\n", "VALUE_INVALID", 2),
        ("bars", (HEADER + '"unterminated').encode(), "VALUE_INVALID", 2),
        ("actions", b"", "EMPTY", 1),
        ("actions", b"symbol,type\n", "HEADER_INVALID", 1),
        ("actions", (ACTION_HEADER + "QQQ,split,2024-01-02,,2,\n").encode(), "ACTION_SYMBOL", 2),
        ("actions", (ACTION_HEADER + "SPY,merger,2024-01-02,,2,\n").encode(), "VALUE_INVALID", 2),
        ("actions", (ACTION_HEADER + "SPY,split,2024-01-02,,0,\n").encode(), "VALUE_INVALID", 2),
        ("actions", (ACTION_HEADER + "SPY,split,2024-01-02,,-1,\n").encode(), "VALUE_INVALID", 2),
        ("actions", (ACTION_HEADER + "SPY,split,2024-01-02,,,\n").encode(), "VALUE_INVALID", 2),
        ("actions", (ACTION_HEADER + "SPY,split,2024-01-02,,1e-400,\n").encode(), "VALUE_INVALID", 2),
        (
            "actions",
            (ACTION_HEADER + "SPY,dividend,2024-01-02,2024-01-03,,-1\n").encode(),
            "VALUE_INVALID",
            2,
        ),
        (
            "actions",
            (ACTION_HEADER + "SPY,split,2024-01-02,," + "9" * 400 + ",\n").encode(),
            "VALUE_INVALID",
            2,
        ),
        ("actions", (ACTION_HEADER + "SPY,split,2024-01-02,2024-01-03,2,\n").encode(), "VALUE_INVALID", 2),
        ("actions", (ACTION_HEADER + "SPY,split,2024-01-02,,2,1\n").encode(), "VALUE_INVALID", 2),
        ("actions", (ACTION_HEADER + "SPY,split,2024-01-02,,2,\n" * 2).encode(), "DUPLICATE", 3),
        ("actions", (ACTION_HEADER + "SPY,dividend,2024-01-02,,,1\n").encode(), "VALUE_INVALID", 2),
        ("actions", (ACTION_HEADER + "SPY,dividend,2024-01-02,2024-01-03,,\n").encode(), "VALUE_INVALID", 2),
        ("actions", (ACTION_HEADER + "SPY,dividend,2024-01-02,2024-01-01,,1\n").encode(), "VALUE_INVALID", 2),
        (
            "actions",
            (ACTION_HEADER + "SPY,dividend,2024-01-02,2024-01-03,2,1\n").encode(),
            "VALUE_INVALID",
            2,
        ),
        ("actions", (ACTION_HEADER + "SPY,dividend,2024-01-02,2024-01-03,,1\n" * 2).encode(), "DUPLICATE", 3),
    ],
)
def test_import_refusal_names_line_and_writes_nothing(
    project: Path, tmp_path: Path, role: str, body: bytes, code: str, line: int
) -> None:
    bars, actions = tmp_path / "bars.csv", tmp_path / "actions.csv"
    bars.write_bytes(body if role == "bars" else (HEADER + BAR).encode())
    actions.write_bytes(body if role == "actions" else ACTION_HEADER.encode())
    before = snapshot(tmp_path)
    result = api.data_import(
        bars=bars, actions=actions, source="Test source", adjustment="raw", project=project
    )
    assert (result.status, result.reason_codes) == ("invalid", [f"DATA_IMPORT_{code}"])
    assert f"{role} line {line}:" in result.summary
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"actions": Path("a"), "no_actions": True},
        {"actions": Path("a"), "adjustment": "adjusted"},
        {"no_actions": True, "adjustment": "other"},
        {"no_actions": True, "source": "  "},
    ],
)
def test_import_usage_errors_write_nothing(project: Path, tmp_path: Path, kwargs: dict) -> None:
    before = snapshot(tmp_path)
    result = api.data_import(
        **{"bars": Path("b"), "source": "Source", "adjustment": "raw", "project": project, **kwargs}
    )
    assert (result.status, result.reason_codes) == ("usage", ["USAGE_INVALID"])
    assert snapshot(tmp_path) == before


def test_import_unreadable_and_missing_project(project: Path, tmp_path: Path) -> None:
    assert api.data_import(
        bars=tmp_path / "missing", source="Source", adjustment="raw", no_actions=True, project=project
    ).reason_codes == ["DATA_IMPORT_FILE_UNREADABLE"]
    assert (
        api.data_import(
            bars=tmp_path / "missing",
            source="Source",
            adjustment="raw",
            no_actions=True,
            project=tmp_path / "absent",
        ).status
        == "invalid"
    )


def test_import_idempotence_bytes_metadata_quality_and_listing(project: Path, tmp_path: Path) -> None:
    bars, actions = tmp_path / "bars.csv", tmp_path / "actions.csv"
    bars.write_bytes((HEADER + BAR.replace("1000000", "0")).replace("\n", "\r\n").encode())
    actions.write_text(ACTION_HEADER)
    kwargs = {
        "bars": bars,
        "actions": actions,
        "source": "  Original source\nverbatim  ",
        "license": " custom licence ",
        "adjustment": "raw",
        "project": project,
    }
    first = api.data_import(**kwargs)
    assert first.status == "ok" and "QUALITY_ZERO_VOLUME" in first.warnings
    library = api.data.library_for(project)
    manifest = library.manifests()[0]
    assert manifest["source"] == kwargs["source"] and manifest["license"] == kwargs["license"]
    assert manifest["actions_supplied"] and manifest["evidence_grade"] == "historical"
    assert manifest["imported_at"] == manifest["fetched_at"]
    assert (
        gzip.decompress(library.page_path(manifest["bar_pages"][0]["sha256"]).read_bytes())
        == bars.read_bytes()
    )
    assert (
        gzip.decompress(library.page_path(manifest["action_pages"][0]["sha256"]).read_bytes())
        == actions.read_bytes()
    )
    assert "123.456789" not in json.dumps(manifest)
    before = snapshot(tmp_path)
    second = api.data_import(**kwargs)
    assert second.data == first.data and snapshot(tmp_path) == before
    assert api.data_verify(project=project).status == "ok"
    row = api.data_ls(project=project).data["datasets"][0]
    assert (row["provider"], row["adjustment"], row["evidence_grade"]) == ("local", "raw", "historical")
    rebuilt = dataset_from_manifest(library, manifest)
    assert rebuilt.price("SPY", "open", 0) == Decimal("123.456789")
    cached = library.page_path(manifest["bar_pages"][0]["sha256"])
    cached.write_bytes(gzip.compress(b"changed", mtime=0))
    assert api.data_verify(project=project).reason_codes == ["DATA_PAGE_CORRUPT"]
    before = snapshot(tmp_path)
    assert api.data_import(**kwargs).reason_codes == ["DATA_PAGE_CORRUPT"]
    assert snapshot(tmp_path) == before
    assert api.data_import(**{**kwargs, "source": "Different declaration"}).reason_codes == [
        "DATA_PAGE_CORRUPT"
    ]
    assert snapshot(tmp_path) == before


def test_import_rows_sort_absent_bars_and_dated_actions() -> None:
    bars = page(
        (HEADER + "2024-01-03,SPY,50,51,49,50,100\n" + BAR + "2024-01-02,QQQ,10,11,9,10,1.5\n").encode()
    )
    actions = page(
        (ACTION_HEADER + "SPY,dividend,2024-01-03,2024-01-05,,0.5\nSPY,split,2024-01-03,,2,\n").encode(),
        local.ACTIONS_ENDPOINT,
    )
    result = build_dataset([bars], [actions], source="local:local", parser=local.PARSER)
    assert result.sessions == (date(2024, 1, 2), date(2024, 1, 3))
    assert result.symbols == ("QQQ", "SPY") and result.price("QQQ", "close", 1) is None
    assert result.splits[0].ratio == 2 and result.dividends[0].pay_date == date(2024, 1, 5)
    assert result.series["QQQ"].volume[0] == 1.5
    with pytest.raises(LibraryError, match="unknown parser"):
        build_dataset([bars], [], source="local:local", parser="unknown")
    with pytest.raises(local.ImportError, match="line 1"):
        local.bars_from_pages([])


def test_import_decimal_notation_and_nonprice_numbers() -> None:
    bars = page((HEADER + "2024-01-02,SPY,.500000,1.,+0.25,0.5,1e3\n").encode())
    actions = page(
        (
            ACTION_HEADER + "SPY,split,2024-01-02,,2e0,\nSPY,dividend,2024-01-03,2024-01-03,,0.1234567\n"
        ).encode(),
        local.ACTIONS_ENDPOINT,
    )
    result = build_dataset([bars], [actions], source="local:local", parser=local.PARSER)
    assert result.price("SPY", "open", 0) == Decimal("0.5")
    assert result.series["SPY"].volume[0] == 1000
    assert result.dividends[0].amount == Decimal("0.1234567")


def test_import_latest_selection_compares_instants_not_variable_fraction_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = {
        "feed": "local",
        "symbols": ["SPY"],
        "imported_at": "2026-10-13T12:00:00Z",
        "dataset_id": "first",
    }
    latest = {**first, "imported_at": "2026-10-13T12:00:00.000001Z", "dataset_id": "latest"}
    monkeypatch.setattr(Library, "manifests", lambda self: [first, latest])
    library = Library(tmp_path / "cache", tmp_path / "manifests")
    assert select_manifest(library, ("SPY",), "local") == latest


@pytest.mark.parametrize("grade", ["historical", "historical_adjusted", "synthetic", "low_evidence_options"])
@pytest.mark.parametrize("frozen,conformance", [(True, True), (False, True), (True, False)])
def test_import_claim_ceiling_even_with_every_gate_passing(
    grade: str, frozen: bool, conformance: bool
) -> None:
    evaluation = Evaluation(
        gates={name: {"ok": True} for name in ("G1_sample", "G2_walk_forward", "G3_stress", "G4_holdout")}
    )
    result = claim_level(evaluation, grade=grade, frozen=frozen, conformance_ok=conformance)
    expected = (
        "none"
        if grade == "synthetic" or not conformance
        else "in_sample"
        if not frozen or grade == "historical_adjusted"
        else "walk_forward"
        if grade == "low_evidence_options"
        else "holdout_passed"
    )
    assert result == expected


def test_claim_level_failing_plugin_and_unopened_holdout() -> None:
    gates = {name: {"ok": True} for name in ("G1_sample", "G2_walk_forward", "G3_stress")}
    assert (
        claim_level(Evaluation(gates=gates), grade="historical", frozen=True, conformance_ok=True)
        == "walk_forward"
    )
    gates["plugin:test"] = {"ok": False}
    assert (
        claim_level(Evaluation(gates=gates), grade="historical", frozen=True, conformance_ok=True)
        == "in_sample"
    )


def test_local_manifest_parser_and_identity_refusals(project: Path, tmp_path: Path) -> None:
    bars = tmp_path / "bars.csv"
    bars.write_text(HEADER + BAR)
    assert (
        api.data_import(bars=bars, source="Source", adjustment="raw", no_actions=True, project=project).status
        == "ok"
    )
    library = api.data.library_for(project)
    original = library.manifests()[0]
    for change, code in [
        ({"parser": "other"}, "DATA_MANIFEST_INVALID"),
        ({"dataset_identity": "wrong"}, "DATA_IDENTITY_MISMATCH"),
    ]:
        manifest = {**original, **change}
        manifest["manifest_hash"] = hash_without(manifest, "manifest_hash")
        with pytest.raises(LibraryError) as error:
            dataset_from_manifest(library, manifest)
        assert error.value.code == code
    bad = page(b"bad header\n")
    library.store_page(bad)
    manifest = {**original, "bar_pages": [{"endpoint": bad.endpoint, "params": {}, "sha256": bad.sha256}]}
    manifest["manifest_hash"] = hash_without(manifest, "manifest_hash")
    with pytest.raises(LibraryError) as error:
        dataset_from_manifest(library, manifest)
    assert error.value.code == "DATA_IMPORT_HEADER_INVALID"


@pytest.mark.parametrize(
    "grade,expected",
    [
        ("historical", "7f49758151d0882be1d14975d7862d49d54a4da6dfdf75d84cb714de28e3f55b"),
        ("synthetic", "04e8c5d1799dd45e9669f1aca3670ed00b67c3f16234b942ae9fa6c54b791c5c"),
        ("low_evidence_options", "c82c95a916a03e96cfebf7ab1923ae27a345f91e71c374d8796e93ccdaaefc92"),
    ],
)
def test_import_report_preserves_existing_grade_bytes(grade: str, expected: str) -> None:
    # Hashes recorded from the unchanged pre-import renderer.
    text = render_report(
        spec={
            "id": "reference",
            "version": "1.0.0",
            "family": "reference",
            "hypothesis": {"statement": "Reference hypothesis.", "falsification": "Reference failure."},
        },
        result={
            "run_id": "reference-run",
            "dataset_id": "reference-dataset",
            "evidence": {"grade": grade, "claim_level": "in_sample"},
            "metrics": {},
        },
        evaluation=None,
        trials={"family_count": 0, "project_count": 0},
        chart=None,
    )
    assert hashlib.sha256(text.encode()).hexdigest() == expected

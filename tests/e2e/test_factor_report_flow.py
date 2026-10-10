# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from signalquarry import api
from signalquarry._internal.canonical import file_sha256
from signalquarry.cli.main import main


def _sqy(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def project(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch) -> Path:
    root = tmp_path / "factor-report"
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(cache))
    code, result = _sqy(
        capsys,
        "init",
        str(root),
        "--demo",
        "--kind",
        "factor",
        "--package",
        "factor_report_" + uuid.uuid4().hex,
    )
    assert code == 0, result
    return root


def _files(root: Path) -> dict[str, bytes]:
    """Every file except Python's own bytecode caches, which importing the factor may write."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def test_factor_report_cli_and_api_are_descriptive_and_isolated(project, capsys) -> None:
    before = _files(project)
    evaluated = api.factor_evaluate(
        "volume-shock",
        synthetic=True,
        synthetic_symbols=30,
        synthetic_seed=9,
        synthetic_planted_ic=0.1,
        project=project,
    )
    assert evaluated.status == "ok"
    assert _files(project) == before
    code, reported = _sqy(
        capsys,
        "factor",
        "report",
        "--factor",
        "volume-shock",
        "--synthetic",
        "--symbols",
        "30",
        "--seed",
        "9",
        "--planted-ic",
        "0.1",
        "--project",
        str(project),
    )
    assert code == 0, reported
    assert reported["command"] == "factor report"
    assert reported["evidence"] == evaluated.evidence
    assert reported["data"]["scope"] == "synthetic"
    assert reported["data"]["horizons"] == [
        {"horizon": row["horizon"], "mean_ic": row["mean_ic"]} for row in evaluated.data["horizons"]
    ]
    out = project / ".signalquarry" / "factor_reports" / reported["data"]["report_id"]
    assert out.name.endswith("-volume-shock")
    assert {path.name for path in out.iterdir()} == {"report.md", "ic.svg", "quintiles.svg"}
    assert {item["kind"] for item in reported["artifacts"]} == {"report", "ic", "quintiles"}
    for artifact in reported["artifacts"]:
        path = project / artifact["path"]
        assert path.parent == out
        assert file_sha256(path) == artifact["sha256"]
    text = (out / "report.md").read_text()
    first = "\n".join(text.splitlines()[:4])
    assert "Scope: `synthetic`" in first
    assert "synthetic data: these numbers say nothing about real markets" in first
    assert "No trial was recorded, no evidence grade assigned and no holdout accessed" in first
    assert "demonstration on synthetic data" in text and "on purpose" in text
    assert "- symbols: 30\n- seed: 9\n- planted_ic: 0.1" in text
    table = text.split("## Diagnostics by horizon", 1)[1].split("## Chronological-block IC", 1)[0]
    rows = [line.split("|")[1:-1] for line in table.splitlines() if line.startswith("| ")][1:]
    assert len(rows) == len(evaluated.data["horizons"])
    for row, expected in zip(rows, evaluated.data["horizons"], strict=True):
        assert int(row[0].strip()) == expected["horizon"]
        assert int(row[1].strip()) == expected["observations"]
        assert row[2].strip() == f"{expected['mean_ic']:.4f}"
    for name in ("ic.svg", "quintiles.svg"):
        ET.fromstring((out / name).read_text())
    after_first = _files(project)
    first_bytes = {path: path.read_bytes() for path in out.iterdir()}
    second = api.factor_report("volume-shock", synthetic=True, synthetic_symbols=30, project=project)
    assert second.status == "ok"
    assert second.data["report_id"] != reported["data"]["report_id"]
    assert all(path.read_bytes() == content for path, content in first_bytes.items())
    assert len(list(out.parent.iterdir())) == 2
    for snapshot in (after_first, _files(project)):
        assert {name: snapshot[name] for name in before} == before
        assert all(
            name.startswith(".signalquarry/factor_reports/") for name in snapshot.keys() - before.keys()
        )
    assert not list((project / "evidence").rglob("*.jsonl"))
    assert not list(project.rglob("*trial*"))
    assert not (project / ".signalquarry" / "runs").exists()


@pytest.mark.parametrize(
    ("factor_id", "flags", "status", "exit_code"),
    [
        ("unknown", ["--synthetic"], "invalid", 65),
        ("volume-shock", [], "usage", 64),
        ("volume-shock", ["--synthetic", "--dataset-id", "D"], "usage", 64),
        ("volume-shock", ["--synthetic", "--symbols", "1"], "usage", 64),
        ("volume-shock", ["--dataset-id", "missing"], "invalid", 65),
    ],
)
def test_report_preserves_evaluation_refusals(project, capsys, factor_id, flags, status, exit_code) -> None:
    before = _files(project)
    common = ["--factor", factor_id, "--project", str(project), *flags]
    eval_code, evaluated = _sqy(capsys, "factor", "evaluate", *common)
    code, reported = _sqy(capsys, "factor", "report", *common)
    assert code == eval_code == exit_code
    assert reported["command"] == "factor report" and reported["status"] == status
    for key in (
        "reason_codes",
        "summary",
        "data",
        "metrics",
        "evidence",
        "artifacts",
        "warnings",
        "next_actions",
    ):
        assert reported[key] == evaluated[key]
    assert _files(project) == before
    assert not (project / ".signalquarry").exists()

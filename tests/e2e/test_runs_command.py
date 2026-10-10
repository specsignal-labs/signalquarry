# SPDX-License-Identifier: Apache-2.0
"""``sqy runs ls | show | compare``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from signalquarry._internal.evidence.runs import result_hash_ok
from signalquarry.cli.main import main


def _sqy(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def lab(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[Path, dict[str, str]]:
    """A demo project with a base run, a variant, and the variant on a shorter window."""
    project = tmp_path / "runs-lab"
    _sqy(
        capsys,
        "init",
        str(project),
        "--demo",
        "--package",
        "runs_lab_" + tmp_path.name[-6:].lower().replace("-", "_"),
    )
    run = ("backtest", "--strategy", "sma-trend", "--project", str(project))
    ids = {}
    for label, extra in (
        ("base", ()),
        ("p100", ("--param", "period=100")),
        ("short", ("--param", "period=100", "--end", "2020-12-31")),
    ):
        code, payload = _sqy(capsys, *run, "--label", label, *extra)
        assert code == 0, payload
        ids[label] = payload["data"]["run_id"]
    return project, ids


def test_ls_lists_newest_first_with_filters(
    lab: tuple[Path, dict[str, str]], capsys: pytest.CaptureFixture[str]
) -> None:
    project, ids = lab
    code, payload = _sqy(capsys, "runs", "ls", "--project", str(project))
    assert code == 0 and payload["data"]["total"] == 3
    rows = payload["data"]["runs"]
    assert [row["run_id"] for row in rows] == [ids["short"], ids["p100"], ids["base"]]
    assert [row["label"] for row in rows] == ["short", "p100", "base"]
    assert rows[0]["end"] == "2020-12-31" and rows[2]["params"]["period"] == 200
    assert rows[1]["command"] == "backtest" and rows[1]["grade"] == "synthetic"

    code, limited = _sqy(capsys, "runs", "ls", "--limit", "1", "--project", str(project))
    assert [row["run_id"] for row in limited["data"]["runs"]] == [ids["short"]]
    assert limited["summary"] == "1 of 3 runs"
    code, other = _sqy(capsys, "runs", "ls", "--strategy", "nothing", "--project", str(project))
    assert other["data"] == {"runs": [], "total": 0}
    code, mine = _sqy(capsys, "runs", "ls", "--strategy", "sma-trend", "--project", str(project))
    assert mine["data"]["total"] == 3 and mine["summary"] == "3 of 3 runs of sma-trend"
    code, bad = _sqy(capsys, "runs", "ls", "--limit", "0", "--project", str(project))
    assert (code, bad["reason_codes"]) == (64, ["USAGE_INVALID"])


def test_show_returns_the_verified_document(
    lab: tuple[Path, dict[str, str]], capsys: pytest.CaptureFixture[str]
) -> None:
    project, ids = lab
    code, payload = _sqy(capsys, "runs", "show", ids["p100"], "--project", str(project))
    assert code == 0, payload
    result = payload["data"]["result"]
    assert result["run_id"] == ids["p100"] and result["label"] == "p100"
    assert result["params"]["period"] == 100 and "params" not in result["spec"]
    assert result_hash_ok(result)
    assert payload["metrics"] == result["metrics"] and payload["evidence"] == result["evidence"]
    assert {a["kind"] for a in payload["artifacts"]} == {
        "result",
        "equity",
        "benchmark",
        "fills",
        "decisions",
    }
    assert "[p100]" in payload["summary"]

    for missing in ("nope", "../" + ids["base"], ""):
        code, payload = _sqy(capsys, "runs", "show", missing, "--project", str(project))
        assert (code, payload["reason_codes"]) == (65, ["RUN_NOT_FOUND"]), missing


def test_an_edited_result_document_is_refused(
    lab: tuple[Path, dict[str, str]], capsys: pytest.CaptureFixture[str]
) -> None:
    project, ids = lab
    path = project / ".signalquarry" / "runs" / ids["p100"] / "result.json"
    document = json.loads(path.read_text())
    assert result_hash_ok(document)
    document["metrics"]["sharpe"] = 9.9
    assert not result_hash_ok(document)
    assert not result_hash_ok({**document, "metrics": float("nan")})
    path.write_text(json.dumps(document))
    code, payload = _sqy(capsys, "runs", "show", ids["p100"], "--project", str(project))
    assert (code, payload["reason_codes"]) == (65, ["RUN_ARTIFACT_INVALID"])
    code, payload = _sqy(capsys, "runs", "compare", ids["base"], ids["p100"], "--project", str(project))
    assert (code, payload["reason_codes"]) == (65, ["RUN_ARTIFACT_INVALID"])
    assert not (project / ".signalquarry" / "comparisons").exists()

    path.write_text("not json")
    code, payload = _sqy(capsys, "runs", "show", ids["p100"], "--project", str(project))
    assert payload["reason_codes"] == ["RUN_ARTIFACT_INVALID"]
    path.write_text(json.dumps({"schema": "something/else"}))
    code, payload = _sqy(capsys, "runs", "show", ids["p100"], "--project", str(project))
    assert payload["reason_codes"] == ["RUN_ARTIFACT_INVALID"]
    path.write_text("[]")
    code, payload = _sqy(capsys, "runs", "show", ids["p100"], "--project", str(project))
    assert payload["reason_codes"] == ["RUN_ARTIFACT_INVALID"]


def test_compare_differences_comparable_runs_and_writes_the_comparison(
    lab: tuple[Path, dict[str, str]], capsys: pytest.CaptureFixture[str]
) -> None:
    project, ids = lab
    code, payload = _sqy(capsys, "runs", "compare", ids["base"], ids["p100"], "--project", str(project))
    assert code == 0, payload
    data = payload["data"]
    assert data["comparable"] is True and data["reasons"] == [] and payload["warnings"] == []
    assert data["reference"] == ids["base"] and payload["summary"] == "2 runs against base"
    assert data["differences"] == {ids["p100"]: {"params": {"period": [200, 100]}, "spec": {}}}
    pair = data["pairs"][ids["p100"]]
    assert pair["sessions"] == data["runs"][0]["metrics"]["sessions"]
    assert 0 < pair["correlation"] < 1
    low, high = pair["sharpe_difference_90"]
    assert low < pair["sharpe_difference"] < high
    assert pair["total_return_difference"] == pytest.approx(
        data["runs"][1]["metrics"]["total_return"] - data["runs"][0]["metrics"]["total_return"]
    )
    assert payload["evidence"] == {"grade": "synthetic", "claim_level": "none"}

    directory = project / ".signalquarry" / "comparisons" / data["comparison_id"]
    assert {p.name for p in directory.iterdir()} == {"comparison.json", "comparison.md", "equity.svg"}
    document = json.loads((directory / "comparison.json").read_text())
    assert document["schema"] == "signalquarry.comparison/v1" and document["pairs"] == data["pairs"]
    text = (directory / "comparison.md").read_text()
    assert "# Comparison of 2 runs" in text and "| base (reference) | sma-trend |" in text
    assert "- **p100**: parameter `period`: `200` → `100`" in text
    assert "## Against the reference" in text and "Synthetic data" in text
    assert "Hypothetical, simulated results" in text and "Not comparable" not in text
    svg = (directory / "equity.svg").read_text()
    assert svg.count("<polyline") == 2 and ">base<" in svg and ">p100<" in svg

    # The same comparison again gets its own directory; the numbers are identical.
    code, again = _sqy(capsys, "runs", "compare", ids["base"], ids["p100"], "--project", str(project))
    assert again["data"]["pairs"] == data["pairs"]
    assert again["data"]["comparison_id"] != data["comparison_id"]


def test_compare_lists_but_does_not_difference_runs_on_different_windows(
    lab: tuple[Path, dict[str, str]], capsys: pytest.CaptureFixture[str]
) -> None:
    project, ids = lab
    code, payload = _sqy(
        capsys, "runs", "compare", ids["base"], ids["p100"], ids["short"], "--project", str(project)
    )
    assert code == 0, payload
    assert payload["warnings"] == ["RUNS_NOT_COMPARABLE"]
    assert payload["data"]["comparable"] is False and payload["data"]["pairs"] == {}
    assert payload["data"]["reasons"] == ["COMPARE_WINDOW_DIFFERS"]
    assert set(payload["data"]["differences"]) == {ids["p100"], ids["short"]}
    assert "not comparable: COMPARE_WINDOW_DIFFERS" in payload["summary"]
    directory = project / ".signalquarry" / "comparisons" / payload["data"]["comparison_id"]
    text = (directory / "comparison.md").read_text()
    assert "**Not comparable** (COMPARE_WINDOW_DIFFERS)" in text
    assert "## Against the reference" not in text and "| short | sma-trend |" in text
    # Curves of different lengths are not drawn on one chart.
    assert (directory / "equity.svg").read_text().count("<polyline") == 2


def test_compare_usage_and_missing_artifacts(
    lab: tuple[Path, dict[str, str]], capsys: pytest.CaptureFixture[str]
) -> None:
    project, ids = lab
    code, payload = _sqy(capsys, "runs", "compare", ids["base"], "--project", str(project))
    assert (code, payload["reason_codes"]) == (64, ["USAGE_INVALID"])
    code, payload = _sqy(capsys, "runs", "compare", ids["base"], ids["base"], "--project", str(project))
    assert (code, payload["reason_codes"]) == (64, ["USAGE_INVALID"])
    code, payload = _sqy(capsys, "runs", "compare", ids["base"], "nope", "--project", str(project))
    assert (code, payload["reason_codes"]) == (65, ["RUN_NOT_FOUND"])

    equity = project / ".signalquarry" / "runs" / ids["p100"] / "equity.csv"
    kept = equity.read_text()
    equity.write_text("\n".join(kept.splitlines()[:-1]) + "\n")  # one session short
    code, payload = _sqy(capsys, "runs", "compare", ids["base"], ids["p100"], "--project", str(project))
    assert (code, payload["reason_codes"]) == (65, ["RUN_ARTIFACT_INVALID"])
    assert "does not match result.json" in payload["summary"]
    equity.unlink()
    code, payload = _sqy(capsys, "runs", "compare", ids["base"], ids["p100"], "--project", str(project))
    assert (code, payload["reason_codes"]) == (65, ["RUN_ARTIFACT_INVALID"])
    assert "missing or unreadable" in payload["summary"]


def test_runs_outside_a_project(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for action in (("ls",), ("show", "x"), ("compare", "a", "b")):
        code, payload = _sqy(capsys, "runs", *action, "--project", str(tmp_path))
        assert (code, payload["reason_codes"]) == (65, ["PROJECT_NOT_FOUND"]), action


def test_the_latest_run_is_the_one_created_last_even_within_one_second(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Run ids carry a timestamp to the second, followed by the configuration hash. Several
    # runs inside one second must still be ordered by when they were made, not by that hash.
    project = tmp_path / "order-lab"
    _sqy(capsys, "init", str(project), "--demo", "--package", "order_lab_runs")
    run = ("backtest", "--strategy", "sma-trend", "--project", str(project))
    made = []
    for period in (150, 60, 200, 90, 120, 30):
        code, payload = _sqy(capsys, *run, "--param", f"period={period}")
        assert code == 0, payload
        made.append(payload["data"]["run_id"])
    code, listing = _sqy(capsys, "runs", "ls", "--project", str(project))
    assert [row["run_id"] for row in listing["data"]["runs"]] == made[::-1]

    # The report follows the latest run and says it is not the declared configuration.
    code, report = _sqy(capsys, "report", "--strategy", "sma-trend", "--project", str(project))
    assert report["data"]["backtest_run"] == made[-1]
    assert "REPORT_CONFIGURATION_CHANGED" in report["warnings"]

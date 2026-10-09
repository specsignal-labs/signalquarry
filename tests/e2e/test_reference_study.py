# SPDX-License-Identifier: Apache-2.0
"""The worked study of the research guide: the example file runs and the guide quotes its numbers."""

from __future__ import annotations

from pathlib import Path

from signalquarry.api import init, study_check, study_run
from signalquarry.api.docs import guide

ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / "examples" / "trend_vs_hold_study" / "study.yaml"


def test_the_guide_shows_the_example_study_and_its_actual_results(tmp_path: Path) -> None:
    text = guide("research")
    assert EXAMPLE.read_text() in text  # the file in the guide is the file that is run

    project = tmp_path / "reference-study"
    assert init(project, demo=True, package="reference_study_lab").status == "ok"
    target = project / "studies" / "trend-vs-hold" / "study.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(EXAMPLE.read_text())

    checked = study_check("trend-vs-hold", project=project)
    assert checked.status == "ok" and len(checked.data["arms"]) == 7
    assert sum(arm["counts"] for arm in checked.data["arms"]) == 4  # "on real data, 4 would be trials"

    result = study_run("trend-vs-hold", project=project)
    assert result.status == "ok"
    assert result.data["verdict"]["outcome"] == "not_supported"
    assert result.evidence == {"grade": "synthetic", "claim_level": "none"}
    roles = {"base": "subject", "buy-and-hold": "baseline", "vol-matched": "baseline"}
    for arm in result.data["arms"]:
        row = (
            f"| {arm['id']} | {roles.get(arm['id'], arm['role'])} | {arm['total_return']:.2%} | "
            f"{arm['sharpe']:.2f} | {arm['max_drawdown']:.2%} |"
        )
        assert row in text, row
    assert f"overfitting of {result.data['pbo']['pbo']:.2f}" in text
    assert result.data["pbo"]["configurations"] == 4

    by_id = {arm["id"]: arm for arm in result.data["arms"]}
    # The statements the guide makes about the table.
    assert by_id["base"]["max_drawdown"] > by_id["buy-and-hold"]["max_drawdown"]
    assert by_id["half-weight"]["max_drawdown"] < by_id["buy-and-hold"]["max_drawdown"]
    assert abs(by_id["vol-matched"]["max_drawdown"] - by_id["half-weight"]["max_drawdown"]) < 0.01
    assert 8.5 < by_id["vol-matched"]["total_return"] / by_id["half-weight"]["total_return"] < 9.5
    assert (
        max(by_id["grid-period-100"]["sharpe"], by_id["grid-period-150"]["sharpe"]) < by_id["base"]["sharpe"]
    )
    assert by_id["costs-x2"]["total_return"] < 0.4 * by_id["base"]["total_return"]

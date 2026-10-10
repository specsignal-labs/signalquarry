# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import signalquarry.api.factor as factor_api
from signalquarry._internal.contracts.factor_spec import FactorSpecV1
from signalquarry._internal.evidence.report import (
    DISCLAIMER,
    factor_ic_svg,
    factor_quintiles_svg,
    render_factor_report,
)
from signalquarry.api.envelope import Envelope

NS = {"svg": "http://www.w3.org/2000/svg"}
SPEC = {
    "id": "sample",
    "hypothesis": {
        "statement": "Volume predicts returns <conditionally> & imperfectly.",
        "falsification": "Nonpositive IC across chronological blocks.",
    },
}


def _diagnostics(scope: str = "synthetic", count: int = 3) -> dict:
    return {
        "scope": scope,
        "configuration_hash": "sha256:configuration",
        "dataset_identity": "sha256:dataset",
        "universe_identity": "sha256:universe",
        "label_identity": "sha256:labels",
        "synthetic": {"symbols": 30, "seed": 7, "planted_ic": 0.05},
        "horizons": [
            {
                "horizon": horizon,
                "observations": 101,
                "mean_ic": mean_ic,
                "icir": 0.6789,
                "top_quintile_net_return": 0.0123,
                "long_short_spread": -0.0234,
                "top_quintile_turnover": 0.3456,
                "max_share_of_adv": 0.4567,
                "quintile_monotonicity": 0.8765,
                "chronological_blocks": [0.2, None, -0.3, 0.0],
                "quintile_returns": [-0.02, None, 0.0, 0.01, 0.03],
            }
            for horizon, mean_ic in [(1, -0.123456789012345), (5, 0.0), (21, 0.25)][:count]
        ],
    }


@pytest.mark.parametrize("scope", ["synthetic", "unverified"])
@pytest.mark.parametrize("count", [1, 3])
def test_report_columns_scope_order_and_identity(scope: str, count: int) -> None:
    data = _diagnostics(scope, count)
    original = copy.deepcopy(data)
    text = render_factor_report(spec=SPEC, diagnostics=data)
    first = "\n".join(text.splitlines()[:4])
    assert f"Scope: `{scope}`" in first
    assert "No trial was recorded, no evidence grade assigned and no holdout accessed" in first
    statement = (
        "synthetic data: these numbers say nothing about real markets"
        if scope == "synthetic"
        else "descriptive diagnostics on recorded data whose provenance is not verified"
    )
    assert statement in first
    assert "Volume predicts returns &lt;conditionally&gt; &amp; imperfectly." in text
    assert SPEC["hypothesis"]["falsification"] in text
    assert (
        "| Horizon | Observations | Mean IC | ICIR | Top-quintile net return | "
        "Long-short spread (statistic) | Top-quintile turnover | Maximum share of ADV | "
        "Quintile monotonicity |"
    ) in text
    for row in data["horizons"]:
        assert (
            f"| {row['horizon']} | 101 | {row['mean_ic']:.4f} | 0.6789 | 1.23% | -2.34% | "
            "34.56% | 45.67% | 0.8765 |"
        ) in text
        assert f"### Horizon {row['horizon']}" in text
    assert "| 1 | 0.2000 |\n| 2 | – |\n| 3 | -0.3000 |\n| 4 | 0.0000 |" in text
    markers = ["## Hypothesis", "## Diagnostics", "## Chronological", "## Charts", "## Identity", DISCLAIMER]
    assert [text.index(marker) for marker in markers] == sorted(text.index(marker) for marker in markers)
    assert "not an executable portfolio" in text
    assert "not compounded portfolio returns" in text
    assert "unannualized" in text and "one-way" in text
    for key in ("configuration_hash", "dataset_identity", "universe_identity", "label_identity"):
        assert data[key] in text
    assert "(ic.svg)" in text and "(quintiles.svg)" in text
    if scope == "synthetic":
        assert "- symbols: 30\n- seed: 7\n- planted_ic: 0.05" in text
    else:
        assert "Synthetic panel arguments" not in text
    assert data == original


def test_missing_values_render_as_en_dashes() -> None:
    data = _diagnostics(count=1)
    row = data["horizons"][0]
    for key in row.keys() - {"horizon", "chronological_blocks", "quintile_returns"}:
        row[key] = None
    row["chronological_blocks"] = [None]
    text = render_factor_report(spec=SPEC, diagnostics=data)
    assert "| 1 | " + " | ".join(["–"] * 8) + " |" in text
    assert "| 1 | – |" in text
    assert "None" not in text


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("renderer", [factor_ic_svg, factor_quintiles_svg])
def test_svg_well_formed_with_missing_values_and_horizon_labels(count: int, renderer) -> None:
    rows = _diagnostics(count=count)["horizons"]
    original = copy.deepcopy(rows)
    tree = ET.fromstring(renderer(rows))
    assert tree.attrib["role"] == "img"
    assert tree.find("svg:title", NS) is not None
    assert tree.find("svg:line[@aria-label='zero line']", NS) is not None
    assert all(float(rect.attrib["height"]) >= 0 for rect in tree.findall("svg:rect", NS))
    labels = [node.text for node in tree.findall("svg:text", NS)]
    if renderer == factor_quintiles_svg:
        assert all(f"Q{number}" in labels for number in range(1, 6))
        assert all(f"Horizon {row['horizon']}" in labels for row in rows)
        assert "–" in labels
    else:
        assert all(str(row["horizon"]) in labels for row in rows)
    assert rows == original


def test_ic_negative_zero_positive_and_missing_bars() -> None:
    rows = [{"horizon": index, "mean_ic": value} for index, value in enumerate([-0.25, 0.0, 0.5, None], 1)]
    tree = ET.fromstring(factor_ic_svg(rows))
    zero_line = tree.find("svg:line[@aria-label='zero line']", NS)
    assert zero_line is not None
    zero = float(zero_line.attrib["y1"])
    bars = [rect for rect in tree.findall("svg:rect", NS) if rect.find("svg:title", NS) is not None]
    assert len(bars) == 3  # A missing IC is not drawn as a zero bar.
    negative, flat, positive = bars
    assert float(negative.attrib["y"]) == zero and float(negative.attrib["height"]) > 0
    assert float(flat.attrib["y"]) == zero and float(flat.attrib["height"]) == 0
    assert float(positive.attrib["y"]) < zero
    assert float(positive.attrib["y"]) + float(positive.attrib["height"]) == pytest.approx(zero, abs=0.02)
    assert "–" in [node.text for node in tree.findall("svg:text", NS)]


@pytest.mark.parametrize("value", [None, 0.0])
@pytest.mark.parametrize("renderer", [factor_ic_svg, factor_quintiles_svg])
def test_svg_with_no_nonzero_values(value: float | None, renderer) -> None:
    tree = ET.fromstring(renderer([{"horizon": 1, "mean_ic": value, "quintile_returns": [value] * 5}]))
    assert tree.find("svg:line[@aria-label='zero line']", NS) is not None
    for node in tree.iter():
        for key in ("x", "y", "height", "width", "y1", "y2"):
            if key in node.attrib:
                assert float(node.attrib[key]) >= 0


@pytest.mark.parametrize("status", ["invalid", "usage", "blocked", "unavailable", "error"])
def test_evaluation_refusal_is_returned_unchanged_except_command(monkeypatch, status) -> None:
    evaluated = Envelope(
        command="factor evaluate",
        status=status,
        reason_codes=["USAGE_INVALID"],
        summary="refusal",
        data={"detail": "original"},
        evidence={"grade": "synthetic", "claim_level": "none"},
        warnings=["warning"],
        next_actions=[{"command": "sqy doctor", "why": "check"}],
    )
    before = copy.deepcopy(vars(evaluated))
    monkeypatch.setattr(factor_api, "factor_evaluate", lambda *args, **kwargs: evaluated)
    result = factor_api.factor_report("sample")
    before["command"] = "factor report"
    assert result is evaluated and vars(result) == before


def _stub_evaluation(monkeypatch, root: Path, scope: str = "unverified") -> dict:
    diagnostics = _diagnostics(scope)
    spec = FactorSpecV1(**SPEC, family="sample", version="1")
    item = SimpleNamespace(spec=spec, configuration_hash=diagnostics["configuration_hash"])
    monkeypatch.setattr(factor_api, "_load", lambda *_: ({"sample": item}, root))
    monkeypatch.setattr(
        factor_api,
        "factor_evaluate",
        lambda *args, **kwargs: Envelope(command="factor evaluate", data=diagnostics),
    )
    return diagnostics


def test_recorded_report_forwards_arguments_and_preserves_no_evidence(tmp_path, monkeypatch) -> None:
    data = _stub_evaluation(monkeypatch, tmp_path)
    captured = {}

    def evaluate(factor_id, dataset_id, **kwargs):
        captured.update(factor_id=factor_id, dataset_id=dataset_id, **kwargs)
        return Envelope(command="factor evaluate", data=data)

    monkeypatch.setattr(factor_api, "factor_evaluate", evaluate)
    paths = [Path("one.json"), Path("two.json")]
    result = factor_api.factor_report(
        "sample",
        "D",
        universe_manifests=paths,
        synthetic_symbols=30,
        synthetic_seed=9,
        synthetic_planted_ic=0.1,
        project=tmp_path,
    )
    assert captured == {
        "factor_id": "sample",
        "dataset_id": "D",
        "universe_manifests": paths,
        "synthetic": False,
        "synthetic_symbols": 30,
        "synthetic_seed": 9,
        "synthetic_planted_ic": 0.1,
        "project": tmp_path,
    }
    assert result.status == "ok" and result.evidence is None
    assert result.data["scope"] == "unverified"
    assert result.data["horizons"] == [
        {"horizon": row["horizon"], "mean_ic": row["mean_ic"]} for row in data["horizons"]
    ]
    assert (
        "recorded data whose provenance is not verified"
        in (tmp_path / result.artifacts[0]["path"]).read_text()
    )


def test_failed_write_removes_partial_report(tmp_path, monkeypatch) -> None:
    _stub_evaluation(monkeypatch, tmp_path)
    original = Path.write_text

    def write(path, *args, **kwargs):
        if path.name == "ic.svg":
            raise OSError("disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write)
    result = factor_api.factor_report("sample", "D", project=tmp_path)
    assert result.status == "error" and result.reason_codes == ["FACTOR_REPORT_WRITE_FAILED"]
    assert result.artifacts == []
    assert list((tmp_path / ".signalquarry" / "factor_reports").iterdir()) == []


def test_configuration_change_refuses_mismatched_hypothesis(tmp_path, monkeypatch) -> None:
    diagnostics = _stub_evaluation(monkeypatch, tmp_path)
    diagnostics["configuration_hash"] = "sha256:changed"
    result = factor_api.factor_report("sample", "D", project=tmp_path)
    assert result.status == "invalid"
    assert result.reason_codes == ["FACTOR_REPORT_CONFIGURATION_CHANGED"]
    assert not (tmp_path / ".signalquarry").exists()


def test_existing_directory_is_refused_and_preserved(tmp_path, monkeypatch) -> None:
    _stub_evaluation(monkeypatch, tmp_path)

    class FixedTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, tzinfo=UTC)

    monkeypatch.setattr(factor_api, "datetime", FixedTime)
    first = factor_api.factor_report("sample", "D", project=tmp_path)
    assert first.status == "ok"

    def snapshot() -> dict[Path, bytes]:
        return {
            path: path.read_bytes()
            for path in tmp_path.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }

    files = snapshot()
    second = factor_api.factor_report("sample", "D", project=tmp_path)
    assert second.status == "error" and second.reason_codes == ["FACTOR_REPORT_WRITE_FAILED"]
    assert files == snapshot()

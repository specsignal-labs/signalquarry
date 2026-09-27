# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from signalquarry._internal.canonical import canonical_hash
from signalquarry.cli.main import main
from tests.paper.harness import hold_syna, rig


def test_snapshot_is_sanitized_and_self_hashed(tmp_path: Path) -> None:
    paper = rig(tmp_path, decide=hold_syna)
    days = paper.dataset.sessions[60:65]
    paper.arm(days[0])
    for session in days:
        paper.session(session)
    document = paper.kernel.snapshot()
    body = {k: v for k, v in document.items() if k != "snapshot_hash"}
    assert document["snapshot_hash"] == canonical_hash(body)
    assert document["schema_version"] == "signalquarry-public-performance/v1"
    assert len(document["equity_history"]) == 5 and document["recent_fills"]
    text = json.dumps(document)
    assert "order_id" not in text and "client_order_id" not in text and "account_id" not in text
    assert document["pnl"]["time_weighted_return"]["status"] == "unavailable"
    assert {p["symbol"] for p in document["positions"]} <= {"SYNA", "SYNB"}


def _sqy(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


def test_perf_publish_follows_the_publication_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "perf-lab"
    _sqy(capsys, "init", str(project), "--demo", "--package", "perf_lab_x")
    out = tmp_path / "feeds"
    code, payload = _sqy(
        capsys, "perf", "publish", "--alias", "demo", "--out", str(out), "--project", str(project)
    )
    assert (code, payload["reason_codes"]) == (2, ["PERF_PUBLISH_DENIED"])
    (project / "publication").mkdir()
    policy = project / "publication" / "sma-trend.publication.yaml"
    policy.write_text(
        "schema: signalquarry.publication/v1\nfamily: sma-trend\nfamily_label: Trend\nlive_feed: allow\n"
        "strategies: [{id: sma-trend, title: SMA trend demo, summary: A commercial strategy that must never get a live feed., assets: SYNA}]\n"
    )
    code, payload = _sqy(
        capsys, "perf", "publish", "--alias", "demo", "--out", str(out), "--project", str(project)
    )
    assert (
        payload["reason_codes"] == ["PUBLICATION_INVALID"]
        and "LIVE_FEED_NOT_FOR_COMMERCIAL" in payload["summary"]
    )
    policy.write_text(
        policy.read_text().replace("live_feed: allow", "live_feed: allow\ncommercial: false\ntier: rules")
    )
    code, payload = _sqy(
        capsys, "perf", "publish", "--alias", "demo", "--out", str(out), "--project", str(project)
    )
    assert code == 0, payload
    snapshot = json.loads((out / "demo" / "latest.json").read_text())
    assert (
        snapshot["snapshot_hash"] == payload["data"]["snapshot_hash"]
        and snapshot["deployment_alias"] == "demo"
    )
    assert snapshot["previous_snapshot_hash"] is None
    first = snapshot["snapshot_hash"]
    code, payload = _sqy(
        capsys, "perf", "publish", "--alias", "demo", "--out", str(out), "--project", str(project)
    )
    chained = json.loads((out / "demo" / "latest.json").read_text())
    assert (
        code == 0 and chained["previous_snapshot_hash"] == first == payload["data"]["previous_snapshot_hash"]
    )
    (out / "demo" / "latest.json").write_text("not json")
    code, payload = _sqy(
        capsys, "perf", "publish", "--alias", "demo", "--out", str(out), "--project", str(project)
    )
    assert (code, payload["reason_codes"]) == (65, ["PERF_FEED_INVALID"])

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from signalquarry.cli.main import main


def _sqy(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


def test_init_check_backtest_on_demo_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path / "demo-lab"
    code, payload = _sqy(capsys, "init", str(project), "--demo", "--package", "demo_lab_flow")
    assert code == 0, payload
    assert {
        "AGENTS.md",
        "CLAUDE.md",
        ".gitignore",
        "signalquarry.toml",
        "src/demo_lab_flow/sma_trend/strategy.yaml",
    } <= set(payload["data"]["files"])
    assert "{{" not in (project / "src/demo_lab_flow/sma_trend/strategy.yaml").read_text()

    code, payload = _sqy(capsys, "check", "--project", str(project))
    assert code == 0, payload
    checks = {item["name"]: item["ok"] for item in payload["data"]["strategies"][0]["checks"]}
    assert checks == {
        "import_policy": True,
        "contract": True,
        "determinism": True,
        "lookahead": True,
        "produces_targets": True,
    }

    code, payload = _sqy(capsys, "backtest", "--strategy", "sma-trend", "--project", str(project))
    assert code == 0, payload
    assert payload["evidence"] == {
        "grade": "synthetic",
        "claim_level": "none",
        "holdout": "not_sealed",
        "trial": None,
    }
    assert payload["metrics"]["sessions"] > 2000
    kinds = {item["kind"] for item in payload["artifacts"]}
    assert kinds == {"result", "equity", "benchmark", "fills", "decisions"}
    result = json.loads(
        (project / next(a["path"] for a in payload["artifacts"] if a["kind"] == "result")).read_text()
    )
    assert result["ledger_hash"] == payload["data"]["ledger_hash"] and result["result_hash"].startswith(
        "sha256:"
    )


def test_check_blocks_a_strategy_that_reads_the_clock(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "rogue"
    _sqy(capsys, "init", str(project), "--demo", "--package", "rogue_lab_flow")
    strategy = project / "src/rogue_lab_flow/sma_trend/strategy.py"
    strategy.write_text(
        strategy.read_text().replace(
            "from decimal import Decimal", "import datetime\nfrom decimal import Decimal"
        )
        + "\nSTAMP = datetime.datetime.now()\n"
    )
    code, payload = _sqy(capsys, "check", "--project", str(project))
    assert code == 2 and payload["reason_codes"] == ["CONFORMANCE_FAILED"]
    failed = [item for item in payload["data"]["strategies"][0]["checks"] if not item["ok"]]
    assert failed[0]["name"] == "import_policy" and "import datetime" in failed[0]["detail"]


def test_real_data_provider_is_reported_unavailable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "real"
    _sqy(capsys, "init", str(project), "--package", "real_lab_flow")
    code, payload = _sqy(capsys, "backtest", "--strategy", "sma-trend", "--project", str(project))
    assert code == 69 and payload["reason_codes"] == ["PROVIDER_UNAVAILABLE"]


def test_init_refuses_non_empty_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "keep.txt").write_text("x")
    code, payload = _sqy(capsys, "init", str(tmp_path))
    assert code == 65 and payload["reason_codes"] == ["PROJECT_DIR_NOT_EMPTY"]


def test_every_emitted_engine_code_is_registered() -> None:
    import re

    from signalquarry._internal.contracts.reason_codes import REASON_CODES

    source = (Path(__file__).parents[2] / "src/signalquarry/_internal/engine/backtest.py").read_text()
    emitted = set(re.findall(r'EngineError\(f?"([A-Z_]+)', source)) | set(
        re.findall(r'unavailable\("([A-Z_]+)"', source)
    )
    emitted |= set(re.findall(r'warnings\.append\(f"([A-Z_]+)', source))
    assert emitted and emitted <= set(REASON_CODES), emitted - set(REASON_CODES)


def test_init_options_kind(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path / "wheel-demo"
    code, payload = _sqy(
        capsys, "init", str(project), "--demo", "--kind", "options", "--package", "wheel_demo_flow"
    )
    assert code == 0 and "src/wheel_demo_flow/wheel/strategy.yaml" in payload["data"]["files"]
    assert not (project / "src/wheel_demo_flow/sma_trend").exists()
    assert "--strategy wheel" in (project / "README.md").read_text()
    assert _sqy(capsys, "check", "--project", str(project))[0] == 0
    code, payload = _sqy(capsys, "backtest", "--strategy", "wheel", "--project", str(project))
    assert code == 0 and payload["evidence"]["grade"] == "synthetic"
    assert _sqy(capsys, "init", str(tmp_path / "x"), "--lab", "--kind", "options")[0] == 64


def test_upgrade_agents_md_refreshes_only_managed_blocks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "agents"
    _sqy(capsys, "init", str(project), "--demo", "--package", "agents_flow")
    agents = project / "AGENTS.md"
    pristine = agents.read_text()
    assert "src/agents_flow/<strategy>" in pristine
    code, payload = _sqy(capsys, "init", "--upgrade-agents-md", "--project", str(project))
    assert code == 0 and payload["data"]["updated"] == [] and agents.read_text() == pristine

    stale = pristine.replace("Try fewer, better-reasoned ideas.", "Try many ideas.")
    stale = stale.replace(
        stale[
            stale.index("<!-- signalquarry:begin project/never") : stale.index(
                "<!-- signalquarry:end project/never -->"
            )
        ],
        "",
    ).replace("<!-- signalquarry:end project/never -->\n", "")
    agents.write_text("My own notes stay.\n\n" + stale + "\nTrailing notes stay.\n")
    code, payload = _sqy(capsys, "init", "--upgrade-agents-md", "--project", str(project))
    text = agents.read_text()
    assert code == 0 and payload["data"]["updated"] == ["project/golden-path"]
    assert payload["data"]["added"] == ["project/never"]
    assert "Try fewer, better-reasoned ideas." in text and "Try many ideas." not in text
    assert text.startswith("My own notes stay.") and "Trailing notes stay." in text

    agents.write_text("# hand-written\n")
    code, payload = _sqy(capsys, "init", "--upgrade-agents-md", "--project", str(project))
    assert code == 65 and payload["reason_codes"] == ["AGENTS_MD_UNMANAGED"]
    assert _sqy(capsys, "init", str(project), "--upgrade-agents-md")[0] == 64
    assert _sqy(capsys, "init")[0] == 64
    code, payload = _sqy(capsys, "init", "--upgrade-agents-md", "--project", str(tmp_path / "none"))
    assert code == 65 and payload["reason_codes"] == ["PROJECT_NOT_FOUND"]


def test_lab_agents_md_upgrades_from_the_lab_template(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lab = tmp_path / "lab"
    _sqy(capsys, "init", str(lab), "--lab", "--package", "labflow")
    code, payload = _sqy(capsys, "init", "--upgrade-agents-md", "--project", str(lab))
    assert code == 0 and payload["data"]["template"] == "lab" and payload["data"]["unchanged"]


def test_check_parity_on_the_demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path / "parity"
    _sqy(capsys, "init", str(project), "--demo", "--package", "parity_flow")
    code, payload = _sqy(capsys, "check", "--parity", "--project", str(project))
    checks = {c["name"]: c for c in payload["data"]["strategies"][0]["checks"]}
    assert code == 0 and checks["parity"]["ok"] and "60 sessions" in checks["parity"]["detail"]


def test_options_freeze_seals_no_holdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path / "wheel-holdout"
    _sqy(capsys, "init", str(project), "--demo", "--kind", "options", "--package", "wheel_holdout_flow")
    code, payload = _sqy(capsys, "spec", "freeze", "--strategy", "wheel", "--project", str(project))
    assert code == 0 and payload["data"]["holdout_start"] is None and payload["data"]["newly_sealed"] is False
    assert "holdout disabled" in payload["summary"]
    code, payload = _sqy(capsys, "holdout", "seal", "--strategy", "wheel", "--project", str(project))
    assert code == 65 and "paper forward" in payload["summary"]

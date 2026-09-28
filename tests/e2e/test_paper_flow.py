# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from signalquarry import api
from signalquarry._internal.data.credentials import load_paper_credentials
from signalquarry._internal.data.dataset import truncated, with_pending_session
from signalquarry._internal.paper.brokers.fake import FakeBroker
from signalquarry._internal.paper.runner import PaperKernel
from signalquarry.cli.main import main


def _sqy(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    project = tmp_path / "paper-lab"
    code, _ = _sqy(capsys, "init", str(project), "--demo", "--package", "paper_lab_flow")
    assert code == 0
    return project


def test_demo_dry_run_preflight_and_status_work_offline(
    demo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (demo / "paper" / "demo.paper.yaml").is_file()
    code, payload = _sqy(capsys, "paper", "dry-run", "--alias", "demo", "--project", str(demo))
    assert code == 0, payload
    assert payload["data"]["session"] == "2025-12-31" and payload["data"]["decision"]["action"] in (
        "target",
        "hold",
    )
    assert payload["evidence"]["grade"] == "simulated"

    code, payload = _sqy(capsys, "paper", "preflight", "--alias", "demo", "--project", str(demo))
    assert code == 2 and set(payload["reason_codes"]) == {"PAPER_NOT_ARMED", "PAPER_SUBMISSION_DISABLED"}

    code, payload = _sqy(capsys, "paper", "status", "--alias", "demo", "--project", str(demo))
    assert code == 0 and payload["data"]["state"] == "unarmed"


def test_simulated_deployments_can_never_submit(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for action in (("run-once",), ("arm", "--reason", "x"), ("halt", "--reason", "x"), ("reconcile",)):
        code, payload = _sqy(
            capsys, "paper", action[0], "--alias", "demo", "--project", str(demo), *action[1:]
        )
        assert (code, payload["reason_codes"]) == (78, ["PAPER_BROKER_SIMULATED"]), action


def test_unknown_alias_and_invalid_config(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _sqy(capsys, "paper", "status", "--alias", "nope", "--project", str(demo))
    assert (code, payload["reason_codes"], payload["data"]["deployments"]) == (
        65,
        ["PAPER_CONFIG_NOT_FOUND"],
        ["demo"],
    )
    config = demo / "paper" / "demo.paper.yaml"
    config.write_text(config.read_text().replace("broker: simulated", "broker: live"))
    code, payload = _sqy(capsys, "paper", "status", "--alias", "demo", "--project", str(demo))
    assert (code, payload["reason_codes"]) == (65, ["PAPER_CONFIG_INVALID"])


def _alpaca_config(project: Path) -> None:
    config = project / "paper" / "demo.paper.yaml"
    config.write_text(config.read_text().replace("broker: simulated", "broker: alpaca-paper"))


def test_arm_refuses_agents_and_missing_credentials(
    demo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _alpaca_config(demo)
    monkeypatch.setenv("SIGNALQUARRY_CONFIG_DIR", str(demo / "no-config"))
    monkeypatch.delenv("SIGNALQUARRY_PAPER_KEY_ID", raising=False)
    code, payload = _sqy(capsys, "paper", "arm", "--alias", "demo", "--reason", "go", "--project", str(demo))
    assert (code, payload["reason_codes"]) == (69, ["PAPER_CREDENTIALS_MISSING"])
    monkeypatch.setenv("SIGNALQUARRY_PAPER_KEY_ID", "PKTESTKEY")
    monkeypatch.setenv("SIGNALQUARRY_PAPER_SECRET_KEY", "not-a-real-secret")
    code, payload = _sqy(capsys, "paper", "arm", "--alias", "demo", "--reason", "go", "--project", str(demo))
    assert (code, payload["reason_codes"]) == (78, ["PAPER_ARM_REQUIRES_HUMAN"])
    assert "not-a-real-secret" not in json.dumps(payload)


def test_denied_key_is_refused(
    demo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    _alpaca_config(demo)
    config = demo / "paper" / "demo.paper.yaml"
    denied = hashlib.sha256(b"PKOTHERDEPLOYMENT").hexdigest()
    config.write_text(config.read_text() + f"denied_key_id_sha256: ['{denied}']\n")
    monkeypatch.setenv("SIGNALQUARRY_PAPER_KEY_ID", "PKOTHERDEPLOYMENT")
    monkeypatch.setenv("SIGNALQUARRY_PAPER_SECRET_KEY", "x")
    code, payload = _sqy(capsys, "paper", "dry-run", "--alias", "demo", "--project", str(demo))
    assert (code, payload["reason_codes"]) == (2, ["PAPER_KEY_DENIED"])


def test_human_arm_binds_the_freeze(
    demo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _alpaca_config(demo)

    def fake_kernel(context):
        dataset = context.resolved.dataset
        broker = FakeBroker(dataset, context.deployment.spec.account.initial_cash)
        broker.pre_open(dataset.sessions[-1])
        return PaperKernel(
            context.deployment,
            broker,
            lambda s: with_pending_session(truncated(dataset, dataset.index_of(s)), s),
            now=lambda: broker.now,
        )

    monkeypatch.setattr(api.paper, "kernel_for", fake_kernel)
    envelope = api.paper_arm("demo", "reviewed", project=demo, confirm=lambda _: "demo", interactive=True)
    assert envelope.reason_codes == ["FREEZE_REQUIRED"]
    assert _sqy(capsys, "spec", "freeze", "--strategy", "sma-trend", "--project", str(demo))[0] == 0
    wrong = api.paper_arm("demo", "reviewed", project=demo, confirm=lambda _: "dmeo", interactive=True)
    assert wrong.reason_codes == ["PAPER_ARM_NOT_CONFIRMED"]
    armed = api.paper_arm("demo", "reviewed", project=demo, confirm=lambda _: "demo", interactive=True)
    assert armed.status == "ok", armed.as_dict()
    journal = (demo / "paper" / "demo" / "journal.jsonl").read_text().splitlines()
    entry = json.loads(journal[-1])
    assert (
        entry["kind"] == "armed"
        and entry["reason"] == "reviewed"
        and entry["freeze_hash"].startswith("sha256:")
    )
    status = api.paper_status("demo", project=demo)
    assert status.data["state"] == "armed"


@pytest.mark.parametrize("target", ["systemd", "launchd", "cron", "github-actions"])
def test_schedule_templates(demo: Path, capsys: pytest.CaptureFixture[str], target: str) -> None:
    code, payload = _sqy(
        capsys, "paper", "schedule", "--alias", "demo", "--target", target, "--project", str(demo)
    )
    assert code == 0 and payload["artifacts"]
    text = "".join((demo / f).read_text() for f in payload["data"]["files"])
    assert "run-once" in text and (
        "--alias demo" in text or "<string>--alias</string>\n    <string>demo</string>" in text
    )
    assert "09:00" in text or "<integer>9</integer>" in text or " 9 * * 1-5" in text or "13,14" in text
    assert "SECRET_KEY: ${{" in text if target == "github-actions" else "SECRET" not in text
    if target == "systemd":  # post-close journal-head commitment and weekly proof upgrade
        names = {Path(f).name for f in payload["data"]["files"]}
        assert {"signalquarry-commit-demo.timer", "signalquarry-ots-upgrade-demo.timer"} <= names
        assert (
            "commit create --strategy sma-trend --alias demo" in text and "16:30:00 America/New_York" in text
        )
        assert "ots upgrade" in text


@pytest.mark.parametrize("target", ["systemd", "launchd", "cron"])
def test_schedule_opt_in_notification(demo: Path, capsys: pytest.CaptureFixture[str], target: str) -> None:
    command = Path("/tmp/signalquarry-notify")
    code, payload = _sqy(
        capsys,
        "paper",
        "schedule",
        "--alias",
        "demo",
        "--target",
        target,
        "--notify-command",
        str(command),
        "--project",
        str(demo),
    )
    assert code == 0, payload
    text = "".join((demo / f).read_text() for f in payload["data"]["files"])
    assert "--notify-command" in text and str(command) in text


def test_notification_requires_local_scheduler(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _sqy(
        capsys,
        "paper",
        "schedule",
        "--alias",
        "demo",
        "--target",
        "github-actions",
        "--notify-command",
        "/tmp/signalquarry-notify",
        "--project",
        str(demo),
    )
    assert (code, payload["reason_codes"]) == (64, ["USAGE_INVALID"])


def test_run_once_notification_preserves_exit_and_scrubs_credentials(
    demo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def notify(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("signalquarry.cli.main.subprocess.run", notify)
    monkeypatch.setenv("SIGNALQUARRY_PAPER_SECRET_KEY", "test-secret")
    code, payload = _sqy(
        capsys,
        "paper",
        "run-once",
        "--alias",
        "demo",
        "--project",
        str(demo),
        "--notify-command",
        "/tmp/signalquarry-notify",
    )
    assert (code, payload["reason_codes"]) == (78, ["PAPER_BROKER_SIMULATED"])
    assert len(calls) == 1 and calls[0][0] == ["/tmp/signalquarry-notify"]
    assert calls[0][1]["env"]["SIGNALQUARRY_EXIT_CODE"] == "78"
    assert calls[0][1]["env"]["SIGNALQUARRY_ALIAS"] == "demo"
    assert "SIGNALQUARRY_PAPER_SECRET_KEY" not in calls[0][1]["env"]
    assert calls[0][1]["timeout"] == 10


def test_notification_failure_does_not_hide_run_once_failure(
    demo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("notify", 10)

    monkeypatch.setattr("signalquarry.cli.main.subprocess.run", fail)
    code, payload = _sqy(
        capsys,
        "paper",
        "run-once",
        "--alias",
        "demo",
        "--project",
        str(demo),
        "--notify-command",
        "/tmp/signalquarry-notify",
    )
    assert (code, payload["reason_codes"]) == (78, ["PAPER_BROKER_SIMULATED"])


def test_paper_credentials_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNALQUARRY_CONFIG_DIR", str(tmp_path))
    path = tmp_path / "credentials.toml"
    path.write_text(
        '[data]\nkey_id = "D"\nsecret_key = "DS"\n[paper.demo]\nkey_id = "P"\nsecret_key = "PS"\n'
    )
    os.chmod(path, 0o600)
    keys = load_paper_credentials("paper.demo", environ={})
    assert (
        keys is not None and (keys.key_id, keys.secret_key) == ("P", "PS") and "[paper.demo]" in keys.source
    )
    assert load_paper_credentials("paper.other", environ={}) is None
    os.chmod(path, 0o644)
    with pytest.raises(PermissionError):
        load_paper_credentials("paper.demo", environ={})


def test_drift_on_an_empty_journal_records_an_artifact(
    demo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = _sqy(capsys, "paper", "drift", "--alias", "demo", "--project", str(demo))
    assert code == 0, payload
    assert payload["data"]["g5"]["ok"] is False and payload["data"]["g5"]["clean_sessions"] == 0
    document = json.loads((demo / payload["data"]["artifact"]).read_text())
    assert document["kind"] == "paper_drift" and document["alias"] == "demo"

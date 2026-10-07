# SPDX-License-Identifier: Apache-2.0
"""The scheduler templates fire at the right New York times and run exactly the intended command."""

from __future__ import annotations

import plistlib
import shlex
import sys
from pathlib import Path

import pytest

from signalquarry._internal.contracts.paper import PaperDeploymentV1
from signalquarry._internal.paper.schedule import MINUTES, TARGETS, write_schedule

CONFIG = PaperDeploymentV1.model_validate(
    {"schema": "signalquarry.paper/v1", "alias": "demo", "strategy": "momentum"}
)


def _command(root: Path) -> str:
    return (
        f"{shlex.quote(sys.executable)} -m signalquarry.cli.main --json paper run-once "
        f"--alias demo --project {shlex.quote(str(root))}"
    )


def _write(root: Path, target: str, **options: object) -> dict[str, str]:
    paths = write_schedule(root, CONFIG, target, **options)  # type: ignore[arg-type]
    return {path.name: path.read_text(encoding="utf-8") for path in paths}


def _lines(text: str) -> list[str]:
    return text.splitlines()


def test_the_targets_and_minutes_are_the_documented_ones() -> None:
    assert TARGETS == ("systemd", "launchd", "cron", "github-actions")
    assert MINUTES == (0, 5, 10, 15, 20, 25)


@pytest.mark.parametrize(
    ("target", "names"),
    [
        (
            "systemd",
            {
                "signalquarry-paper-demo.service",
                "signalquarry-paper-demo.timer",
                "signalquarry-commit-demo.service",
                "signalquarry-commit-demo.timer",
                "signalquarry-ots-upgrade-demo.service",
                "signalquarry-ots-upgrade-demo.timer",
            },
        ),
        ("launchd", {"dev.signalquarry.paper.demo.plist"}),
        ("cron", {"demo.crontab"}),
        ("github-actions", {"paper-demo.yml"}),
    ],
)
def test_each_target_writes_its_files_under_a_per_target_directory(
    tmp_path: Path, target: str, names: set[str]
) -> None:
    paths = write_schedule(tmp_path, CONFIG, target)
    assert {path.name for path in paths} == names
    assert {path.parent for path in paths} == {tmp_path / "paper" / "schedule" / target}
    assert all(path.read_text(encoding="utf-8").endswith("\n") for path in paths)


def test_the_systemd_service_runs_the_command_with_the_documented_hardening(tmp_path: Path) -> None:
    lines = _lines(_write(tmp_path, "systemd")["signalquarry-paper-demo.service"])
    assert "Description=SignalQuarry paper run-once (demo)" in lines
    assert "Type=oneshot" in lines and f"WorkingDirectory={tmp_path}" in lines
    assert f"ExecStart={_command(tmp_path)}" in lines
    assert "SuccessExitStatus=75 69" in lines
    assert {"NoNewPrivileges=yes", "ProtectSystem=strict", "ProtectHome=read-only", "PrivateTmp=yes"} <= set(
        lines
    )
    assert f"ReadWritePaths={tmp_path}/paper/demo %h/.cache/signalquarry" in lines
    assert "After=network-online.target" in lines and "Wants=network-online.target" in lines
    assert "systemctl --user enable --now signalquarry-paper-demo.timer" in lines[0]


def test_the_systemd_timer_fires_weekdays_nine_to_nine_twenty_five_new_york_time(tmp_path: Path) -> None:
    lines = _lines(_write(tmp_path, "systemd")["signalquarry-paper-demo.timer"])
    assert "OnCalendar=Mon..Fri *-*-* 09:00,05,10,15,20,25:00 America/New_York" in lines
    assert "Persistent=false" in lines and "AccuracySec=10s" in lines
    assert "WantedBy=timers.target" in lines


def test_the_systemd_commitment_units_run_after_the_close_and_weekly_upgrade(tmp_path: Path) -> None:
    files = _write(tmp_path, "systemd")
    commit = _lines(files["signalquarry-commit-demo.service"])
    expected = (
        f"{shlex.quote(sys.executable)} -m signalquarry.cli.main --json commit create --strategy momentum "
        f"--alias demo --stamp --project {shlex.quote(str(tmp_path))}"
    )
    assert f"ExecStart={expected}" in commit
    assert f"ReadWritePaths={tmp_path}/evidence %h/.cache/signalquarry" in commit
    timer = _lines(files["signalquarry-commit-demo.timer"])
    assert "OnCalendar=Mon..Fri *-*-* 16:30:00 America/New_York" in timer and "Persistent=true" in timer
    upgrade = files["signalquarry-ots-upgrade-demo.service"]
    assert "ots upgrade" in upgrade and f"{tmp_path}/evidence/commitments" in upgrade
    assert f"ReadWritePaths={tmp_path}/evidence" in _lines(upgrade)
    assert "OnCalendar=Sat *-*-* 10:00:00 America/New_York" in _lines(
        files["signalquarry-ots-upgrade-demo.timer"]
    )


def test_the_launchd_plist_parses_and_schedules_thirty_weekday_ticks(tmp_path: Path) -> None:
    plist = plistlib.loads(_write(tmp_path, "launchd")["dev.signalquarry.paper.demo.plist"].encode())
    assert plist["Label"] == "dev.signalquarry.paper.demo"
    assert plist["WorkingDirectory"] == str(tmp_path)
    assert plist["ProgramArguments"] == shlex.split(_command(tmp_path))
    log = f"{tmp_path}/paper/demo/launchd.log"
    assert plist["StandardOutPath"] == plist["StandardErrorPath"] == log
    ticks = plist["StartCalendarInterval"]
    assert len(ticks) == 30
    assert {(item["Weekday"], item["Hour"], item["Minute"]) for item in ticks} == {
        (day, 9, minute) for day in range(1, 6) for minute in MINUTES
    }
    assert "StartInterval" not in plist


def test_launchd_arguments_are_xml_escaped(tmp_path: Path) -> None:
    notify = Path("/opt/notify & page.sh")
    text = _write(tmp_path, "launchd", notify_command=notify)["dev.signalquarry.paper.demo.plist"]
    assert "&amp;" in text
    assert plistlib.loads(text.encode())["ProgramArguments"][-2:] == ["--notify-command", str(notify)]


def test_the_cron_entry_uses_new_york_time_weekdays_and_logs_to_the_deployment(tmp_path: Path) -> None:
    lines = _lines(_write(tmp_path, "cron")["demo.crontab"])
    assert "CRON_TZ=America/New_York" in lines
    entry = lines[-1]
    assert entry.startswith(f"0,5,10,15,20,25 9 * * 1-5 cd {shlex.quote(str(tmp_path))} && ")
    assert entry.endswith(f"{_command(tmp_path)} >> paper/demo/cron.log 2>&1")


def test_paths_with_spaces_are_quoted_in_commands(tmp_path: Path) -> None:
    root = tmp_path / "my project"
    entry = write_schedule(root, CONFIG, "cron")[0].read_text(encoding="utf-8").splitlines()[-1]
    assert f"cd {shlex.quote(str(root))} && " in entry
    assert f"--project {shlex.quote(str(root))}" in entry
    assert "'" in shlex.quote(str(root))


def test_the_github_workflow_runs_the_command_in_both_dst_offsets_and_commits_state(tmp_path: Path) -> None:
    text = _write(tmp_path, "github-actions")["paper-demo.yml"]
    lines = _lines(text)
    assert 'cron: "0,5,10,15,20,25 13,14 * * 1-5"' in text
    assert "name: paper-demo" in lines and "  group: paper-demo" in lines
    assert "cancel-in-progress: false" in text and "concurrency:" in text
    assert "sqy --json paper run-once --alias demo" in text
    assert "case $code in 0|69|75) exit 0 ;; *) exit $code ;; esac" in text
    assert "${{ secrets.SIGNALQUARRY_PAPER_KEY_ID }}" in text
    assert "${{ secrets.SIGNALQUARRY_PAPER_SECRET_KEY }}" in text
    assert "paper/demo/journal.jsonl" in text and "paper/demo/arm.json" in text
    assert "DEMO ONLY" in lines[0]


def test_polling_systemd_replaces_only_the_timer(tmp_path: Path) -> None:
    files = _write(tmp_path, "systemd", poll=True)
    timer = _lines(files["signalquarry-paper-demo.timer"])
    assert "OnCalendar=Mon..Fri *-*-* 09:30..59:00 America/New_York" in timer
    assert "OnCalendar=Mon..Fri *-*-* 10..15:*:00 America/New_York" in timer
    assert "AccuracySec=5s" in timer and "Persistent=false" in timer
    assert f"ExecStart={_command(tmp_path)}" in _lines(files["signalquarry-paper-demo.service"])


def test_polling_cron_covers_the_regular_session_each_minute(tmp_path: Path) -> None:
    lines = _lines(_write(tmp_path, "cron", poll=True)["demo.crontab"])
    entries = [line for line in lines if line and not line.startswith(("#", "CRON_TZ"))]
    assert [entry.split(" cd ")[0] for entry in entries] == ["30-59 9 * * 1-5", "* 10-15 * * 1-5"]
    assert all(f"{_command(tmp_path)} >> paper/demo/cron.log 2>&1" in entry for entry in entries)


def test_polling_launchd_uses_a_sixty_second_interval_instead_of_a_calendar(tmp_path: Path) -> None:
    plist = plistlib.loads(
        _write(tmp_path, "launchd", poll=True)["dev.signalquarry.paper.demo.plist"].encode()
    )
    assert plist["StartInterval"] == 60
    assert "StartCalendarInterval" not in plist
    assert plist["Label"] == "dev.signalquarry.paper.demo" and plist["StandardOutPath"].endswith(
        "launchd.log"
    )


def test_notify_commands_must_be_absolute_and_unsupported_for_github_actions(tmp_path: Path) -> None:
    for target in ("systemd", "launchd", "cron"):
        with pytest.raises(ValueError, match="^SCHEDULE_NOTIFY_COMMAND_UNSUPPORTED$"):
            write_schedule(tmp_path, CONFIG, target, notify_command=Path("relative.sh"))
        text = "".join(_write(tmp_path, target, notify_command=Path("/opt/notify.sh")).values())
        assert "--notify-command /opt/notify.sh" in text or "<string>--notify-command</string>" in text
    with pytest.raises(ValueError, match="^SCHEDULE_NOTIFY_COMMAND_UNSUPPORTED$"):
        write_schedule(tmp_path, CONFIG, "github-actions", notify_command=Path("/opt/notify.sh"))
    with pytest.raises(ValueError, match="^SCHEDULE_TARGET_UNSUPPORTED_FOR_OPTIONS$"):
        write_schedule(tmp_path, CONFIG, "github-actions", poll=True)
    with pytest.raises(KeyError):
        write_schedule(tmp_path, CONFIG, "windows-task-scheduler")


@pytest.mark.parametrize("poll", [False, True])
@pytest.mark.parametrize("target", ["systemd", "launchd", "cron"])
def test_the_notify_command_is_appended_to_the_full_command_in_every_variant(
    tmp_path: Path, target: str, poll: bool
) -> None:
    notify = Path("/opt/notify.sh")
    expected = f"{_command(tmp_path)} --notify-command /opt/notify.sh"
    files = _write(tmp_path, target, poll=poll, notify_command=notify)
    text = "\n".join(files.values())
    if target == "launchd":
        arguments = plistlib.loads(next(iter(files.values())).encode())["ProgramArguments"]
        assert arguments == shlex.split(expected)
    else:
        assert expected in text
    assert "--notify-command" not in "\n".join(_write(tmp_path / "plain", target, poll=poll).values())


def test_launchd_lists_one_tick_and_one_argument_per_line(tmp_path: Path) -> None:
    text = _write(tmp_path, "launchd")["dev.signalquarry.paper.demo.plist"]
    lines = _lines(text)
    assert sum("<key>Weekday</key>" in line for line in lines) == 30
    arguments = shlex.split(_command(tmp_path))
    assert [line.strip() for line in lines if line.startswith("    <string>")] == [
        f"<string>{part}</string>" for part in arguments
    ]


def test_polling_launchd_keeps_everything_else_and_drops_only_the_calendar(tmp_path: Path) -> None:
    regular = _write(tmp_path, "launchd")["dev.signalquarry.paper.demo.plist"]
    polling = _write(tmp_path, "launchd", poll=True)["dev.signalquarry.paper.demo.plist"]
    head = regular[: regular.index("  <key>StartCalendarInterval</key>")]
    tail = regular[regular.index("  <key>StandardOutPath</key>") :]
    assert polling.startswith(head) and polling.endswith(tail)
    assert (
        "  <key>StartInterval</key><integer>60</integer>\n  <!-- polls act only while the market is open -->\n"
        in polling
    )
    assert "Weekday" not in polling


def test_regenerating_a_schedule_overwrites_the_previous_files(tmp_path: Path) -> None:
    first = write_schedule(tmp_path, CONFIG, "cron")
    first[0].write_text("stale", encoding="utf-8")
    second = write_schedule(tmp_path, CONFIG, "cron")
    assert second == first and "CRON_TZ=America/New_York" in second[0].read_text(encoding="utf-8")


def test_files_are_written_as_utf8(tmp_path: Path) -> None:
    root = tmp_path / "café"
    (path,) = write_schedule(root, CONFIG, "cron")
    assert "café" in path.read_bytes().decode("utf-8")

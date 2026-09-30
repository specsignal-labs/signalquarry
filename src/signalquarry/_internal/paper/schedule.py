# SPDX-License-Identifier: Apache-2.0
"""Scheduler templates for ``sqy paper run-once``. Written for review; never installed automatically.

Every template fires every 5 minutes from 09:00 to 09:25 New York time on weekdays.
``run-once`` is idempotent and refuses outside its window, so extra ticks are harmless.
Exit codes: 0 done or nothing to do · 75 retry on the next tick · 69 provider or broker
unavailable, retry · 2 or 78 blocked or disabled, needs a human.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from signalquarry._internal.contracts.paper import PaperDeploymentV1

TARGETS = ("systemd", "launchd", "cron", "github-actions")
MINUTES = (0, 5, 10, 15, 20, 25)


def _command(root: Path, alias: str, notify_command: Path | None = None) -> str:
    command = f"{shlex.quote(sys.executable)} -m signalquarry.cli.main --json paper run-once --alias {alias} --project {shlex.quote(str(root))}"
    if notify_command is not None:
        command += f" --notify-command {shlex.quote(str(notify_command))}"
    return command


def _systemd(root: Path, alias: str, notify_command: Path | None = None) -> dict[str, str]:
    unit = f"signalquarry-paper-{alias}"
    return {
        f"{unit}.service": f"""# Install as a user unit: ~/.config/systemd/user/, then `systemctl --user enable --now {unit}.timer`.
[Unit]
Description=SignalQuarry paper run-once ({alias})
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory={root}
ExecStart={_command(root, alias, notify_command)}
# 75 = retry on the next tick, 69 = broker/provider unavailable: not failures of the unit.
SuccessExitStatus=75 69
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths={root}/paper/{alias} %h/.cache/signalquarry
PrivateTmp=yes
""",
        f"{unit}.timer": f"""[Unit]
Description=SignalQuarry paper run-once ({alias}), weekdays 09:00-09:25 New York

[Timer]
OnCalendar=Mon..Fri *-*-* 09:00,05,10,15,20,25:00 America/New_York
AccuracySec=10s
Persistent=false

[Install]
WantedBy=timers.target
""",
    }


def _launchd(root: Path, alias: str, notify_command: Path | None = None) -> dict[str, str]:
    intervals = "\n".join(
        f"    <dict><key>Weekday</key><integer>{day}</integer><key>Hour</key><integer>9</integer>"
        f"<key>Minute</key><integer>{minute}</integer></dict>"
        for day in range(1, 6)
        for minute in MINUTES
    )
    arguments = "\n".join(
        f"    <string>{escape(part)}</string>" for part in shlex.split(_command(root, alias, notify_command))
    )
    return {
        f"dev.signalquarry.paper.{alias}.plist": f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<!-- launchd uses the host's local time zone: set the host to America/New_York or adjust Hour. -->
<plist version="1.0">
<dict>
  <key>Label</key><string>dev.signalquarry.paper.{alias}</string>
  <key>WorkingDirectory</key><string>{root}</string>
  <key>ProgramArguments</key>
  <array>
{arguments}
  </array>
  <key>StartCalendarInterval</key>
  <array>
{intervals}
  </array>
  <key>StandardOutPath</key><string>{root}/paper/{alias}/launchd.log</string>
  <key>StandardErrorPath</key><string>{root}/paper/{alias}/launchd.log</string>
</dict>
</plist>
""",
    }


def _cron(root: Path, alias: str, notify_command: Path | None = None) -> dict[str, str]:
    minutes = ",".join(str(m) for m in MINUTES)
    return {
        f"{alias}.crontab": f"""# SignalQuarry paper run-once ({alias}). Install with `crontab -e`; needs a cron with CRON_TZ.
CRON_TZ=America/New_York
{minutes} 9 * * 1-5 cd {shlex.quote(str(root))} && {_command(root, alias, notify_command)} >> paper/{alias}/cron.log 2>&1
""",
    }


def _github_actions(root: Path, alias: str) -> dict[str, str]:
    return {
        f"paper-{alias}.yml": f"""# DEMO ONLY. GitHub schedules can be delayed or skipped; use systemd on a small VM for forward tests.
# Copy to .github/workflows/. Needs repository secrets SIGNALQUARRY_PAPER_KEY_ID / SIGNALQUARRY_PAPER_SECRET_KEY
# for a dedicated paper account. The journal is committed to the `paper-state` branch; a concurrent run's
# push fails (compare-and-swap) instead of overwriting.
name: paper-{alias}
on:
  schedule:
    - cron: "0,5,10,15,20,25 13,14 * * 1-5"  # 09:00-09:25 New York in both EDT and EST; run-once skips outside its window
  workflow_dispatch: {{}}
concurrency:
  group: paper-{alias}
  cancel-in-progress: false
permissions:
  contents: write
jobs:
  run-once:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install .
      - name: Restore journal
        run: |
          git fetch origin +refs/heads/paper-state:refs/remotes/origin/paper-state || true
          git checkout origin/paper-state -- paper/{alias} 2>/dev/null || true
      - name: run-once
        env:
          SIGNALQUARRY_PAPER_KEY_ID: ${{{{ secrets.SIGNALQUARRY_PAPER_KEY_ID }}}}
          SIGNALQUARRY_PAPER_SECRET_KEY: ${{{{ secrets.SIGNALQUARRY_PAPER_SECRET_KEY }}}}
        run: |
          set +e
          sqy --json paper run-once --alias {alias}
          code=$?
          case $code in 0|69|75) exit 0 ;; *) exit $code ;; esac
      - name: Save journal
        if: always()
        run: |
          git config user.name "signalquarry-paper"
          git config user.email "paper@users.noreply.github.com"
          if git rev-parse --verify -q origin/paper-state >/dev/null; then
            git worktree add --detach ../state origin/paper-state
          else
            git worktree add --detach ../state && git -C ../state switch --orphan paper-state-new
          fi
          mkdir -p ../state/paper/{alias}
          cp paper/{alias}/journal.jsonl ../state/paper/{alias}/ 2>/dev/null || exit 0
          cp paper/{alias}/arm.json ../state/paper/{alias}/ 2>/dev/null || true
          git -C ../state add paper/{alias}
          git -C ../state commit -m "paper {alias} journal" || exit 0
          git -C ../state push origin HEAD:refs/heads/paper-state  # rejected if another run pushed first
""",
    }


def _systemd_poll(root: Path, alias: str, notify_command: Path | None = None) -> dict[str, str]:
    files = _systemd(root, alias, notify_command)
    unit = f"signalquarry-paper-{alias}"
    files[f"{unit}.timer"] = f"""[Unit]
Description=SignalQuarry options poll ({alias}), every minute of the regular session

[Timer]
OnCalendar=Mon..Fri *-*-* 09:30..59:00 America/New_York
OnCalendar=Mon..Fri *-*-* 10..15:*:00 America/New_York
AccuracySec=5s
Persistent=false

[Install]
WantedBy=timers.target
"""
    return files


def _cron_poll(root: Path, alias: str, notify_command: Path | None = None) -> dict[str, str]:
    command = f"cd {shlex.quote(str(root))} && {_command(root, alias, notify_command)} >> paper/{alias}/cron.log 2>&1"
    return {
        f"{alias}.crontab": f"""# SignalQuarry options poll ({alias}): every minute of the regular session. Needs CRON_TZ.
CRON_TZ=America/New_York
30-59 9 * * 1-5 {command}
* 10-15 * * 1-5 {command}
""",
    }


def _launchd_poll(root: Path, alias: str, notify_command: Path | None = None) -> dict[str, str]:
    files = _launchd(root, alias, notify_command)
    name = next(iter(files))
    text = files[name]
    start = text.index("  <key>StartCalendarInterval</key>")
    end = text.index("  </array>", text.index("<array>", start)) + len("  </array>")
    files[name] = (
        text[:start]
        + "  <key>StartInterval</key><integer>60</integer>\n  <!-- polls act only while the market is open -->"
        + text[end:]
    )
    return files


def _systemd_commitments(root: Path, alias: str, strategy: str) -> dict[str, str]:
    """Post-close journal-head commitment with an OpenTimestamps stamp, and a weekly proof upgrade."""
    commit = f"signalquarry-commit-{alias}"
    upgrade = f"signalquarry-ots-upgrade-{alias}"
    command = (
        f"{shlex.quote(sys.executable)} -m signalquarry.cli.main --json commit create --strategy {strategy} "
        f"--alias {alias} --stamp --project {shlex.quote(str(root))}"
    )
    commitments = shlex.quote(str(root / "evidence" / "commitments"))
    return {
        f"{commit}.service": f"""# Commit to the journal head after the close and timestamp it (egress: OpenTimestamps calendars).
[Unit]
Description=SignalQuarry journal-head commitment ({alias})
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory={root}
ExecStart={command}
SuccessExitStatus=75 69
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths={root}/evidence %h/.cache/signalquarry
PrivateTmp=yes
""",
        f"{commit}.timer": f"""[Unit]
Description=SignalQuarry journal-head commitment ({alias}), weekdays 16:30 New York

[Timer]
OnCalendar=Mon..Fri *-*-* 16:30:00 America/New_York
Persistent=true

[Install]
WantedBy=timers.target
""",
        f"{upgrade}.service": f"""# Complete pending OpenTimestamps proofs once they are anchored in a Bitcoin block.
[Unit]
Description=SignalQuarry OpenTimestamps upgrade ({alias})
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory={root}
ExecStart=/bin/sh -c 'command -v ots >/dev/null || exit 0; find {commitments} -name "*.ots" -exec ots upgrade {{}} + || true'
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths={root}/evidence
PrivateTmp=yes
""",
        f"{upgrade}.timer": f"""[Unit]
Description=SignalQuarry OpenTimestamps upgrade ({alias}), weekly

[Timer]
OnCalendar=Sat *-*-* 10:00:00 America/New_York
Persistent=true

[Install]
WantedBy=timers.target
""",
    }


def write_schedule(
    root: Path,
    config: PaperDeploymentV1,
    target: str,
    *,
    poll: bool = False,
    notify_command: Path | None = None,
) -> list[Path]:
    """``poll``: an options deployment, which runs every minute of the regular session."""
    builders = {"systemd": _systemd, "launchd": _launchd, "cron": _cron, "github-actions": _github_actions}
    if poll:
        if target == "github-actions":
            raise ValueError("SCHEDULE_TARGET_UNSUPPORTED_FOR_OPTIONS")
        builders = {"systemd": _systemd_poll, "launchd": _launchd_poll, "cron": _cron_poll}
    if notify_command is not None and (not notify_command.is_absolute() or target == "github-actions"):
        raise ValueError("SCHEDULE_NOTIFY_COMMAND_UNSUPPORTED")
    directory = root / "paper" / "schedule" / target
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    files = (
        builders[target](root, config.alias)
        if target == "github-actions"
        else builders[target](root, config.alias, notify_command)
    )
    if target == "systemd":
        files.update(_systemd_commitments(root, config.alias, config.strategy))
    for name, text in files.items():
        path = directory / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written

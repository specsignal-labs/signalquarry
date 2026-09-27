"""Deploy a paper runner release to a host, refusing risky moments.

    python deploy/deploy.py --host runner@vm --alias momentum-a            # dry run: prints the plan
    python deploy/deploy.py --host runner@vm --alias momentum-a --execute

Guards: never within market hours ±30 minutes (09:00–16:30 New York, weekdays); never
while any path in deploy/config.toml `forbidden_markers` exists on the host (for
example another system's deploy-in-progress marker; checked read-only). Releases go to
/srv/signalquarry/<alias>/releases/<timestamp>/ and `current` flips atomically.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tomllib
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
NEW_YORK = ZoneInfo("America/New_York")


def market_blackout(now: datetime) -> bool:
    local = now.astimezone(NEW_YORK)
    return local.weekday() < 5 and time(9, 0) <= local.time() <= time(16, 30)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = (
        tomllib.loads((ROOT / "deploy" / "config.toml").read_text())
        if (ROOT / "deploy" / "config.toml").is_file()
        else {}
    )
    now = datetime.now(UTC)
    if market_blackout(now):
        print("REFUSED: inside market hours ±30 minutes (New York)")
        return 2
    for marker in config.get("forbidden_markers", []):
        probe = subprocess.run(["ssh", args.host, f"test -e {shlex.quote(marker)}"], check=False)
        if probe.returncode == 0:
            print(f"REFUSED: marker present on host: {marker}")
            return 2
    release = f"/srv/signalquarry/{args.alias}/releases/{now:%Y%m%dT%H%M%SZ}"
    steps = [
        ["uv", "build", "--wheel", "--out-dir", "dist"],
        ["ssh", args.host, f"mkdir -p {release}"],
        [
            "rsync",
            "-a",
            "--delete",
            "--exclude",
            ".git",
            "--exclude",
            ".signalquarry",
            "--exclude",
            "deals",
            "./",
            f"{args.host}:{release}/",
        ],
        [
            "ssh",
            args.host,
            f"ln -sfn {release} /srv/signalquarry/{args.alias}/current.new && mv -T /srv/signalquarry/{args.alias}/current.new /srv/signalquarry/{args.alias}/current",
        ],
    ]
    for step in steps:
        print(("RUN  " if args.execute else "PLAN ") + shlex.join(step))
        if args.execute and subprocess.run(step, cwd=ROOT, check=False).returncode != 0:
            print("FAILED; the previous release stays current")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Prepare an NDA data-room package for one family and one recipient.

    python tools/dataroom.py --family momentum --recipient fund-a

Exports the family's NDA-tier bundle into deals/<recipient>/<date>/ (git-ignored) and
writes a watermark record binding the bundle hash to the recipient. Upload it to a
private repository created for this deal only, with a read-only collaborator and a
set revocation date. No code before a letter of intent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--family", required=True)
    parser.add_argument("--recipient", required=True, help="short slug for the counterparty")
    args = parser.parse_args()
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    out = ROOT / "deals" / args.recipient / today / "bundle"
    completed = subprocess.run(
        ["sqy", "--json", "evidence", "export", "--family", args.family, "--tier", "nda", "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    envelope = json.loads(completed.stdout)
    if envelope["status"] != "ok":
        print(json.dumps(envelope, indent=2))
        return 1
    nonce = secrets.token_hex(16)
    watermark = hashlib.sha256(
        f"{envelope['data']['bundle_hash']}|{args.recipient}|{today}|{nonce}".encode()
    ).hexdigest()
    record = {
        "recipient": args.recipient,
        "family": args.family,
        "date": today,
        "bundle_hash": envelope["data"]["bundle_hash"],
        "watermark": f"sha256:{watermark}",
        "nonce": nonce,
    }
    (out.parent / "watermark.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"package ready in {out.parent}; share it only through a per-deal private repository")
    return 0


if __name__ == "__main__":
    sys.exit(main())

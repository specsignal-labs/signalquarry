# SPDX-License-Identifier: Apache-2.0
"""Add Bubblewrap user namespace operations to a pinned Docker default profile.

Usage: python evals/codex_seccomp.py docker-default.json > codex-seccomp.json
Download the input from moby/profiles commit 2ceae35d351c156cb5a8efc0fdc4a08cf94569d8.
This changes only the disposable container profile, never host kernel settings.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BASE_SHA256 = "6416b47770785a41ac59073cdc77d9fe98517df2799dc83ef207e622de3053f6"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    content = Path(sys.argv[1]).read_bytes()
    if hashlib.sha256(content).hexdigest() != BASE_SHA256:
        raise SystemExit("Docker baseline profile differs from the reviewed revision")
    profile = json.loads(content)
    profile["syscalls"].extend(
        [
            {
                "names": ["clone"],
                "action": "SCMP_ACT_ALLOW",
                "args": [
                    {"index": 0, "value": 0x10000000, "valueTwo": 0x10000000, "op": "SCMP_CMP_MASKED_EQ"}
                ],
            },
            {"names": ["unshare", "mount", "umount2", "pivot_root"], "action": "SCMP_ACT_ALLOW"},
        ]
    )
    print(json.dumps(profile))


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Progress events: one JSON object per line on stderr, never on stdout.

Silent unless the CLI enables it (JSON mode, or ``SIGNALQUARRY_PROGRESS=1``). Stdout
keeps exactly one envelope; progress is advisory and never part of the evidence.
"""

from __future__ import annotations

import json
import time
from typing import Any, TextIO

_stream: TextIO | None = None
_started = time.monotonic()


def configure(stream: TextIO | None) -> None:
    global _stream, _started
    _stream, _started = stream, time.monotonic()


def emit(stage: str, **fields: Any) -> None:
    """Write ``{"type": "progress", "stage": ..., "elapsed_ms": ..., **fields}`` if enabled."""
    if _stream is None:
        return
    event = {
        "type": "progress",
        "stage": stage,
        "elapsed_ms": int((time.monotonic() - _started) * 1000),
        **fields,
    }
    try:
        _stream.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        _stream.flush()
    except (OSError, ValueError):  # a closed stderr must not fail the command
        pass

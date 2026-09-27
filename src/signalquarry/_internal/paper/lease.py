# SPDX-License-Identifier: Apache-2.0
"""``RunLease``: one state-changing paper command per deployment at a time (POSIX ``flock``)."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from types import TracebackType

from signalquarry._internal.paper.models import PaperError


class RunLease:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def __enter__(self) -> RunLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise PaperError("RUN_LEASE_BUSY", "busy", str(self.path)) from exc
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._fd = fd
        return self

    def __exit__(
        self, kind: type[BaseException] | None, value: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

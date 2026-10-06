# SPDX-License-Identifier: Apache-2.0
"""The benchmark's ceilings: strict, configurable only for time, and fixed for memory."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools" / "bench.py"


def _bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sqy_bench_under_test", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_default_ceiling_is_three_minutes_and_two_gigabytes() -> None:
    bench = _bench()
    assert (bench.DEFAULT_CEILING_SECONDS, bench.MEMORY_CEILING_MB) == (180.0, 2000.0)
    assert bench.ceiling_verdict(143.9, 1041) == "ok"
    assert bench.ceiling_verdict(179.99, 1999.9) == "ok"
    assert bench.ceiling_verdict(180.0, 100) == "FAIL"
    assert bench.ceiling_verdict(10, 2000.0) == "FAIL"
    assert bench.ceiling_verdict(235.3, 858) == "FAIL"


def test_a_looser_time_ceiling_does_not_loosen_memory() -> None:
    bench = _bench()
    assert bench.ceiling_verdict(259.3, 856, 300.0) == "ok"
    assert bench.ceiling_verdict(299.99, 856, 300.0) == "ok"
    assert bench.ceiling_verdict(300.0, 856, 300.0) == "FAIL"
    assert bench.ceiling_verdict(100.0, 2000.0, 300.0) == "FAIL"


def test_the_command_line_accepts_a_numeric_spooled_ceiling_only() -> None:
    bench = _bench()
    with pytest.raises(SystemExit) as stopped:
        bench.main(["--spooled-ceiling", "soon"])
    assert stopped.value.code == 2
    with pytest.raises(SystemExit) as helped:
        bench.main(["--help"])
    assert helped.value.code == 0

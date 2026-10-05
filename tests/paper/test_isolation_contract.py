# SPDX-License-Identifier: Apache-2.0
"""What the isolated strategy child actually sees and how its failures reach the kernel."""

from __future__ import annotations

import os
import resource
from decimal import Decimal
from pathlib import Path

import pytest

from signalquarry._internal.data.dataset import truncated, with_pending_session
from signalquarry._internal.paper import isolate
from signalquarry._internal.paper.isolate import plan_in_child
from signalquarry._internal.paper.models import PaperError
from signalquarry.sdk import Ctx, Decision, definition_of, strategy
from tests.paper.harness import Momentum, paper_spec, parity_dataset


@strategy(params=Momentum, lookback=lambda p: p.lookback)
def probe(ctx: Ctx, p: Momentum) -> Decision:
    import os as child_os
    import resource as child_resource
    import tempfile

    return Decision.target(
        {},
        "WAIT",
        state={
            "env": dict(child_os.environ),
            "cwd": child_os.getcwd(),
            "tmp": tempfile.gettempdir(),
            "cpu": list(child_resource.getrlimit(child_resource.RLIMIT_CPU)),
            "argv0": __import__("sys").argv[:2],
        },
    )


@strategy(params=Momentum, lookback=lambda p: p.lookback)
def raises(ctx: Ctx, p: Momentum) -> Decision:
    raise RuntimeError("kaboom " + "x" * 700)


@strategy(params=Momentum, lookback=lambda p: p.lookback)
def hard_exit(ctx: Ctx, p: Momentum) -> Decision:
    import os as child_os

    child_os._exit(3)


@strategy(params=Momentum, lookback=lambda p: p.lookback)
def writes_garbage_then_fails(ctx: Ctx, p: Momentum) -> Decision:
    import sys

    sys.stderr.write("first line\nlast line of the failure\n")
    sys.stderr.flush()
    import os as child_os

    child_os._exit(5)


def _inputs():
    data = parity_dataset()
    session = data.sessions[80]
    pending = with_pending_session(truncated(data, data.index_of(session)), session)
    kwargs = dict(quantity={}, cash=Decimal(10000), state={}, last_target=None, target_complete=True)
    return pending, kwargs


def _run(definition, **options):
    pending, kwargs = _inputs()
    return plan_in_child(paper_spec(), definition_of(definition), Momentum(), pending, **kwargs, **options)


def test_the_child_environment_is_exactly_the_scrubbed_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", "visible-to-parent-only")
    monkeypatch.setenv("SECRET_TOKEN", "x")
    state = _run(probe).state
    env = state["env"]
    home = env["HOME"]
    assert set(env) >= {"PATH", "HOME", "TMPDIR", "SIGNALQUARRY_CONFIG_DIR", "SIGNALQUARRY_CACHE_DIR"}
    assert env["PATH"] == os.defpath
    assert env["TMPDIR"] == home == env["SIGNALQUARRY_CONFIG_DIR"] == env["SIGNALQUARRY_CACHE_DIR"]
    assert Path(home).name.startswith("sq-isolate-")
    assert env["SIGNALQUARRY_OFFLINE"] == "1"
    assert env["LANG"] == "C.UTF-8"
    assert "APCA_API_KEY_ID" not in env and "SECRET_TOKEN" not in env
    assert {key for key in env if key not in {"LC_CTYPE", "__CF_USER_TEXT_ENCODING"}} == {
        "PATH",
        "HOME",
        "TMPDIR",
        "SIGNALQUARRY_CONFIG_DIR",
        "SIGNALQUARRY_CACHE_DIR",
        "SIGNALQUARRY_OFFLINE",
        "LANG",
    }
    assert Path(state["cwd"]).resolve() == Path(home).resolve()
    assert Path(state["tmp"]).resolve() == Path(home).resolve()
    assert not Path(home).exists()  # removed after the run


def test_the_child_has_a_cpu_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isolate.CPU_SECONDS == 60 and isolate.MEMORY_BYTES == 4 * 1024**3
    soft, hard = _run(probe).state["cpu"]
    assert (soft, hard) == (isolate.CPU_SECONDS, isolate.CPU_SECONDS + 5)
    assert resource.getrlimit(resource.RLIMIT_CPU)[0] != isolate.CPU_SECONDS  # the parent keeps its own


def test_a_strategy_exception_becomes_a_blocked_paper_error_with_type_and_message() -> None:
    with pytest.raises(PaperError) as info:
        _run(raises)
    assert info.value.code == "PAPER_STRATEGY_FAILED" and info.value.status == "blocked"
    assert info.value.detail.startswith("RuntimeError: kaboom xxx")
    assert len(info.value.detail) == 500


def test_a_child_that_dies_without_output_reports_its_exit_status_or_last_stderr_line() -> None:
    with pytest.raises(PaperError) as died:
        _run(hard_exit)
    assert died.value.code == "PAPER_STRATEGY_FAILED" and died.value.status == "blocked"
    assert died.value.detail == "exit 3"
    with pytest.raises(PaperError) as noisy:
        _run(writes_garbage_then_fails)
    assert noisy.value.detail == "last line of the failure"


def test_a_timeout_names_the_limit_and_blocks() -> None:
    from tests.paper.test_isolation import spins

    with pytest.raises(PaperError) as info:
        _run(spins, timeout=2.4)
    assert (info.value.code, info.value.status, info.value.detail) == (
        "PAPER_STRATEGY_TIMEOUT",
        "blocked",
        "decide exceeded 2s",
    )


@strategy(params=Momentum, lookback=lambda p: p.lookback)
def stderr_with_bad_bytes(ctx: Ctx, p: Momentum) -> Decision:
    import os as child_os
    import sys

    sys.stderr.buffer.write(b"bad \xff byte " + b"y" * 700 + b"\n")
    sys.stderr.flush()
    child_os._exit(9)


def test_undecodable_child_stderr_is_replaced_not_fatal_and_capped_at_500_characters() -> None:
    with pytest.raises(PaperError) as info:
        _run(stderr_with_bad_bytes)
    assert info.value.code == "PAPER_STRATEGY_FAILED"
    assert info.value.detail.startswith("bad � byte yyy")
    assert len(info.value.detail) == 500


def test_the_default_decision_timeout_is_two_minutes() -> None:
    import inspect

    assert inspect.signature(plan_in_child).parameters["timeout"].default == 120.0

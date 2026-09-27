# SPDX-License-Identifier: Apache-2.0
"""Run a paper session's decide-and-size step in a separate process.

The child gets a scrubbed environment (no broker or data credentials, an empty
config directory, a temporary home, ``SIGNALQUARRY_OFFLINE=1``), CPU and memory
limits, and only the inputs it needs over stdin. It re-imports the strategy by
module and name and runs the same ``plan_pre_open`` as the backtest.

This keeps credentials and the broker session out of strategy code's process. It is
**not a sandbox**: code running as the same OS user can still read files that user
can read. Run paper deployments under a dedicated user for stronger separation.
"""

from __future__ import annotations

import json
import os
import pickle
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.data.dataset import Dataset
from signalquarry._internal.engine.backtest import EngineError, PreOpenPlan, plan_pre_open
from signalquarry._internal.paper.models import PaperError
from signalquarry.sdk.strategy import Params, StrategyDef, definition_of

CPU_SECONDS = 60
MEMORY_BYTES = 4 * 1024**3


@dataclass(frozen=True)
class _Outcome:
    ok: bool
    plan: PreOpenPlan | None = None
    error: str = ""
    engine_error: bool = False


def _limits() -> None:  # pragma: no cover - runs in the child before exec
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS + 5))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    except (ValueError, OSError):
        pass  # not supported on every platform (macOS)


def plan_in_child(
    spec: StrategySpecV1,
    definition: StrategyDef,
    params: Params,
    dataset: Dataset,
    *,
    timeout: float = 120.0,
    **kwargs: Any,
) -> PreOpenPlan:
    header = {"sys_path": sys.path, "module": definition.module, "name": definition.name}
    payload = json.dumps(header).encode() + b"\n" + pickle.dumps((spec, params, dataset, kwargs))
    with tempfile.TemporaryDirectory(prefix="sq-isolate-") as home:
        env = {
            "PATH": os.defpath,
            "HOME": home,
            "TMPDIR": home,
            "SIGNALQUARRY_CONFIG_DIR": home,
            "SIGNALQUARRY_CACHE_DIR": home,
            "SIGNALQUARRY_OFFLINE": "1",
            "LANG": "C.UTF-8",
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "signalquarry._internal.paper.isolate"],
                input=payload,
                capture_output=True,
                env=env,
                cwd=home,
                timeout=timeout,
                preexec_fn=_limits,  # noqa: PLW1509 - single-threaded CLI process
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PaperError("PAPER_STRATEGY_TIMEOUT", "blocked", f"decide exceeded {timeout:.0f}s") from exc
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [
            f"exit {completed.returncode}"
        ]
        raise PaperError("PAPER_STRATEGY_FAILED", "blocked", detail[0][:500])
    outcome: _Outcome = pickle.loads(completed.stdout)  # noqa: S301 - produced by our own child process
    if outcome.ok and outcome.plan is not None:
        return outcome.plan
    if outcome.engine_error:
        raise EngineError(outcome.error)
    raise PaperError("PAPER_STRATEGY_FAILED", "blocked", outcome.error[:500])


def _child() -> int:  # pragma: no cover - exercised through plan_in_child in a subprocess
    raw = sys.stdin.buffer.read()
    header_bytes, _, body = raw.partition(b"\n")
    header = json.loads(header_bytes)
    sys.path[:] = header["sys_path"]
    try:
        import importlib

        module = importlib.import_module(header["module"])
        definition = definition_of(getattr(module, header["name"]))
        spec, params, dataset, kwargs = pickle.loads(body)  # noqa: S301 - from the parent process
        outcome = _Outcome(True, plan_pre_open(spec, definition, params, dataset, **kwargs))
    except EngineError as exc:
        outcome = _Outcome(False, error=str(exc), engine_error=True)
    except Exception as exc:  # noqa: BLE001 - reported to the parent, which halts the deployment
        outcome = _Outcome(False, error=f"{type(exc).__name__}: {exc}")
    sys.stdout.buffer.write(pickle.dumps(outcome))
    return 0


if __name__ == "__main__":  # pragma: no cover
    # Re-import under the package name so pickled results reference importable classes.
    from signalquarry._internal.paper import isolate as _isolate

    raise SystemExit(_isolate._child())

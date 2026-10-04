# SPDX-License-Identifier: Apache-2.0
"""Run deterministic behavior probes for an agent-authored strategy."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import sys
import tomllib
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _load_strategy(root: Path, strategy_id: str) -> tuple[Any, Any, dict[str, Any]]:
    """Load the registered strategy and its colocated spec from the project."""
    from signalquarry.sdk.strategy import definition_of

    config = tomllib.loads((root / "signalquarry.toml").read_text(encoding="utf-8"))
    modules = (config.get("strategies") or {}).get("modules")
    if not isinstance(modules, list) or not all(isinstance(item, str) for item in modules):
        raise ValueError("project has no valid registered strategy modules")
    source_root = (root / "src").resolve()
    matching_module = None
    matching_spec = None
    for name in modules:
        parts = name.split(".")
        if not all(part.isidentifier() for part in parts):
            continue
        relative = Path(*parts)
        candidates = (source_root / relative / "__init__.py", source_root / relative.with_suffix(".py"))
        module_file = next((path for path in candidates if path.is_file()), None)
        if module_file is None:
            continue
        module_file = module_file.resolve()
        if not module_file.is_relative_to(source_root):
            continue
        spec_path = module_file.with_name("strategy.yaml")
        if not spec_path.is_file():
            continue
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict) or spec.get("id") != strategy_id:
            continue
        matching_module = name
        matching_spec = spec
        break
    if matching_module is None or matching_spec is None:
        raise ValueError(f"registered strategy {strategy_id!r} has no matching strategy.yaml")
    source = str(source_root)
    if source not in sys.path:
        sys.path.insert(0, source)
    module = importlib.import_module(matching_module)
    decision_function = getattr(module, "decide", None)
    definition = definition_of(decision_function)
    params = definition.params(**(matching_spec.get("params") or {}))
    return definition, params, matching_spec


def _same_value(actual: object, expected: object) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, str):
        try:
            return Decimal(str(actual)) == Decimal(expected)
        except Exception:  # noqa: BLE001 - non-numeric strings compare as text
            return actual == expected
    return actual == expected


def _check_declared_spec(spec: dict[str, Any], contract: dict[str, Any], errors: list[str]) -> None:
    expected_kind = contract.get("kind", "equity_daily")
    if spec.get("kind", "equity_daily") != expected_kind:
        errors.append(f"strategy kind must be {expected_kind}")
    data = spec.get("data") or {}
    if data.get("feed") != contract.get("feed", "synthetic"):
        errors.append("strategy data feed does not match the synthetic task")
    if data.get("symbols") != contract.get("symbols"):
        errors.append("strategy symbols do not match the requested symbols")
    params = spec.get("params") or {}
    for name, expected in (contract.get("params") or {}).items():
        if name not in params or not _same_value(params[name], expected):
            errors.append(f"strategy parameter {name!r} does not match the task")
    expected_options = contract.get("options") or {}
    actual_options = spec.get("options") or {}
    for name, expected in expected_options.items():
        if actual_options.get(name) != expected:
            errors.append(f"strategy options field {name!r} does not match the task")


def _bars(symbol: str, closes: np.ndarray) -> Any:
    from signalquarry.sdk.context import Bars

    values = np.asarray(closes, dtype=np.float64)
    return Bars(
        symbol=symbol,
        sessions=np.arange(len(values)).astype("datetime64[D]"),
        open=values,
        high=values,
        low=values,
        close=values,
        volume=np.ones(len(values), dtype=np.float64),
    )


def _check_sma_crossover(definition: Any, params: Any, contract: dict[str, Any], errors: list[str]) -> None:
    from signalquarry.sdk.context import Ctx

    symbol = str(contract["params"]["symbol"])
    fast = int(contract["params"]["fast"])
    slow = int(contract["params"]["slow"])
    weight = Decimal(str(contract["params"]["weight"]))
    if fast >= slow:
        errors.append("SMA crossover task requires fast period below slow period")
        return
    for direction, recent_level, expected_reason, expected_weights in (
        ("rising", 110.0, "FAST_ABOVE_SLOW", {symbol: weight}),
        ("falling", 90.0, "FAST_BELOW_SLOW", {}),
    ):
        closes = np.full(slow, 100.0)
        closes[-fast:] = recent_level
        # Keep the current price identical in both probes. The decision must use
        # the 50/200-session averages, not a threshold on the latest close.
        closes[-1] = 100.0
        context = Ctx(
            decision_session=date(2025, 1, 2),
            _bars={symbol: _bars(symbol, closes)},
            positions={},
            weights={},
            cash=Decimal("100000"),
            equity=Decimal("100000"),
            state={},
        )
        decision = definition.decide(context, params)
        if decision.action != "target" or decision.weights != expected_weights:
            errors.append(f"SMA crossover {direction} probe returned the wrong target")
        if decision.reason_codes != (expected_reason,):
            errors.append(f"SMA crossover {direction} probe returned the wrong reason code")


def _check_cash_secured_put(
    definition: Any, params: Any, contract: dict[str, Any], errors: list[str]
) -> None:
    from signalquarry.sdk.options import LegView, OptionsCtx, WheelView

    underlying = str(contract["params"]["underlying"])
    minimum_dte = int(contract["params"]["min_dte"])
    maximum_dte = int(contract["params"]["max_dte"])
    otm = Decimal(str(contract["params"]["otm"]))
    threshold = Decimal(str(contract["params"]["take_profit"]))

    def make_context(*, wheel_state: str = "flat", captured: Decimal | None = None) -> Any:
        wheel = WheelView(
            underlying=underlying,
            state=wheel_state,
            shares=100 if wheel_state == "long_shares" else 0,
            share_cost_basis=Decimal("100") if wheel_state == "long_shares" else None,
        )
        legs = {}
        if captured is not None:
            legs[underlying] = LegView(
                symbol=underlying,
                right="PUT",
                strike=Decimal("95"),
                expiration=date(2025, 2, 1),
                dte=30,
                entry_credit=Decimal("2"),
                close_cost=Decimal("1"),
                captured_fraction=captured,
            )
        return OptionsCtx(
            decision_time=datetime(2025, 1, 2, tzinfo=UTC),
            _bars={},
            _wheels={underlying: wheel},
            _legs=legs,
            spots={underlying: Decimal("100")},
            cash=Decimal("100000"),
            equity=Decimal("100000"),
            state={},
        )

    opened = definition.decide(make_context(), params)
    selector = opened.selector
    if (
        opened.action != "open"
        or opened.underlying != underlying
        or opened.reason_codes != ("SELL_PUT",)
        or selector is None
        or selector.underlying != underlying
        or selector.right != "PUT"
        or (selector.min_dte, selector.max_dte) != (minimum_dte, maximum_dte)
        or selector.strike.kind != "otm"
        or selector.strike.value != otm
    ):
        errors.append("flat-wheel probe did not open the requested cash-secured put selector")

    for captured, expected_action, expected_reason in (
        (threshold, "hold", "HOLD_LEG"),
        (threshold + Decimal("0.01"), "close", "TAKE_PROFIT"),
    ):
        decision = definition.decide(make_context(captured=captured), params)
        if (
            decision.action != expected_action
            or decision.reason_codes != (expected_reason,)
            or decision.selector is not None
            or (expected_action == "close" and decision.underlying != underlying)
        ):
            errors.append(f"open-leg probe at captured fraction {captured} returned the wrong action")

    assigned = definition.decide(make_context(wheel_state="long_shares"), params)
    if (
        assigned.action != "hold"
        or assigned.reason_codes != ("HOLD_SHARES",)
        or assigned.selector is not None
    ):
        errors.append("assigned-shares probe must hold without opening another contract")


def check_behavior(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Return task-specific behavior failures without trusting agent-authored claims."""
    errors: list[str] = []
    strategy_id = contract.get("strategy_id")
    if not isinstance(strategy_id, str):
        return {"ok": False, "errors": ["behavior contract has no strategy id"]}
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                definition, params, spec = _load_strategy(root, strategy_id)
                _check_declared_spec(spec, contract, errors)
                check = contract.get("check")
                if check == "sma_crossover":
                    _check_sma_crossover(definition, params, contract, errors)
                elif check == "cash_secured_put":
                    _check_cash_secured_put(definition, params, contract, errors)
                else:
                    errors.append(f"unsupported behavior contract {check!r}")
    except Exception as exc:  # noqa: BLE001 - invalid strategy code is a failed eval
        errors.append(f"behavior probe failed: {type(exc).__name__}: {exc}")
    return {"ok": not errors, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    try:
        contract = json.loads(args.contract)
    except ValueError as exc:
        print(json.dumps({"ok": False, "errors": [f"invalid behavior contract: {exc}"]}))
        return 1
    if not isinstance(contract, dict):
        print(json.dumps({"ok": False, "errors": ["behavior contract must be an object"]}))
        return 1
    result = check_behavior(args.project.resolve(), contract)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0
"""The ``@strategy`` decorator: the only way to declare a strategy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from pydantic import BaseModel, ConfigDict

from signalquarry.sdk.context import Ctx
from signalquarry.sdk.decision import Decision


class Params(BaseModel):
    """Base class for strategy parameters. Field bounds double as the sweep space."""

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class StrategyDef:
    decide: Callable[[Ctx, Params], Decision]
    params: type[Params]
    lookback: Callable[[Params], int]
    module: str
    name: str
    kind: str = "equity"  # "equity" (Decision) or "options" (OptionsDecision)


ATTRIBUTE = "__signalquarry_strategy__"


def strategy[P: Params](
    *, params: type[P], lookback: Callable[[P], int]
) -> Callable[[Callable[[Ctx, P], Decision]], Callable[[Ctx, P], Decision]]:
    """Declare a pure decision function ``decide(ctx, params) -> Decision``.

    ``lookback(params)`` is the number of completed sessions the strategy needs;
    the engine returns ``unavailable(INSUFFICIENT_HISTORY)`` until they exist and
    passes exactly that many bars.
    """
    if not (isinstance(params, type) and issubclass(params, Params)):  # pyright: ignore[reportUnnecessaryIsInstance] -- untyped callers
        raise TypeError("STRATEGY_PARAMS_MUST_SUBCLASS_PARAMS")

    def wrap(function: Callable[[Ctx, P], Decision]) -> Callable[[Ctx, P], Decision]:
        # The engine always calls ``decide`` with an instance of ``params``.
        definition = StrategyDef(
            cast(Callable[[Ctx, Params], Decision], function),
            params,
            cast(Callable[[Params], int], lookback),
            function.__module__,
            getattr(function, "__name__", "decide"),
        )
        setattr(function, ATTRIBUTE, definition)
        return function

    return wrap


def definition_of(obj: object) -> StrategyDef:
    found = getattr(obj, ATTRIBUTE, None)
    if not isinstance(found, StrategyDef):
        raise TypeError("NOT_A_STRATEGY")
    return found

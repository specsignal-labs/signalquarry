# SPDX-License-Identifier: Apache-2.0
"""Pure, cross-sectional factor authoring over completed-session panels."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np

from signalquarry.sdk.strategy import Params


@dataclass(frozen=True)
class FactorCtx:
    """A read-only window ending strictly before the decision session.

    Columns follow ``universe`` order. Missing observations are NaN in numeric
    panels and false in ``present``. Prices are raw, without lifecycle adjustment.
    The engine constructs this from a verified, truncated panel and an explicit
    universe; its membership provenance is checked separately.
    """

    decision_session: date
    sessions: tuple[date, ...]
    universe: tuple[str, ...]
    _panels: Mapping[str, np.ndarray]

    def panel(self, field: str) -> np.ndarray:
        try:
            return self._panels[field]
        except KeyError:
            raise KeyError(f"FACTOR_PANEL_UNKNOWN:{field}") from None


@dataclass(frozen=True)
class FactorDef:
    score: Callable[[FactorCtx, Params], Mapping[str, float]]
    params: type[Params]
    lookback: Callable[[Params], int]
    module: str
    name: str


ATTRIBUTE = "__signalquarry_factor__"


def factor[P: Params](
    *, params: type[P], lookback: Callable[[P], int]
) -> Callable[[Callable[[FactorCtx, P], Mapping[str, float]]], Callable[[FactorCtx, P], Mapping[str, float]]]:
    """Declare a factor returning finite scores for covered universe symbols."""
    if not (isinstance(params, type) and issubclass(params, Params)):  # pyright: ignore[reportUnnecessaryIsInstance] -- untyped callers
        raise TypeError("FACTOR_PARAMS_MUST_SUBCLASS_PARAMS")

    def wrap(
        function: Callable[[FactorCtx, P], Mapping[str, float]],
    ) -> Callable[[FactorCtx, P], Mapping[str, float]]:
        setattr(
            function,
            ATTRIBUTE,
            FactorDef(
                cast(Callable[[FactorCtx, Params], Mapping[str, float]], function),
                params,
                cast(Callable[[Params], int], lookback),
                function.__module__,
                getattr(function, "__name__", "score"),
            ),
        )
        return function

    return wrap


def factor_definition_of(obj: object) -> FactorDef:
    found = getattr(obj, ATTRIBUTE, None)
    if not isinstance(found, FactorDef):
        raise TypeError("NOT_A_FACTOR")
    return found


def checked_scores(scores: Mapping[str, float], universe: tuple[str, ...]) -> dict[str, float]:
    """Reject undeclared or nonfinite scores; missing keys mean no coverage."""
    if not isinstance(scores, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance] -- untyped factor code
        raise TypeError("FACTOR_SCORES_INVALID")
    allowed = set(universe)
    result: dict[str, float] = {}
    for symbol, score in scores.items():
        if symbol not in allowed or not isinstance(symbol, str):  # pyright: ignore[reportUnnecessaryIsInstance] -- untyped factor code
            raise ValueError("FACTOR_SYMBOL_UNDECLARED")
        if isinstance(score, bool) or not isinstance(score, (int, float, np.integer, np.floating)):  # pyright: ignore[reportUnnecessaryIsInstance] -- untyped factor code
            raise TypeError("FACTOR_SCORE_INVALID")
        value = float(score)
        if not np.isfinite(value):
            raise ValueError("FACTOR_SCORE_NONFINITE")
        result[symbol] = value
    return dict(sorted(result.items()))

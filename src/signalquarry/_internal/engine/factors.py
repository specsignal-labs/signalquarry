# SPDX-License-Identifier: Apache-2.0
"""Construct bounded factor contexts from verified, pre-decision panel windows."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from signalquarry._internal.data.dataset import FIELDS, MICRO
from signalquarry._internal.data.panel import PanelWindow
from signalquarry.sdk.factors import FactorCtx, FactorDef, checked_scores
from signalquarry.sdk.strategy import Params


def _immutable(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.tobytes(order="C"), dtype=values.dtype).reshape(values.shape)


def factor_context(window: PanelWindow, universe: tuple[str, ...], *, lookback: int) -> FactorCtx:
    """Select eligible columns and exactly the declared completed-bar history."""
    if type(lookback) is not int or lookback < 1:
        raise ValueError("FACTOR_LOOKBACK_INVALID")
    if len(window.sessions) < lookback:
        raise ValueError("FACTOR_INSUFFICIENT_HISTORY")
    if any(session >= window.decision_session for session in window.sessions):
        raise ValueError("FACTOR_LOOKAHEAD")
    if len(universe) != len(set(universe)) or not set(universe) <= set(window.symbols):
        raise ValueError("FACTOR_UNIVERSE_INVALID")
    shape = (len(window.sessions), len(window.symbols))
    if (
        window.present.shape != shape
        or window.volume.shape != shape
        or any(window.micro[field].shape != shape for field in FIELDS)
    ):
        raise ValueError("FACTOR_PANEL_SHAPE")
    columns = [window.symbols.index(symbol) for symbol in universe]
    recent = slice(-lookback, None)
    present = np.asarray(window.present[recent][:, columns], dtype=np.bool_)
    panels: dict[str, np.ndarray] = {"present": _immutable(present)}
    for field in (*FIELDS, "volume"):
        raw = window.micro[field] if field in FIELDS else window.volume
        values = np.asarray(raw[recent][:, columns], dtype=np.float64)
        if field in FIELDS:
            values /= MICRO
        values[~present] = np.nan
        panels[field] = _immutable(values)
    return FactorCtx(
        decision_session=window.decision_session,
        sessions=window.sessions[recent],
        universe=universe,
        _panels=MappingProxyType(panels),
    )


def run_factor(
    definition: FactorDef, params: Params, window: PanelWindow, universe: tuple[str, ...]
) -> Mapping[str, float]:
    """Evaluate one declared factor; no trial or evidence claim is made here."""
    if not isinstance(params, definition.params):
        raise TypeError("FACTOR_PARAMS_INVALID")
    context = factor_context(window, universe, lookback=definition.lookback(params))
    return MappingProxyType(checked_scores(definition.score(context, params), universe))

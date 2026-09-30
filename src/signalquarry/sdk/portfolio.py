# SPDX-License-Identifier: Apache-2.0
"""Build ordinary long-only strategy decisions from cross-sectional factors."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_DOWN, Decimal
from types import MappingProxyType
from typing import cast

import numpy as np

from signalquarry.sdk.context import Ctx
from signalquarry.sdk.decision import Decision
from signalquarry.sdk.factors import FactorCtx, checked_scores, factor_definition_of
from signalquarry.sdk.strategy import Params
from signalquarry.sdk.xs import zscore

_ID = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_WEIGHT_QUANTUM = Decimal("0.000001")
_FIELDS = ("open", "high", "low", "close", "volume")


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float, np.integer, np.floating))


@dataclass(frozen=True)
class FactorInput:
    """One factor function, its stable portfolio ID, and frozen parameters."""

    id: str
    function: Callable[..., Mapping[str, float]]
    params: Params


@dataclass(frozen=True)
class FactorWeightSnapshot:
    """Prior-only factor weights effective from a later decision date."""

    known_at: date
    effective_from: date
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))


def _immutable(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.tobytes(order="C"), dtype=values.dtype).reshape(values.shape)


def _context(ctx: Ctx, lookback: int) -> FactorCtx | None:
    """Adapt the strategy's completed-bar cross-section to a bounded factor context."""
    symbols = ctx.symbols
    if not symbols:
        raise ValueError("FACTOR_UNIVERSE_INVALID")
    bars = [ctx.bars(symbol) for symbol in symbols]
    if any(len(item) < lookback for item in bars):
        return None

    recent = [item.sessions[-lookback:] for item in bars]
    reference = recent[0]
    if any(not np.array_equal(item, reference) for item in recent[1:]):
        raise ValueError("FACTOR_PANEL_SESSIONS_INVALID")
    sessions = tuple(
        date.fromisoformat(np.datetime_as_string(cast(np.datetime64, item), unit="D")) for item in reference
    )
    if any(session >= ctx.decision_session for session in sessions):
        raise ValueError("FACTOR_LOOKAHEAD")

    raw = {
        field: np.column_stack(
            [np.asarray(getattr(item, field)[-lookback:], dtype=np.float64) for item in bars]
        )
        for field in _FIELDS
    }
    present = np.logical_and.reduce([np.isfinite(values) for values in raw.values()])
    panels: dict[str, np.ndarray] = {"present": _immutable(present.astype(np.bool_))}
    for field, values in raw.items():
        values[~present] = np.nan
        panels[field] = _immutable(values)
    return FactorCtx(
        decision_session=ctx.decision_session,
        sessions=sessions,
        universe=symbols,
        _panels=MappingProxyType(panels),
    )


def _checked_weights(
    inputs: tuple[FactorInput, ...],
    schedule: Sequence[FactorWeightSnapshot],
    decision_session: date,
) -> dict[str, float] | None:
    ids = {item.id for item in inputs}
    previous_effective: date | None = None
    for snapshot in schedule:
        if not isinstance(snapshot, FactorWeightSnapshot):  # pyright: ignore[reportUnnecessaryIsInstance] -- untyped callers
            raise TypeError("FACTOR_WEIGHT_SCHEDULE_INVALID")
        if (
            type(snapshot.known_at) is not date
            or type(snapshot.effective_from) is not date
            or snapshot.known_at >= snapshot.effective_from
            or (previous_effective is not None and snapshot.effective_from <= previous_effective)
        ):
            raise ValueError("FACTOR_WEIGHT_SCHEDULE_INVALID")
        previous_effective = snapshot.effective_from
        if not isinstance(snapshot.weights, Mapping) or not snapshot.weights:  # pyright: ignore[reportUnnecessaryIsInstance] -- untyped callers
            raise ValueError("FACTOR_WEIGHT_SCHEDULE_INVALID")
        if set(snapshot.weights) - ids:
            raise ValueError("FACTOR_WEIGHT_ID_UNKNOWN")
        for factor_id, weight in snapshot.weights.items():
            if (
                not isinstance(factor_id, str)  # pyright: ignore[reportUnnecessaryIsInstance] -- untyped callers
                or not _is_number(weight)
                or not np.isfinite(weight)
                or weight < 0
            ):
                raise ValueError("FACTOR_WEIGHT_INVALID")
        if not any(weight > 0 for weight in snapshot.weights.values()):
            raise ValueError("FACTOR_WEIGHT_INVALID")

    eligible = [item for item in schedule if item.effective_from <= decision_session]
    if not eligible:
        return None
    selected = eligible[-1]
    if selected.known_at >= decision_session:
        return None
    total = sum(float(value) for value in selected.weights.values())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("FACTOR_WEIGHT_INVALID")
    return {factor_id: float(value) / total for factor_id, value in selected.weights.items()}


def _validate_inputs(inputs: Sequence[FactorInput]) -> tuple[FactorInput, ...]:
    items = tuple(inputs)
    if not items:
        raise ValueError("FACTOR_INPUTS_EMPTY")
    if any(not isinstance(item, FactorInput) for item in items):  # pyright: ignore[reportUnnecessaryIsInstance] -- untyped callers
        raise TypeError("FACTOR_INPUT_INVALID")
    ids = [item.id for item in items]
    if any(not isinstance(item.id, str) or not _ID.fullmatch(item.id) for item in items):  # pyright: ignore[reportUnnecessaryIsInstance] -- untyped callers
        raise ValueError("FACTOR_PORTFOLIO_ID_INVALID")
    if len(ids) != len(set(ids)):
        raise ValueError("FACTOR_PORTFOLIO_ID_DUPLICATE")
    for item in items:
        definition = factor_definition_of(item.function)
        if not isinstance(item.params, definition.params):
            raise TypeError("FACTOR_PARAMS_INVALID")
    return tuple(sorted(items, key=lambda item: item.id))


def factor_portfolio(
    ctx: Ctx,
    factors: Sequence[FactorInput],
    *,
    top_quantile: float,
    max_weight_per_symbol: float,
    turnover_buffer: float = 0.0,
    weight_schedule: Sequence[FactorWeightSnapshot] = (),
) -> Decision:
    """Combine factor z-scores into a capped target using the ordinary strategy path.

    With no ``weight_schedule``, factor z-scores receive equal weight. A supplied
    schedule uses its latest row effective on the decision date; each row must
    state an earlier ``known_at`` date. The schedule is caller-produced and its
    statistical provenance is not asserted here.

    Missing factor scores are not imputed. Each symbol combines the factors that
    both have a score and a positive weight. The top ``top_quantile`` of eligible
    symbols is targeted; existing holdings may remain through the wider
    ``top_quantile + turnover_buffer`` exit band. Equal target weights are capped
    at ``max_weight_per_symbol`` and any unused exposure remains cash. The normal
    engine and paper planner apply the configured costs and minimum order size.
    """
    values = (top_quantile, max_weight_per_symbol, turnover_buffer)
    if any(not _is_number(value) for value in values):
        raise ValueError("FACTOR_PORTFOLIO_CONFIG_INVALID")
    if (
        not np.isfinite(values).all()
        or not 0 < top_quantile <= 1
        or not 0 < max_weight_per_symbol <= 1
        or not 0 <= turnover_buffer <= 1 - top_quantile
    ):
        raise ValueError("FACTOR_PORTFOLIO_CONFIG_INVALID")

    inputs = _validate_inputs(factors)
    if weight_schedule:
        weights = _checked_weights(inputs, weight_schedule, ctx.decision_session)
        if weights is None:
            return Decision.unavailable("FACTOR_PORTFOLIO_UNAVAILABLE")
    else:
        weights = {item.id: 1.0 / len(inputs) for item in inputs}

    symbols = ctx.symbols
    combined = np.full(len(symbols), np.nan, dtype=np.float64)
    denominator = np.zeros(len(symbols), dtype=np.float64)
    numerator = np.zeros(len(symbols), dtype=np.float64)
    for item in inputs:
        weight = weights.get(item.id, 0.0)
        if weight <= 0:
            continue
        definition = factor_definition_of(item.function)
        lookback = definition.lookback(item.params)
        if type(lookback) is not int or lookback < 1:
            raise ValueError("FACTOR_LOOKBACK_INVALID")
        factor_ctx = _context(ctx, lookback)
        if factor_ctx is None:
            return Decision.unavailable("INSUFFICIENT_HISTORY")
        raw = checked_scores(definition.score(factor_ctx, item.params), symbols)
        scores = np.full(len(symbols), np.nan, dtype=np.float64)
        index = {symbol: position for position, symbol in enumerate(symbols)}
        for symbol, score in raw.items():
            scores[index[symbol]] = score
        standardized = zscore(scores)
        covered = np.isfinite(standardized)
        numerator[covered] += standardized[covered] * weight
        denominator[covered] += weight

    available = denominator > 0
    combined[available] = numerator[available] / denominator[available]
    order = sorted(
        (index for index, score in enumerate(combined) if np.isfinite(score)),
        key=lambda index: (-combined[index], symbols[index]),
    )
    if not order:
        return Decision.unavailable("FACTOR_PORTFOLIO_UNAVAILABLE")

    target_count = max(1, int(np.ceil(len(order) * top_quantile)))
    exit_count = max(1, int(np.ceil(len(order) * (top_quantile + turnover_buffer))))
    retained = [index for index in order[:exit_count] if ctx.weights.get(symbols[index], Decimal(0)) > 0]
    selected = retained[:target_count]
    selected_set = set(selected)
    for index in order:
        if len(selected) == target_count:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)

    per_name = min(
        Decimal(1) / Decimal(len(selected)),
        Decimal(str(max_weight_per_symbol)),
    ).quantize(_WEIGHT_QUANTUM, rounding=ROUND_DOWN)
    if per_name <= 0:
        raise ValueError("FACTOR_WEIGHT_CAP_TOO_SMALL")
    return Decision.target({symbols[index]: per_name for index in selected})

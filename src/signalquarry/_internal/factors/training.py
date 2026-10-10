# SPDX-License-Identifier: Apache-2.0
"""Pure, prior-only inputs for exploratory formula search; no ledger authority."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, time
from types import MappingProxyType

import numpy as np

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.data.dataset import FIELDS, MICRO, Dataset, truncated
from signalquarry._internal.data.panel import LoadedPanel
from signalquarry._internal.factors.evaluate import ScorePanel, UniverseAt, score_factor
from signalquarry._internal.factors.labels import ForwardReturnLabels, derive_forward_return_labels
from signalquarry._internal.factors.search import FormulaTrainingInput
from signalquarry.sdk.factors import FactorCtx, FactorDef
from signalquarry.sdk.strategy import Params


def _immutable(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.tobytes(order="C"), dtype=values.dtype).reshape(values.shape)


def formula_training_input(
    dataset: Dataset,
    memberships: Sequence[UniverseAt],
    *,
    training_cutoff: date,
    horizon: int,
) -> FormulaTrainingInput:
    """Copy strictly before cutoff and shift decision labels onto completed-bar rows.

    The API must verify manifests and the sealed boundary first. Even label
    calculation receives no cutoff-day/future bars or future ex-date actions.
    Eligibility uses the latest verified build at or before each bar session.
    """
    stop = bisect_left(dataset.sessions, training_cutoff)
    if stop < 2 or not memberships:
        raise ValueError("FACTOR_SEARCH_INPUT_UNVERIFIED")
    training = truncated(dataset, stop)
    training.splits = tuple(action for action in training.splits if action.ex_date < training_cutoff)
    training.dividends = tuple(action for action in training.dividends if action.ex_date < training_cutoff)
    symbols = tuple(sorted({symbol for item in memberships for symbol in item.symbols}))
    if not symbols or not set(symbols) <= training.series.keys():
        raise ValueError("FACTOR_SEARCH_INPUT_UNVERIFIED")
    build_dates = tuple(item.session for item in memberships)
    if build_dates != tuple(sorted(set(build_dates))):
        raise ValueError("FACTOR_SEARCH_INPUT_UNVERIFIED")
    shape = (stop, len(symbols))
    eligible = np.zeros(shape, dtype=np.bool_)
    for row, session in enumerate(training.sessions):
        index = bisect_right(build_dates, session) - 1
        if index < 0:
            continue
        item = memberships[index]
        if item.decision_cutoff.utcoffset() is None or item.decision_cutoff.date() >= item.session:
            raise ValueError("FACTOR_UNIVERSE_TIMING_INVALID")
        eligible[row] = [symbol in item.symbols for symbol in symbols]
    if not eligible.any():
        raise ValueError("FACTOR_SEARCH_INPUT_UNVERIFIED")
    identity = training.identity()
    universe_identity = canonical_hash(
        {
            "schema": "signalquarry.factor-search-universe/v1",
            "builds": [item.identity for item in memberships],
        }
    )
    raw_labels = derive_forward_return_labels(
        training, decision_sessions=training.sessions[1:], symbols=symbols, horizons=(horizon,)
    )
    outcomes = np.full(shape, np.nan)
    outcomes[:-1] = raw_labels.forward_returns[horizon]
    adv = np.full(shape, np.nan)
    adv[:-1] = raw_labels.predecision_adv
    labels = ForwardReturnLabels(
        dataset_identity=identity,
        label_identity=canonical_hash(
            {"schema": "signalquarry.completed-bar-forward-labels/v1", "labels": raw_labels.label_identity}
        ),
        sessions=training.sessions,
        symbols=symbols,
        forward_returns=MappingProxyType({horizon: _immutable(outcomes)}),
        outcome_end_sessions=MappingProxyType({horizon: (*raw_labels.outcome_end_sessions[horizon], None)}),
        predecision_adv=_immutable(adv),
    )
    panels = {
        name: np.column_stack([training.series[symbol].micro[name] for symbol in symbols]) / MICRO
        for name in FIELDS
    }
    panels["volume"] = np.column_stack([training.series[symbol].volume for symbol in symbols])
    present = np.column_stack([training.series[symbol].present for symbol in symbols])
    for values in panels.values():
        values[~present] = np.nan
    return FormulaTrainingInput(
        identity,
        universe_identity,
        FactorCtx(
            training_cutoff,
            training.sessions,
            symbols,
            MappingProxyType({name: _immutable(values) for name, values in panels.items()}),
        ),
        _immutable(eligible),
        labels,
    )


def accepted_training_scores(data: FormulaTrainingInput, definition: FactorDef, params: Params) -> ScorePanel:
    """Evaluate registered comparison code using only the copied training panel."""
    context = data.context
    panel = LoadedPanel(
        data.dataset_identity,
        context.sessions,
        context.universe,
        MappingProxyType({name: _immutable(context.panel(name) * MICRO) for name in FIELDS}),
        context.panel("volume"),
        np.isfinite(context.panel("close")),
    )
    # score_factor supplies completed bars strictly before each next decision.
    memberships = tuple(
        UniverseAt(
            session=decision,
            observed_at=datetime.combine(context.sessions[row], time.min, UTC),
            decision_cutoff=datetime.combine(context.sessions[row], time.min, UTC),
            symbols=tuple(
                symbol for column, symbol in enumerate(context.universe) if data.eligible[row, column]
            ),
            identity=data.universe_identity,
        )
        for row, decision in enumerate(context.sessions[1:])
        if data.eligible[row].any()
    )
    scored = score_factor(definition, params, panel, memberships)
    values = np.full(data.eligible.shape, np.nan)
    rows = {session: row for row, session in enumerate(context.sessions[1:])}
    for row, decision in enumerate(scored.sessions):
        values[rows[decision]] = scored.scores[row]
    return replace(
        scored,
        universe_identity=data.universe_identity,
        sessions=context.sessions,
        scores=_immutable(values),
        eligible=data.eligible,
    )

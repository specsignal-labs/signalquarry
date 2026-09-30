# SPDX-License-Identifier: Apache-2.0
"""Derive close-to-close factor outcomes from raw bars and modeled actions.

This calculation is deliberately separate from source verification. The caller
must still verify that its dataset and corporate-action pages are complete
before using these labels for real-data research or evidence. It also does not
enforce the project's sealed-holdout protocol.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from types import MappingProxyType
from typing import Literal

import numpy as np

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.data.dataset import MICRO, Dataset


def _immutable(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.tobytes(order="C"), dtype=values.dtype).reshape(values.shape)


@dataclass(frozen=True)
class ForwardReturnLabels:
    """Calculated outcomes; this type carries no source-verification authority."""

    dataset_identity: str
    label_identity: str
    sessions: tuple[date, ...]
    symbols: tuple[str, ...]
    forward_returns: Mapping[int, np.ndarray]
    outcome_end_sessions: Mapping[int, tuple[date | None, ...]]
    predecision_adv: np.ndarray
    provenance_verified: Literal[False] = False


def derive_forward_return_labels(
    dataset: Dataset,
    *,
    decision_sessions: Sequence[date],
    symbols: Sequence[str],
    horizons: Sequence[int],
    adv_lookback: int = 20,
) -> ForwardReturnLabels:
    """Calculate total returns from the prior close through each horizon close.

    A decision is made before the first session in its horizon. Horizon 1 exits
    at that session's close. Modeled cash dividends accrue at the ex-date and
    splits change share count on the ex-date. Missing prices leave that symbol's
    outcome unavailable; unsupported lifecycle events are not inferred here.

    This pure helper does not verify provider manifests, instrument continuity,
    the completeness of corporate-action history, or eligibility for grading.
    It does not enforce sealed-holdout access; callers must pass only authorized
    decision sessions.
    """
    sessions = tuple(decision_sessions)
    selected = tuple(symbols)
    requested_horizons = tuple(horizons)
    if (
        not sessions
        or any(type(item) is not date for item in sessions)
        or sessions != tuple(sorted(set(sessions)))
        or not set(sessions) <= set(dataset.sessions)
        or not selected
        or any(not isinstance(item, str) or not item for item in selected)
        or len(selected) != len(set(selected))
        or not set(selected) <= dataset.series.keys()
        or type(adv_lookback) is not int
        or adv_lookback < 1
    ):
        raise ValueError("FACTOR_LABEL_ALIGNMENT_INVALID")
    if (
        not requested_horizons
        or any(type(item) is not int or item < 1 or item > 252 for item in requested_horizons)
        or requested_horizons != tuple(sorted(set(requested_horizons)))
    ):
        raise ValueError("FACTOR_LABEL_HORIZONS_INVALID")

    session_indices = {session: index for index, session in enumerate(dataset.sessions)}
    first_decision = session_indices[sessions[0]]
    final_end = min(
        len(dataset.sessions) - 1,
        max(session_indices[session] + horizon - 1 for session in sessions for horizon in requested_horizons),
    )
    action_start = dataset.sessions[max(0, first_decision - 1)]
    action_end = dataset.sessions[final_end]

    # Dividend terms are per share held before the ex-date. On a date with both
    # events, accrue the dividend first and then apply the split ratio.
    actions: dict[str, dict[date, list[tuple[str, Decimal]]]] = defaultdict(lambda: defaultdict(list))
    seen_actions: set[tuple[str, date, str]] = set()
    for split in dataset.splits:
        if split.symbol not in selected:
            continue
        if type(split.ex_date) is not date:
            raise ValueError("FACTOR_LABEL_ACTIONS_INVALID")
        if not action_start < split.ex_date <= action_end:
            continue
        key = (split.symbol, split.ex_date, "split")
        if (
            key in seen_actions
            or not isinstance(split.ratio, Decimal)
            or not split.ratio.is_finite()
            or split.ratio <= 0
        ):
            raise ValueError("FACTOR_LABEL_ACTIONS_INVALID")
        seen_actions.add(key)
        actions[split.symbol][split.ex_date].append(("split", split.ratio))
    for dividend in dataset.dividends:
        if dividend.symbol not in selected:
            continue
        if type(dividend.ex_date) is not date or type(dividend.pay_date) is not date:
            raise ValueError("FACTOR_LABEL_ACTIONS_INVALID")
        if not action_start < dividend.ex_date <= action_end:
            continue
        key = (dividend.symbol, dividend.ex_date, "dividend")
        if (
            key in seen_actions
            or not isinstance(dividend.amount, Decimal)
            or not dividend.amount.is_finite()
            or dividend.amount < 0
            or dividend.pay_date < dividend.ex_date
        ):
            raise ValueError("FACTOR_LABEL_ACTIONS_INVALID")
        seen_actions.add(key)
        actions[dividend.symbol][dividend.ex_date].append(("dividend", dividend.amount))
    for by_date in actions.values():
        for items in by_date.values():
            items.sort(key=lambda item: item[0])  # dividend sorts before split

    dataset_identity = dataset.identity()
    shape = (len(sessions), len(selected))
    adv = np.full(shape, np.nan, dtype=np.float64)
    result_arrays = {horizon: np.full(shape, np.nan, dtype=np.float64) for horizon in requested_horizons}
    end_sessions: dict[int, tuple[date | None, ...]] = {}

    for row, decision in enumerate(sessions):
        decision_index = session_indices[decision]
        for column, symbol in enumerate(selected):
            series = dataset.series[symbol]

            adv_start = decision_index - adv_lookback
            if adv_start >= 0:
                prior_present = series.present[adv_start:decision_index]
                prior_close = series.micro["close"][adv_start:decision_index]
                prior_volume = series.volume[adv_start:decision_index]
                if (
                    len(prior_present) == adv_lookback
                    and bool(prior_present.all())
                    and bool((prior_close > 0).all())
                    and bool(np.isfinite(prior_volume).all())
                    and bool((prior_volume >= 0).all())
                ):
                    with np.errstate(over="ignore", invalid="ignore"):
                        dollar_volume = prior_close.astype(np.float64) / MICRO * prior_volume
                        value = float(dollar_volume.mean())
                    if math.isfinite(value):
                        adv[row, column] = value

            for horizon in requested_horizons:
                end_index = decision_index + horizon - 1
                if end_index >= len(dataset.sessions) or decision_index == 0:
                    continue
                entry_index = decision_index - 1
                present = series.present[entry_index : end_index + 1]
                closes = series.micro["close"][entry_index : end_index + 1]
                if len(present) != horizon + 1 or not bool(present.all()) or not bool((closes > 0).all()):
                    continue

                shares = Decimal(1)
                cash = Decimal(0)
                entry_session = dataset.sessions[entry_index]
                exit_session = dataset.sessions[end_index]
                for ex_date in sorted(actions.get(symbol, {})):
                    events = actions[symbol][ex_date]
                    if not entry_session < ex_date <= exit_session:
                        continue
                    for kind, amount in events:
                        if kind == "dividend":
                            cash += shares * amount
                        else:
                            shares *= amount
                entry_price = Decimal(int(closes[0])) / Decimal(MICRO)
                exit_price = Decimal(int(closes[-1])) / Decimal(MICRO)
                total_return = (shares * exit_price + cash) / entry_price - Decimal(1)
                value = float(total_return)
                # Valid positive prices/shares and nonnegative cash flows keep
                # total return at or above -100%; only overflow can fail here.
                if not math.isfinite(value):
                    raise ValueError("FACTOR_LABEL_HORIZONS_INVALID")
                result_arrays[horizon][row, column] = value

    # Build the end-date vectors separately to avoid conflating right-censoring
    # with symbol-specific missing observations.
    for horizon in requested_horizons:
        end_sessions[horizon] = tuple(
            dataset.sessions[index + horizon - 1] if index + horizon - 1 < len(dataset.sessions) else None
            for index in (session_indices[session] for session in sessions)
        )

    identity = canonical_hash(
        {
            "schema": "signalquarry.factor-forward-return-labels/v1",
            "dataset_identity": dataset_identity,
            "decision_sessions": sessions,
            "symbols": selected,
            "horizons": requested_horizons,
            "adv_lookback": adv_lookback,
            "method": {
                "entry": "prior_completed_close",
                "exit": "hth_session_close_including_decision_session",
                "actions": "cash_dividend_at_ex_date_and_split_share_ratio",
                "missing_bars": "outcome_unavailable",
            },
        }
    )
    return ForwardReturnLabels(
        dataset_identity=dataset_identity,
        label_identity=identity,
        sessions=sessions,
        symbols=selected,
        forward_returns=MappingProxyType(
            {horizon: _immutable(values) for horizon, values in result_arrays.items()}
        ),
        outcome_end_sessions=MappingProxyType(end_sessions),
        predecision_adv=_immutable(adv),
    )

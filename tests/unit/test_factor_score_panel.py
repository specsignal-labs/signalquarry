# SPDX-License-Identifier: Apache-2.0
"""What binds a score panel to its dated universes, and what timing and membership it refuses."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from signalquarry._internal.factors.evaluate import UniverseAt, score_factor
from signalquarry.sdk import factor_definition_of

from .test_factor_evaluation import P, _membership, _panel, previous_close


def _score(membership: tuple[UniverseAt, ...]) -> object:
    return score_factor(factor_definition_of(previous_close), P(), _panel(), membership)


def _first() -> tuple[UniverseAt, ...]:
    return _membership(_panel())[:2]


def test_the_universe_identity_binds_every_dated_membership_field() -> None:
    base = _score(_first())
    identity = base.universe_identity  # type: ignore[attr-defined]
    assert _score(_first()).universe_identity == identity  # type: ignore[attr-defined]
    first, second = _first()
    variants = [
        (replace(first, observed_at=first.observed_at - timedelta(hours=1)), second),
        (replace(first, decision_cutoff=first.decision_cutoff + timedelta(hours=1)), second),
        (replace(first, symbols=first.symbols[:-1]), second),
        (replace(first, identity="sha256:" + "9" * 64), second),
        (first, replace(second, identity="sha256:" + "9" * 64)),
        (first,),
    ]
    seen = {identity}
    for membership in variants:
        seen.add(_score(membership).universe_identity)  # type: ignore[attr-defined]
    assert len(seen) == len(variants) + 1


def test_only_the_named_symbols_are_eligible_and_scored_in_each_row() -> None:
    panel = _panel()
    first = _membership(panel)[0]
    subset = replace(first, symbols=("S03", "S07"))
    result = score_factor(factor_definition_of(previous_close), P(), panel, (subset,))
    assert result.eligible[0].tolist() == [index in (3, 7) for index in range(10)]
    assert result.scores[0, 3] == 13.0 and result.scores[0, 7] == 17.0
    assert np.isnan(result.scores[0, [0, 1, 2, 4, 5, 6, 8, 9]]).all()
    assert result.symbols == panel.symbols
    assert result.sessions == (first.session,)
    assert result.dataset_identity == panel.dataset_identity


@pytest.mark.parametrize("identity", ["sha256:xyz", "", "sha256:" + "A" * 64, "sha256:" + "1" * 63])
def test_a_malformed_dataset_identity_is_refused(identity: str) -> None:
    panel = replace(_panel(), dataset_identity=identity)
    with pytest.raises(ValueError, match="^FACTOR_EVALUATION_IDENTITY_INVALID$"):
        score_factor(factor_definition_of(previous_close), P(), panel, _first())


def test_an_empty_membership_is_refused() -> None:
    with pytest.raises(ValueError, match="^FACTOR_EVALUATION_IDENTITY_INVALID$"):
        score_factor(factor_definition_of(previous_close), P(), _panel(), ())


def test_sessions_must_be_sorted_unique_and_inside_the_panel() -> None:
    panel = _panel()
    first, second = _first()
    for membership in (
        (second, first),
        (first, first),
        (replace(first, session=panel.sessions[-1] + timedelta(days=1)),),
    ):
        with pytest.raises(ValueError, match="^FACTOR_EVALUATION_SESSIONS_INVALID$"):
            score_factor(factor_definition_of(previous_close), P(), panel, membership)
    last = replace(
        first,
        session=panel.sessions[-1],
        observed_at=first.observed_at,
        decision_cutoff=first.decision_cutoff,
    )
    assert score_factor(factor_definition_of(previous_close), P(), panel, (last,)).sessions == (
        panel.sessions[-1],
    )


def test_timing_must_be_aware_ordered_and_before_the_session_date() -> None:
    panel = _panel()
    first = _membership(panel)[0]
    naive = datetime(2024, 1, 2)
    zone = timezone(timedelta(hours=-5))
    cases = [
        replace(first, observed_at=naive),
        replace(first, decision_cutoff=naive),
        replace(first, observed_at=first.decision_cutoff + timedelta(seconds=1)),
        replace(first, decision_cutoff=datetime.combine(first.session, datetime.min.time(), UTC)),
        replace(first, identity="sha256:xyz"),
    ]
    for item in cases:
        with pytest.raises(ValueError, match="^FACTOR_UNIVERSE_TIMING_INVALID$"):
            score_factor(factor_definition_of(previous_close), P(), panel, (item,))
    # observed exactly at the cutoff is allowed; the cutoff is judged by its UTC date
    equal = replace(first, observed_at=first.decision_cutoff)
    score_factor(factor_definition_of(previous_close), P(), panel, (equal,))
    late_utc_evening = datetime.combine(
        first.session - timedelta(days=1), datetime.min.time(), zone
    ) + timedelta(hours=18)
    # 18:00 at UTC-5 on the previous day is 23:00 UTC the same day: still before the session
    score_factor(
        factor_definition_of(previous_close),
        P(),
        panel,
        (replace(first, observed_at=late_utc_evening, decision_cutoff=late_utc_evening),),
    )
    next_utc_day = late_utc_evening + timedelta(hours=2)  # 01:00 UTC on the session date
    with pytest.raises(ValueError, match="^FACTOR_UNIVERSE_TIMING_INVALID$"):
        score_factor(
            factor_definition_of(previous_close),
            P(),
            panel,
            (replace(first, observed_at=late_utc_evening, decision_cutoff=next_utc_day),),
        )


def test_universe_members_must_exist_be_unique_and_not_empty() -> None:
    panel = _panel()
    first = _membership(panel)[0]
    for symbols in ((), ("S00", "S00"), ("S00", "NOPE")):
        with pytest.raises(ValueError, match="^FACTOR_UNIVERSE_INVALID$"):
            score_factor(factor_definition_of(previous_close), P(), panel, (replace(first, symbols=symbols),))

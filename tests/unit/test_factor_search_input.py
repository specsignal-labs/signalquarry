# SPDX-License-Identifier: Apache-2.0
"""Every way a formula-search training input can be malformed, and what is masked, not read."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import date, timedelta
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from signalquarry._internal.factors.search import FormulaTrainingInput, _identity, _validate_input
from signalquarry.sdk.factors import FactorCtx

from .test_factor_search import _config, _training_input

_INVALID = "^FACTOR_SEARCH_INPUT_INVALID$"


def _data() -> FormulaTrainingInput:
    return _training_input(train_rows=40, future_rows=6, symbols_count=5)


def _cfg(data: FormulaTrainingInput) -> Any:
    return _config(training_cutoff=data.context.sessions[-1], budget=4, family_budget=10, population_size=4)


def _with_context(data: FormulaTrainingInput, **updates: object) -> FormulaTrainingInput:
    return replace(data, context=replace(data.context, **updates))


def _with_labels(data: FormulaTrainingInput, **updates: object) -> FormulaTrainingInput:
    return replace(data, labels=replace(data.labels, **updates))


def _panels_with(data: FormulaTrainingInput, field: str, values: np.ndarray) -> FormulaTrainingInput:
    panels = dict(data.context._panels)
    panels[field] = values
    return _with_context(data, _panels=MappingProxyType(panels))


def _returns(data: FormulaTrainingInput, change: Callable[[np.ndarray], None]) -> FormulaTrainingInput:
    values = np.array(data.labels.forward_returns[1], dtype=np.float64, copy=True)
    change(values)
    return _with_labels(data, forward_returns={1: values})


def _ends(data: FormulaTrainingInput, change: Callable[[list[date | None]], None]) -> FormulaTrainingInput:
    ends = list(data.labels.outcome_end_sessions[1])  # type: ignore[index]
    change(ends)
    return _with_labels(data, outcome_end_sessions={1: tuple(ends)})


def _sessions(data: FormulaTrainingInput) -> tuple[date, ...]:
    return data.context.sessions


def _bad_inputs() -> dict[str, Callable[[FormulaTrainingInput], Any]]:
    other = "sha256:" + "9" * 64
    return {
        "not-an-input": lambda d: object(),
        "dataset-identity-text": lambda d: replace(d, dataset_identity="sha256:xyz"),
        "dataset-identity-uppercase": lambda d: replace(d, dataset_identity="sha256:" + "A" * 64),
        "dataset-identity-short": lambda d: replace(d, dataset_identity="sha256:" + "1" * 63),
        "dataset-identity-no-prefix": lambda d: replace(d, dataset_identity="1" * 64),
        "universe-identity-bad": lambda d: replace(d, universe_identity="nope"),
        "context-not-a-context": lambda d: replace(d, context=object()),
        "labels-not-labels": lambda d: replace(d, labels=object()),
        "label-dataset-identity-bad": lambda d: _with_labels(d, dataset_identity="bad"),
        "label-identity-bad": lambda d: _with_labels(d, label_identity="bad"),
        "label-dataset-differs": lambda d: _with_labels(d, dataset_identity=other),
        "decision-session-datetime": lambda d: _with_context(d, decision_session=None),
        "sessions-list": lambda d: _with_context(d, sessions=list(_sessions(d))),
        "sessions-empty": lambda d: _with_context(d, sessions=()),
        "sessions-duplicate": lambda d: _with_context(d, sessions=(*_sessions(d)[:-1], _sessions(d)[-2])),
        "sessions-unsorted": lambda d: _with_context(d, sessions=tuple(reversed(_sessions(d)))),
        "sessions-not-dates": lambda d: _with_context(d, sessions=(*_sessions(d)[:-1], "2024-02-11")),
        "session-on-decision-day": lambda d: _with_context(d, decision_session=_sessions(d)[-1]),
        "cutoff-differs": lambda d: _with_context(d, sessions=_sessions(d)[:-1]),
        "no-symbols": lambda d: _with_context(d, universe=()),
        "empty-symbol": lambda d: _with_context(d, universe=("", *d.context.universe[1:])),
        "non-string-symbol": lambda d: _with_context(d, universe=(7, *d.context.universe[1:])),
        "duplicate-symbols": lambda d: _with_context(
            d, universe=(d.context.universe[0], *d.context.universe[:-1])
        ),
        "label-sessions-diverge": lambda d: _with_labels(
            d, sessions=(date(2000, 1, 1), *d.labels.sessions[1:])
        ),
        "label-symbols-diverge": lambda d: _with_labels(d, symbols=tuple(reversed(d.labels.symbols))),
        "horizon-absent": lambda d: _with_labels(d, forward_returns={2: d.labels.forward_returns[1]}),
        "outcome-ends-missing": lambda d: _with_labels(d, outcome_end_sessions=None),
        "outcome-ends-other-horizon": lambda d: _with_labels(d, outcome_end_sessions={2: ()}),
        "eligible-not-bool": lambda d: replace(d, eligible=d.eligible.astype(np.int8)),
        "eligible-wrong-shape": lambda d: replace(d, eligible=d.eligible[:, :-1]),
        "panel-wrong-shape": lambda d: _panels_with(d, "close", d.context.panel("close")[:-1]),
        "panel-infinite": lambda d: _panels_with(
            d, "volume", np.where(np.eye(*d.eligible.shape, dtype=bool), np.inf, 1.0)
        ),
        "panel-missing": lambda d: _with_context(
            d, _panels=MappingProxyType({k: v for k, v in d.context._panels.items() if k != "high"})
        ),
        "panel-not-numeric": lambda d: _panels_with(d, "open", np.full(d.eligible.shape, "x", dtype=object)),
        "label-sessions-list": lambda d: _with_labels(d, sessions=list(d.labels.sessions)),
        "label-sessions-not-dates": lambda d: _with_labels(
            d, sessions=(*d.labels.sessions[:-1], "2024-03-01")
        ),
        "label-sessions-short": lambda d: _with_labels(
            d, sessions=d.labels.sessions[: len(_sessions(d)) - 1]
        ),
        "outcomes-wrong-shape": lambda d: _with_labels(
            d, forward_returns={1: d.labels.forward_returns[1][:-1]}
        ),
        "outcome-ends-list": lambda d: _with_labels(
            d, outcome_end_sessions={1: list(d.labels.outcome_end_sessions[1])}
        ),  # type: ignore[index]
        "outcome-ends-short": lambda d: _with_labels(
            d, outcome_end_sessions={1: d.labels.outcome_end_sessions[1][:-1]}
        ),  # type: ignore[index]
        "outcome-infinite": lambda d: _returns(d, lambda v: v.__setitem__((3, 1), np.inf)),
        "outcome-below-total-loss": lambda d: _returns(d, lambda v: v.__setitem__((3, 1), -1.0001)),
        "outcome-end-not-a-date": lambda d: _ends(d, lambda e: e.__setitem__(3, "2024-02-01")),
        "outcome-ends-on-decision-day": lambda d: _ends(d, lambda e: e.__setitem__(3, _sessions(d)[3])),
        "outcome-ends-before-decision-day": lambda d: _ends(d, lambda e: e.__setitem__(3, _sessions(d)[2])),
        "unended-outcome-has-values": lambda d: _ends(d, lambda e: e.__setitem__(3, None)),
        "adv-wrong-shape": lambda d: _with_labels(d, predecision_adv=np.ones((3, 2))),
    }


@pytest.mark.parametrize("name", list(_bad_inputs()))
def test_malformed_training_input_is_rejected_with_the_stable_code(name: str) -> None:
    data = _data()
    broken = _bad_inputs()[name](data)
    with pytest.raises(ValueError, match=_INVALID):
        _validate_input(broken, _cfg(data))


def test_the_baseline_input_validates() -> None:
    data = _data()
    _validate_input(data, _cfg(data))


def test_a_total_loss_outcome_is_allowed() -> None:
    data = _data()
    allowed = _returns(data, lambda v: v.__setitem__((3, 1), -1.0))
    _validate_input(allowed, _cfg(data))


def test_a_missing_forward_return_with_an_unended_outcome_is_accepted() -> None:
    data = _data()
    ended = _ends(data, lambda e: e.__setitem__(3, None))
    cleared = _returns(ended, lambda v: v.__setitem__(3, np.nan))
    validated = _validate_input(cleared, _cfg(data))
    assert np.isnan(validated.labels.forward_returns[1][3]).all()


def test_outcomes_ending_after_the_cutoff_are_masked_but_those_ending_on_it_are_kept() -> None:
    data = _data()
    sessions = _sessions(data)
    rows = len(sessions)
    cutoff = sessions[-1]
    late = cutoff + timedelta(days=3)

    def reroute(ends: list[date | None]) -> None:
        ends[rows - 3] = late
        ends[rows - 2] = cutoff
        ends[rows - 1] = None

    marked = _returns(_ends(data, reroute), lambda v: v.__setitem__(slice(None), 0.01))
    marked = _returns(marked, lambda v: v.__setitem__(rows - 1, np.nan))
    validated = _validate_input(marked, _cfg(data))
    outcomes = validated.labels.forward_returns[1]
    assert outcomes.shape[0] == rows
    assert np.isnan(outcomes[rows - 3]).all()
    assert (outcomes[rows - 2] == 0.01).all()
    assert (outcomes[0] == 0.01).all()
    assert validated.labels.outcome_end_sessions is not None
    assert validated.labels.outcome_end_sessions[1][rows - 3] == late
    assert validated.labels.outcome_end_sessions[1][rows - 2] == cutoff
    assert validated.labels.outcome_end_sessions[1][rows - 1] is None


def test_validated_training_data_is_a_frozen_copy_trimmed_to_the_training_window() -> None:
    data = _data()
    config = _cfg(data)
    validated = _validate_input(data, config)
    rows = len(_sessions(data))
    assert validated.context.sessions == _sessions(data)
    assert validated.context.universe == data.context.universe
    assert validated.context.decision_session == data.context.decision_session
    assert validated.labels.sessions == _sessions(data)
    assert validated.labels.forward_returns[1].shape == (rows, len(data.context.universe))
    assert validated.dataset_identity == data.dataset_identity
    assert validated.universe_identity == data.universe_identity
    assert validated.labels.dataset_identity == data.dataset_identity
    assert validated.labels.label_identity != data.labels.label_identity
    for field in ("open", "high", "low", "close", "volume"):
        panel = validated.context.panel(field)
        assert not panel.flags.writeable
        assert np.array_equal(panel, data.context.panel(field), equal_nan=True)
    assert not validated.eligible.flags.writeable
    assert np.array_equal(validated.eligible, data.eligible)
    assert not np.shares_memory(validated.eligible, data.eligible)
    assert validated.labels.predecision_adv is None


def test_predecision_adv_is_trimmed_and_frozen() -> None:
    data = _data()
    total = len(data.labels.sessions)
    adv = np.arange(total * len(data.context.universe), dtype=np.float64).reshape(total, -1)
    validated = _validate_input(_with_labels(data, predecision_adv=adv), _cfg(data))
    assert validated.labels.predecision_adv is not None
    assert not validated.labels.predecision_adv.flags.writeable
    assert np.array_equal(validated.labels.predecision_adv, adv[: len(_sessions(data))])


def test_the_training_label_identity_binds_its_content() -> None:
    data = _data()
    config = _cfg(data)
    base = _validate_input(data, config).labels.label_identity
    assert base.startswith("sha256:")
    assert _validate_input(data, config).labels.label_identity == base
    assert (
        _validate_input(replace(data, universe_identity="sha256:" + "8" * 64), config).labels.label_identity
        != base
    )
    nudged = _returns(data, lambda v: v.__setitem__((0, 0), 0.123))
    assert _validate_input(nudged, config).labels.label_identity != base
    # held-out rows past the cutoff do not influence it
    future = _returns(data, lambda v: v.__setitem__(slice(len(_sessions(data)), None), 0.5))
    assert _validate_input(future, config).labels.label_identity == base


def test_identity_accepts_only_lowercase_sha256_hex_of_the_exact_length() -> None:
    assert _identity("sha256:" + "0123456789abcdef" * 4)
    assert not _identity("sha256:" + "0123456789abcdeF" * 4)
    assert not _identity("sha256:" + "g" * 64)
    assert not _identity("sha256:" + "a" * 65)
    assert not _identity("sha256:" + "a" * 63)
    assert not _identity("sha1:" + "a" * 64)
    assert not _identity(None)  # type: ignore[arg-type]
    assert not _identity(b"sha256:" + b"a" * 64)  # type: ignore[arg-type]


def test_context_type_is_the_factor_context() -> None:
    assert isinstance(_data().context, FactorCtx)


def _library_panel(data: FormulaTrainingInput, **updates: object) -> Any:
    from signalquarry._internal.factors.evaluate import ScorePanel

    shape = data.eligible.shape
    panel = ScorePanel(
        data.dataset_identity,
        data.universe_identity,
        data.context.sessions,
        data.context.universe,
        np.zeros(shape),
        np.ones(shape, dtype=np.bool_),
    )
    return replace(panel, **updates)


def _library_cases(data: FormulaTrainingInput) -> dict[str, tuple[str, Any]]:
    shape = data.eligible.shape
    return {
        "empty-name": ("", _library_panel(data)),
        "not-a-score-panel": ("other", object()),
        "dataset-differs": ("other", _library_panel(data, dataset_identity="sha256:" + "7" * 64)),
        "universe-differs": ("other", _library_panel(data, universe_identity="sha256:" + "7" * 64)),
        "sessions-differ": ("other", _library_panel(data, sessions=data.context.sessions[:-1])),
        "symbols-differ": ("other", _library_panel(data, symbols=data.context.universe[:-1])),
        "scores-shape": ("other", _library_panel(data, scores=np.zeros((shape[0] - 1, shape[1])))),
        "eligible-shape": (
            "other",
            _library_panel(data, eligible=np.ones((shape[0], shape[1] - 1), dtype=np.bool_)),
        ),
    }


@pytest.mark.parametrize(
    "name",
    [
        "empty-name",
        "not-a-score-panel",
        "dataset-differs",
        "universe-differs",
        "sessions-differ",
        "symbols-differ",
        "scores-shape",
        "eligible-shape",
    ],
)
def test_an_accepted_library_must_match_the_training_panel_exactly(name: str) -> None:
    from signalquarry._internal.factors.search import _validate_library

    data = _data()
    validated = _validate_input(data, _cfg(data))
    key, panel = _library_cases(data)[name]
    with pytest.raises(ValueError, match=_INVALID):
        _validate_library(validated, {key: panel})


def test_a_matching_library_is_accepted() -> None:
    from signalquarry._internal.factors.search import _validate_library

    data = _data()
    validated = _validate_input(data, _cfg(data))
    _validate_library(validated, {})
    _validate_library(validated, {"planted": _library_panel(data)})


def test_the_training_label_identity_schema_is_pinned() -> None:
    # A change to the hashed fields or schema string changes every stored training-label identity.
    data = _data()
    identity = _validate_input(data, _cfg(data)).labels.label_identity
    assert identity == "sha256:5160c8409045e34ff8c7b445cbe1db912de8cc90b8cef3d4d73071f571d3e2ef"

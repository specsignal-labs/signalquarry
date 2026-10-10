# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.data.synthetic import synthetic_panel
from signalquarry._internal.factors.evaluate import UniverseAt
from signalquarry._internal.factors.search import FormulaSearchConfig, search_expressions
from signalquarry._internal.factors.training import accepted_training_scores, formula_training_input
from signalquarry._internal.validation import ledger
from signalquarry._internal.validation.factor_search import search_trial_accounting, verify_search_trials
from signalquarry.sdk import FactorCtx, Params, factor


def _panel(seed=7, planted=0.5):
    return synthetic_panel(
        date(2023, 1, 2),
        date(2023, 6, 30),
        n_symbols=16,
        seed=seed,
        planted_ic=planted,
        late_listing_fraction=0,
        delisting_fraction=0,
        missing_rate=0,
        split_fraction=0,
        dividend_fraction=0,
    ).dataset


def _members(dataset):
    return tuple(
        UniverseAt(
            dataset.sessions[index],
            datetime.combine(dataset.sessions[index] - timedelta(days=1), datetime.min.time(), UTC),
            datetime.combine(dataset.sessions[index] - timedelta(days=1), datetime.min.time(), UTC),
            symbols,
            canonical_hash({"index": index, "symbols": symbols}),
        )
        for index, symbols in ((20, dataset.symbols[:8]), (40, dataset.symbols))
    )


def test_training_cutoff_perturbation_changes_no_score_or_trial_identity(monkeypatch):
    dataset = _panel()
    members = _members(dataset)
    cutoff = dataset.sessions[-10]
    data = formula_training_input(dataset, members, training_cutoff=cutoff, horizon=3)
    config = FormulaSearchConfig(cutoff, 3, 17, 7)
    first = search_expressions(data, config)
    for series in dataset.series.values():
        for values in (*series.micro.values(), series.volume, series.present):
            values[-10:] = 0
    second_data = formula_training_input(dataset, members, training_cutoff=cutoff, horizon=3)
    second = search_expressions(second_data, config)
    assert first == second
    assert data.dataset_identity == second_data.dataset_identity
    assert data.labels.label_identity == second_data.labels.label_identity
    assert data.context.sessions[-1] < cutoff
    assert not data.eligible[:20].any()
    assert data.eligible[20:40, :8].all() and not data.eligible[20:40, 8:].any()
    assert data.eligible[40:].all()
    assert np.isnan(data.labels.forward_returns[3][-3:]).all()
    # The copy is isolated even if the original pre-cutoff source is changed later.
    before = data.context.panel("close").copy()
    dataset.series[dataset.symbols[0]].micro["close"][:] = 0
    np.testing.assert_array_equal(before, data.context.panel("close"))
    with pytest.raises(ValueError):
        data.context.panel("close").flags.writeable = True


@pytest.mark.parametrize("damage", ["short", "empty", "symbols", "dates", "late", "timing"])
def test_training_invalid_inputs_are_refused(damage):
    dataset = _panel()
    members = _members(dataset)
    cutoff = dataset.sessions[-10]
    expected = "FACTOR_SEARCH_INPUT_UNVERIFIED"
    if damage == "short":
        cutoff = dataset.sessions[1]
    elif damage == "empty":
        members = ()
    elif damage == "symbols":
        members = (replace(members[0], symbols=("missing",)),)
    elif damage == "dates":
        members = (*members, members[0])
    elif damage == "late":
        members = (replace(members[0], session=dataset.sessions[-1]),)
    elif damage == "timing":
        members = (
            replace(
                members[0], decision_cutoff=datetime.combine(members[0].session, datetime.min.time(), UTC)
            ),
        )
        expected = "FACTOR_UNIVERSE_TIMING_INVALID"
    with pytest.raises(ValueError, match=expected):
        formula_training_input(dataset, members, training_cutoff=cutoff, horizon=1)


def test_accepted_factors_receive_only_completed_training_bars():
    dataset = _panel()
    cutoff = dataset.sessions[-10]
    data = formula_training_input(dataset, _members(dataset), training_cutoff=cutoff, horizon=1)
    observed = []

    @factor(params=Params, lookback=lambda p: 1)
    def score(ctx: FactorCtx, p: Params):
        observed.extend(ctx.sessions)
        assert all(session < cutoff for session in ctx.sessions)
        return {symbol: float(ctx.panel("close")[-1, i]) for i, symbol in enumerate(ctx.universe)}

    scores = accepted_training_scores(data, score.__signalquarry_factor__, Params())
    assert scores.sessions == data.context.sessions and scores.universe_identity == data.universe_identity
    assert observed and max(observed) < cutoff
    assert np.isnan(scores.scores[:20]).all()
    assert np.isfinite(scores.scores[40:-1]).all()
    assert np.isnan(scores.scores[-1]).all()


def test_resume_invalid_search_rows_and_duplicate_trials_refuse(tmp_path):
    root = tmp_path
    (root / "signalquarry.toml").write_text("")
    ledger.append(
        root,
        "factor_holdouts",
        "alpha",
        {
            "kind": "seal",
            "family": "alpha",
            "holdout_start": "2023-06-15",
            "trial_budget": 128,
            "holdout": {"months": 0, "training_cutoff": "2023-06-14"},
        },
    )
    ledger.append(
        root,
        "factor_trials",
        "alpha",
        {
            "kind": "factor_trial",
            "family": "alpha",
            "metrics": {"search_identity": "bad"},
        },
    )
    with pytest.raises(ledger.LedgerError, match="FACTOR_SEARCH_INPUT_UNVERIFIED"):
        search_trial_accounting(root, "alpha", "bad")
    row = {"trial_configuration_hash": "x", "metrics": {"p_value": 1}}
    assert verify_search_trials([row], [row]) == {"x"}
    for recorded in (
        [row, row],
        [{**row, "trial_configuration_hash": "y"}],
        [{**row, "metrics": {"p_value": 0}}],
    ):
        with pytest.raises(ledger.LedgerError, match="FACTOR_SEARCH_NONDETERMINISTIC"):
            verify_search_trials(recorded, [row])


@pytest.mark.slow
def test_slow_null_200_panels_use_real_ledger_accounting(tmp_path):
    """Deselect with -m 'not slow'; 64 unique formulas on each of 200 panels."""
    (tmp_path / "signalquarry.toml").write_text("")
    ledger.append(
        tmp_path,
        "factor_holdouts",
        "null",
        {
            "kind": "seal",
            "family": "null",
            "holdout_start": "2023-06-15",
            "trial_budget": 128,
            "holdout": {"months": 0, "training_cutoff": "2023-06-14"},
        },
    )
    start = time.perf_counter()
    discoveries = 0
    for seed in range(200):
        dataset = _panel(seed=seed, planted=0)
        cutoff = dataset.sessions[-20]
        data = formula_training_input(dataset, _members(dataset), training_cutoff=cutoff, horizon=1)
        accounting, recorded = search_trial_accounting(tmp_path, "null", canonical_hash({"seed": seed}))
        assert not recorded
        config = FormulaSearchConfig(
            cutoff,
            1,
            seed,
            64,
            family_budget=accounting.effective_budget,
            family_trials_used=accounting.family_trials_used,
            project_trials_used=accounting.project_trials_used,
            previous_family_p_values=accounting.previous_family_p_values,
        )
        report = search_expressions(data, config)
        assert report.trials_evaluated == 64
        discoveries += bool(report.discoveries)
    elapsed = time.perf_counter() - start
    print(f"null discovery fraction: {discoveries}/200 = {discoveries / 200:.3%}; elapsed: {elapsed:.3f}s")
    # Local pytest output survives -q and is also available for the final report.
    (tmp_path / "null-result.json").write_text(
        json.dumps({"discoveries": discoveries, "panels": 200, "seconds": elapsed})
    )
    assert discoveries / 200 <= 0.10

# SPDX-License-Identifier: Apache-2.0
"""Both sides of every limit in the formula-search configuration contract."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, datetime

import pytest

from signalquarry._internal.factors.search import FormulaSearchConfig, _validate_config

_INVALID = "^FACTOR_SEARCH_CONFIG_INVALID$"
_EXHAUSTED = "^FACTOR_SEARCH_BUDGET_EXHAUSTED$"


def _base(**updates: object) -> FormulaSearchConfig:
    return replace(
        FormulaSearchConfig(
            training_cutoff=date(2024, 1, 2),
            horizon=1,
            seed=17,
            budget=18,
            family_budget=40,
            population_size=8,
        ),
        **updates,
    )


_ACCEPTED: list[tuple[str, dict[str, object]]] = [
    ("horizon-min", {"horizon": 1}),
    ("horizon-max", {"horizon": 252}),
    ("seed-min", {"seed": 0}),
    ("seed-max", {"seed": 2**64 - 1}),
    ("budget-min", {"budget": 1}),
    ("budget-max", {"budget": 10_000, "family_budget": 10_000}),
    ("family-budget-min", {"budget": 1, "family_budget": 1}),
    ("family-budget-max", {"family_budget": 10_000}),
    ("trials-used-leaves-the-batch", {"family_trials_used": 22, "project_trials_used": 22}),
    ("project-trials-equal-family", {"family_trials_used": 5, "project_trials_used": 5}),
    ("project-trials-above-family", {"family_trials_used": 5, "project_trials_used": 500}),
    ("population-min", {"population_size": 2}),
    ("population-max", {"population_size": 128}),
    ("blocks-min", {"chronological_blocks": 1}),
    ("blocks-max", {"chronological_blocks": 100}),
    ("complexity-zero", {"complexity_penalty": 0}),
    ("complexity-max", {"complexity_penalty": 10}),
    ("complexity-float", {"complexity_penalty": 0.5}),
    ("redundancy-zero", {"redundancy_penalty": 0.0}),
    ("redundancy-max", {"redundancy_penalty": 10.0}),
    ("redundancy-int", {"redundancy_penalty": 3}),
    ("alpha-max", {"fdr_alpha": 0.2}),
    ("alpha-tiny", {"fdr_alpha": 1e-9}),
    (
        "previous-p-values",
        {"previous_family_p_values": (0.0, 1, 0.5), "family_trials_used": 3, "project_trials_used": 3},
    ),
    (
        "previous-p-values-fewer-than-trials",
        {"previous_family_p_values": (0.5,), "family_trials_used": 3, "project_trials_used": 3},
    ),
]

_REJECTED: list[tuple[str, dict[str, object]]] = [
    ("horizon-zero", {"horizon": 0}),
    ("horizon-too-long", {"horizon": 253}),
    ("horizon-bool", {"horizon": True}),
    ("horizon-float", {"horizon": 1.0}),
    ("seed-negative", {"seed": -1}),
    ("seed-overflow", {"seed": 2**64}),
    ("seed-float", {"seed": 1.5}),
    ("budget-zero", {"budget": 0}),
    ("budget-above-cap", {"budget": 10_001, "family_budget": 10_000}),
    ("budget-float", {"budget": 18.0}),
    ("family-budget-zero", {"family_budget": 0, "budget": 1}),
    ("family-budget-above-cap", {"family_budget": 10_001}),
    ("trials-used-negative", {"family_trials_used": -1}),
    ("trials-used-above-budget", {"family_trials_used": 41, "project_trials_used": 41}),
    ("project-trials-below-family", {"family_trials_used": 5, "project_trials_used": 4}),
    ("population-one", {"population_size": 1}),
    ("population-above-cap", {"population_size": 129}),
    ("blocks-zero", {"chronological_blocks": 0}),
    ("blocks-above-cap", {"chronological_blocks": 101}),
    ("complexity-negative", {"complexity_penalty": -0.001}),
    ("complexity-above-cap", {"complexity_penalty": 10.001}),
    ("complexity-nan", {"complexity_penalty": math.nan}),
    ("complexity-inf", {"complexity_penalty": math.inf}),
    ("complexity-text", {"complexity_penalty": "0.1"}),
    ("complexity-bool", {"complexity_penalty": True}),
    ("redundancy-negative", {"redundancy_penalty": -0.001}),
    ("redundancy-above-cap", {"redundancy_penalty": 10.001}),
    ("redundancy-nan", {"redundancy_penalty": math.nan}),
    ("redundancy-inf", {"redundancy_penalty": -math.inf}),
    ("redundancy-text", {"redundancy_penalty": "0.1"}),
    ("alpha-zero", {"fdr_alpha": 0.0}),
    ("alpha-negative", {"fdr_alpha": -0.05}),
    ("alpha-above-cap", {"fdr_alpha": 0.2000001}),
    ("alpha-nan", {"fdr_alpha": math.nan}),
    ("alpha-text", {"fdr_alpha": "0.05"}),
    ("cutoff-datetime", {"training_cutoff": datetime(2024, 1, 2)}),
    ("cutoff-text", {"training_cutoff": "2024-01-02"}),
    (
        "previous-p-values-list",
        {"previous_family_p_values": [0.5], "family_trials_used": 3, "project_trials_used": 3},
    ),
    (
        "previous-p-values-too-many",
        {"previous_family_p_values": (0.5, 0.5), "family_trials_used": 1, "project_trials_used": 1},
    ),
    (
        "previous-p-values-negative",
        {"previous_family_p_values": (-0.1,), "family_trials_used": 3, "project_trials_used": 3},
    ),
    (
        "previous-p-values-above-one",
        {"previous_family_p_values": (1.1,), "family_trials_used": 3, "project_trials_used": 3},
    ),
    (
        "previous-p-values-nan",
        {"previous_family_p_values": (math.nan,), "family_trials_used": 3, "project_trials_used": 3},
    ),
    (
        "previous-p-values-text",
        {"previous_family_p_values": ("0.5",), "family_trials_used": 3, "project_trials_used": 3},
    ),
]


@pytest.mark.parametrize(("name", "updates"), _ACCEPTED, ids=[name for name, _ in _ACCEPTED])
def test_every_limit_is_inclusive_where_the_contract_says_so(name: str, updates: dict[str, object]) -> None:
    _validate_config(_base(**updates))


@pytest.mark.parametrize(("name", "updates"), _REJECTED, ids=[name for name, _ in _REJECTED])
def test_values_beyond_each_limit_are_rejected_with_the_stable_code(
    name: str, updates: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match=_INVALID):
        _validate_config(_base(**updates))


def test_a_non_config_object_is_rejected() -> None:
    with pytest.raises(ValueError, match=_INVALID):
        _validate_config(object())  # type: ignore[arg-type]


def test_a_batch_larger_than_the_remaining_family_budget_is_exhausted_not_invalid() -> None:
    _validate_config(_base(budget=18, family_budget=40, family_trials_used=22, project_trials_used=22))
    with pytest.raises(ValueError, match=_EXHAUSTED):
        _validate_config(_base(budget=19, family_budget=40, family_trials_used=22, project_trials_used=22))
    with pytest.raises(ValueError, match=_EXHAUSTED):
        _validate_config(_base(budget=1, family_budget=40, family_trials_used=40, project_trials_used=40))

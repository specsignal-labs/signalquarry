# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from signalquarry._internal.contracts.factor_spec import FactorEvaluationSpecV1
from signalquarry._internal.contracts.spec import HoldoutSpec
from signalquarry._internal.validation.factor_trials import FactorTrialConfiguration


def _identity(character: str) -> str:
    return "sha256:" + character * 64


def _trial() -> FactorTrialConfiguration:
    return FactorTrialConfiguration(
        factor_configuration_hash=_identity("a"),
        dataset_identity=_identity("b"),
        universe_identity=_identity("c"),
        label_identity=_identity("d"),
        decision_sessions=(date(2026, 1, 2), date(2026, 1, 5)),
        evaluation=FactorEvaluationSpecV1(),
        accepted_factor_hashes=(_identity("e"),),
    )


def test_factor_trial_identity_covers_all_inputs() -> None:
    base = _trial()
    assert base.configuration_hash == _trial().configuration_hash
    changed = (
        replace(base, factor_configuration_hash=_identity("f")),
        replace(base, dataset_identity=_identity("f")),
        replace(base, universe_identity=_identity("f")),
        replace(base, label_identity=_identity("f")),
        replace(base, decision_sessions=(date(2026, 1, 2),)),
        replace(base, accepted_factor_hashes=()),
        replace(base, evaluation=FactorEvaluationSpecV1(cost_bps=Decimal("11"))),
        replace(base, evaluation=FactorEvaluationSpecV1(trial_budget=51)),
        replace(base, evaluation=FactorEvaluationSpecV1(horizons=(1, 21))),
        replace(base, evaluation=FactorEvaluationSpecV1(chronological_blocks=7)),
        replace(base, evaluation=FactorEvaluationSpecV1(capital=Decimal("2000000"))),
        replace(base, evaluation=FactorEvaluationSpecV1(holdout=HoldoutSpec(months=6))),
    )
    assert all(item.configuration_hash != base.configuration_hash for item in changed)


def test_factor_evaluation_accepts_yaml_list_horizons() -> None:
    assert FactorEvaluationSpecV1.model_validate({"horizons": [1, 5, 21]}).horizons == (1, 5, 21)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_identity", "bad", "FACTOR_TRIAL_IDENTITY_INVALID"),
        ("accepted_factor_hashes", ("bad",), "FACTOR_TRIAL_IDENTITY_INVALID"),
        ("decision_sessions", (), "FACTOR_TRIAL_SESSIONS_INVALID"),
        ("decision_sessions", (date(2026, 1, 5), date(2026, 1, 2)), "FACTOR_TRIAL_SESSIONS_INVALID"),
        ("accepted_factor_hashes", (_identity("f"), _identity("e")), "FACTOR_TRIAL_ACCEPTED_FACTORS_INVALID"),
        ("evaluation", "invalid", "FACTOR_TRIAL_EVALUATION_INVALID"),
    ],
)
def test_factor_trial_rejects_invalid_inputs(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_trial(), **{field: value})


@pytest.mark.parametrize(
    "evaluation",
    [
        {"horizons": []},
        {"horizons": [5, 1]},
        {"horizons": [1, 1]},
        {"horizons": [0]},
        {"horizons": [True]},
        {"chronological_blocks": 0},
        {"trial_budget": 0},
        {"cost_bps": -1},
        {"capital": 0},
        {"unknown": 1},
    ],
)
def test_factor_evaluation_choices_are_strict(evaluation: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        FactorEvaluationSpecV1.model_validate(evaluation)

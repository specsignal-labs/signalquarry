# SPDX-License-Identifier: Apache-2.0
"""T1 artifacts and formula-trial records are withheld even from NDA exports."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from signalquarry import api
from signalquarry._internal.contracts.factor_spec import FactorEvaluationSpecV1
from signalquarry._internal.factors.expr import GRAMMAR_VERSION, parse_expression
from signalquarry._internal.validation import ledger
from signalquarry._internal.validation.factor_trials import FactorTrialConfiguration, record_formula_trial

from .test_export import _publication, lab  # noqa: F401


@pytest.mark.parametrize("tier", ["public", "nda"])
def test_factor_search_and_formula_trials_never_export(lab: Path, tier: str) -> None:  # noqa: F811
    identity = "sha256:" + "a" * 64
    expression = parse_expression("rank(volume)")
    ledger.append(
        lab,
        "factor_holdouts",
        "sma-trend",
        {
            "kind": "seal",
            "family": "sma-trend",
            "holdout_start": "2026-01-01",
            "trial_budget": 50,
            "holdout": {"months": 12, "training_cutoff": None},
        },
    )
    configuration = FactorTrialConfiguration(
        expression.identity,
        identity,
        identity,
        identity,
        (date(2025, 1, 2),),
        FactorEvaluationSpecV1(),
    )
    record_formula_trial(
        lab,
        family="sma-trend",
        configuration=configuration,
        expression=expression,
        grammar_version=GRAMMAR_VERSION,
        p_value=0.5,
        at=datetime(2025, 1, 3, tzinfo=UTC),
        metrics={"search_identity": identity, "scope": "exploratory"},
    )
    directory = lab / ".signalquarry/factor_searches/private-proposal"
    directory.mkdir(parents=True)
    (directory / "search.json").write_text(
        json.dumps({"scope": "exploratory", "expression": expression.canonical, "search_identity": identity})
    )
    _publication(
        lab,
        "schema: signalquarry.publication/v1\nfamily: sma-trend\nfamily_label: Trend\n"
        "strategies: [{id: sma-trend, title: Trend strategy A, summary: A commercial trend strategy published at the category tier only., assets: US equities}]\n",
    )
    result = api.evidence_export("sma-trend", tier=tier, project=lab)
    assert result.status == "ok", result.as_dict()
    bundle = Path(result.data["path"])
    contents = "\n".join(path.read_text() for path in bundle.rglob("*.json"))
    assert "factor_searches" not in contents and "factor_trials" not in contents
    assert "factor-search" not in contents and "exploratory" not in contents
    assert expression.identity not in contents and expression.canonical not in contents
    assert not any(
        "factor_searches" in path.parts or "factor_trials" in path.name for path in bundle.rglob("*")
    )

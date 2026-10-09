# SPDX-License-Identifier: Apache-2.0
"""Identity and persistence helpers for factor evaluation trials.

Constructing a key does not authorize evaluation. Callers must verify data and
label provenance before recording a real-data evaluation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.contracts.factor_spec import FactorEvaluationSpecV1
from signalquarry._internal.contracts.spec import SLUG_PATTERN
from signalquarry._internal.validation import ledger
from signalquarry._internal.validation.factor_holdouts import factor_evidence_write, factor_family_seal

_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SLUG = re.compile(SLUG_PATTERN)


@dataclass(frozen=True)
class FactorTrialConfiguration:
    factor_configuration_hash: str
    dataset_identity: str
    universe_identity: str
    label_identity: str
    decision_sessions: tuple[date, ...]
    evaluation: FactorEvaluationSpecV1
    accepted_factor_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        identities = (
            self.factor_configuration_hash,
            self.dataset_identity,
            self.universe_identity,
            self.label_identity,
            *self.accepted_factor_hashes,
        )
        if any(not isinstance(value, str) or not _IDENTITY.fullmatch(value) for value in identities):
            raise ValueError("FACTOR_TRIAL_IDENTITY_INVALID")
        if not isinstance(self.evaluation, FactorEvaluationSpecV1):
            raise ValueError("FACTOR_TRIAL_EVALUATION_INVALID")
        if (
            not self.decision_sessions
            or any(type(session) is not date for session in self.decision_sessions)
            or self.decision_sessions != tuple(sorted(set(self.decision_sessions)))
        ):
            raise ValueError("FACTOR_TRIAL_SESSIONS_INVALID")
        if self.accepted_factor_hashes != tuple(sorted(set(self.accepted_factor_hashes))):
            raise ValueError("FACTOR_TRIAL_ACCEPTED_FACTORS_INVALID")

    @property
    def configuration_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "signalquarry.factor-trial-configuration/v1",
                "factor_configuration_hash": self.factor_configuration_hash,
                "dataset_identity": self.dataset_identity,
                "universe_identity": self.universe_identity,
                "label_identity": self.label_identity,
                "decision_sessions": self.decision_sessions,
                "evaluation": self.evaluation.model_dump(mode="json"),
                "accepted_factor_hashes": self.accepted_factor_hashes,
            }
        )


def record_factor_trial(
    root: Path,
    *,
    family: str,
    factor_id: str,
    configuration: FactorTrialConfiguration,
    at: datetime,
    metrics: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Persist one verified evaluation identity; repeated identities are idempotent.

    Identity is per family: the same configuration evaluated under a second family is that family's
    own trial, in the shared project log as well as in per-family logs.

    This is a storage helper, not a provenance verifier or evaluation gate.
    """
    if not _SLUG.fullmatch(family) or not _SLUG.fullmatch(factor_id):
        raise ValueError("FACTOR_TRIAL_FAMILY_INVALID")
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("FACTOR_TRIAL_TIME_INVALID")
    if not isinstance(metrics, dict):
        raise ValueError("FACTOR_TRIAL_METRICS_INVALID")
    trial = {
        "kind": "factor_trial",
        "at": at,
        "family": family,
        "factor_id": factor_id,
        "factor_configuration_hash": configuration.factor_configuration_hash,
        "trial_configuration_hash": configuration.configuration_hash,
        "dataset_identity": configuration.dataset_identity,
        "universe_identity": configuration.universe_identity,
        "label_identity": configuration.label_identity,
        "decision_sessions": configuration.decision_sessions,
        "evaluation": configuration.evaluation.model_dump(mode="json"),
        "accepted_factor_hashes": configuration.accepted_factor_hashes,
        "metrics": metrics,
    }
    log = ledger.factor_trials(root, family)
    for entry in log.entries():
        if (
            entry.get("kind") == "factor_trial"
            and entry.get("family") == family
            and entry.get("trial_configuration_hash") == trial["trial_configuration_hash"]
        ):
            return entry, False
    return ledger.append(root, "factor_trials", family, trial), True


@dataclass(frozen=True)
class FactorTrialAccounting:
    family_trials_used: int
    project_trials_used: int
    previous_family_p_values: tuple[float, ...]
    trials_without_p_value: int
    effective_budget: int
    remaining_budget: int
    seal: dict[str, Any]


class NormalizedFormula(Protocol):
    """The interpreter's normalized result, accepted without importing its layer."""

    @property
    def canonical(self) -> str: ...

    @property
    def identity(self) -> str: ...


def _p_value(value: Any) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError("boolean p-value")
        result = float(value)
        if not math.isfinite(result) or not 0 <= result <= 1:
            raise ValueError("p-value outside [0, 1]")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ledger.LedgerError("FACTOR_SEARCH_INPUT_UNVERIFIED", "invalid trial p-value") from exc
    return result


def factor_trial_accounting(root: Path, family: str) -> FactorTrialAccounting:
    """Derive counts and prior p-values from history, using only the fixed seal budget.

    Every recorded trial row counts conservatively, including a legacy trial
    without a p-value. Strategy trials/extensions cannot change this budget.
    """
    seal = factor_family_seal(root, family)
    if seal is None:
        raise ledger.LedgerError("FACTOR_HOLDOUT_UNSEALED", family)
    trials = [
        entry for entry in ledger.all_entries(root, "factor_trials") if entry.get("kind") == "factor_trial"
    ]
    family_trials = [entry for entry in trials if entry["family"] == family]
    p_values = tuple(
        _p_value(entry["metrics"]["p_value"])
        for entry in family_trials
        if entry.get("metrics", {}).get("p_value") is not None
    )
    budget = seal["trial_budget"]
    return FactorTrialAccounting(
        family_trials_used=len(family_trials),
        project_trials_used=len(trials),
        previous_family_p_values=p_values,
        trials_without_p_value=len(family_trials) - len(p_values),
        effective_budget=budget,
        remaining_budget=max(0, budget - len(family_trials)),
        seal=seal,
    )


def reserve_factor_trials(root: Path, family: str, new_trials: int) -> FactorTrialAccounting:
    """Read-only preflight, not a durable reservation; refuse rather than truncate."""
    if type(new_trials) is not int or new_trials < 0:
        raise ledger.LedgerError("FACTOR_SEARCH_CONFIG_INVALID", "new_trials must be a nonnegative integer")
    accounting = factor_trial_accounting(root, family)
    if new_trials > accounting.remaining_budget:
        raise ledger.LedgerError(
            "FACTOR_SEARCH_BUDGET_EXHAUSTED",
            f"requested {new_trials}; remaining {accounting.remaining_budget}",
        )
    return accounting


def record_formula_trial(
    root: Path,
    *,
    family: str,
    configuration: FactorTrialConfiguration,
    expression: NormalizedFormula,
    grammar_version: int,
    p_value: float,
    at: datetime,
    metrics: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Record one evaluated formula with ledger-enforced seal and budget checks.

    The caller supplies the interpreter's normalized expression and grammar
    version after checking them. The configuration must bind that identity in place of the
    project factor-code identity. This verifies identity/accounting, not input
    provenance, and gives no grade or holdout access. Repeats return the original
    record even at an exhausted budget. No caller-supplied budget is accepted.
    """
    if type(grammar_version) is not int or grammar_version < 1:
        raise ledger.LedgerError("FACTOR_SEARCH_INPUT_UNVERIFIED", "invalid grammar version")
    if configuration.factor_configuration_hash != expression.identity:
        raise ledger.LedgerError(
            "FACTOR_SEARCH_INPUT_UNVERIFIED", "formula identity does not match configuration"
        )
    formula_metrics = {
        **metrics,
        "expression": expression.canonical,
        "grammar_version": grammar_version,
        "p_value": _p_value(p_value),
    }
    with factor_evidence_write(root):
        accounting = factor_trial_accounting(root, family)
        formula_metrics["factor_holdout"] = {
            "seal_hash": accounting.seal["hash"],
            "holdout_start": accounting.seal["holdout_start"],
            "trial_budget": accounting.effective_budget,
            "holdout": accounting.seal["holdout"],
        }
        duplicate = any(
            entry.get("kind") == "factor_trial"
            and entry.get("family") == family
            and entry.get("trial_configuration_hash") == configuration.configuration_hash
            for entry in ledger.factor_trials(root, family).entries()
        )
        if not duplicate and accounting.remaining_budget < 1:
            raise ledger.LedgerError("FACTOR_SEARCH_BUDGET_EXHAUSTED", "remaining 0")
        return record_factor_trial(
            root,
            family=family,
            factor_id="formula-" + expression.identity[7:23],
            configuration=configuration,
            at=at,
            metrics=formula_metrics,
        )

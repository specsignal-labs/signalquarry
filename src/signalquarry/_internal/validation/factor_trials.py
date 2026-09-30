# SPDX-License-Identifier: Apache-2.0
"""Identity and persistence helpers for factor evaluation trials.

Constructing a key does not authorize evaluation. Callers must verify data and
label provenance before recording a real-data evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.contracts.factor_spec import FactorEvaluationSpecV1
from signalquarry._internal.contracts.spec import SLUG_PATTERN
from signalquarry._internal.validation import ledger

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
            and entry.get("trial_configuration_hash") == trial["trial_configuration_hash"]
        ):
            return entry, False
    return ledger.append(root, "factor_trials", family, trial), True

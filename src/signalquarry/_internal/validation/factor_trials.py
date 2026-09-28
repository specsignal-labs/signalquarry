# SPDX-License-Identifier: Apache-2.0
"""Pure identity for a future, provenance-verified factor evaluation trial.

Constructing a key does not authorize evaluation or append to the trial ledger.
The real-data entry point must verify each supplied identity before recording it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.contracts.factor_spec import FactorEvaluationSpecV1

_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")


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

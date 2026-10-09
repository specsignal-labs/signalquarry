# SPDX-License-Identifier: Apache-2.0
"""Fixed factor-family seals and conservative session boundaries; no opening path."""

from __future__ import annotations

import fcntl
import re
from bisect import bisect_left
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from signalquarry._internal.contracts.factor_spec import FactorSpecV1
from signalquarry._internal.contracts.spec import SLUG_PATTERN, HoldoutSpec
from signalquarry._internal.validation import ledger
from signalquarry._internal.validation.evaluate import add_months


def validate_factor_family(family: str) -> None:
    if not re.fullmatch(SLUG_PATTERN, family):
        raise ledger.LedgerError("USAGE_INVALID", "family must be a project slug")


@contextmanager
def factor_evidence_write(root: Path) -> Iterator[None]:
    """Serialize new seal/formula writes without creating or modifying a lock file.

    The existing project config is the POSIX lock anchor. This does not gate the
    legacy storage helper; callers must keep that helper out of search paths.
    """
    with (root / "signalquarry.toml").open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ledger.LedgerError(
                "FACTOR_EVIDENCE_BUSY", "another factor evidence writer is active"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def effective_factor_declaration(factors: Sequence[tuple[FactorSpecV1, str]]) -> dict[str, Any]:
    """Combine a family's registered declarations, keeping each contributor's identity."""
    if not factors:
        raise ledger.LedgerError("FACTOR_FAMILY_EMPTY", "no registered factor in this family")
    cutoffs = [spec.evaluation.holdout.training_cutoff for spec, _ in factors]
    declared_cutoffs = [cutoff for cutoff in cutoffs if cutoff is not None]
    holdout = HoldoutSpec(
        months=max(spec.evaluation.holdout.months for spec, _ in factors),
        training_cutoff=min(declared_cutoffs) if declared_cutoffs else None,
    )
    return {
        "trial_budget": min(spec.evaluation.trial_budget for spec, _ in factors),
        "holdout": holdout.model_dump(mode="json"),
        "factors": [
            {
                "id": spec.id,
                "configuration_hash": identity,
                "trial_budget": spec.evaluation.trial_budget,
                "holdout": spec.evaluation.holdout.model_dump(mode="json"),
            }
            for spec, identity in sorted(factors, key=lambda item: item[0].id)
        ],
    }


def factor_holdout_start_for(holdout: HoldoutSpec, last_session: date) -> date | None:
    """Apply the strategy seal's month-end arithmetic to the effective declaration."""
    starts: list[date] = []
    if holdout.months > 0:
        starts.append(add_months(last_session, -holdout.months) + timedelta(days=1))
    if holdout.training_cutoff is not None:
        starts.append(holdout.training_cutoff + timedelta(days=1))
    return min(starts) if starts else None


def factor_family_seal(root: Path, family: str) -> dict[str, Any] | None:
    """Read a single immutable seal, refusing corrupt or index-mismatched history."""
    validate_factor_family(family)
    problems = ledger.verify_index(root)
    if problems:
        raise ledger.LedgerError("EVIDENCE_INDEX_MISMATCH", "; ".join(problems))
    seals = [
        entry for entry in ledger.factor_holdouts(root, family).entries() if entry.get("family") == family
    ]
    if not seals:
        return None
    try:
        seal = seals[0]
        if len(seals) != 1 or seal["kind"] != "seal" or type(seal["trial_budget"]) is not int:
            raise ValueError("expected one seal with an integer budget")
        if seal["trial_budget"] < 1:
            raise ValueError("budget must be positive")
        date.fromisoformat(seal["holdout_start"])
        HoldoutSpec.model_validate(seal["holdout"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ledger.LedgerError("EVIDENCE_LOG_CORRUPT", "invalid factor seal") from exc
    return seal


def check_factor_training_window(
    *, training_cutoff: date, seal_start: date, longest_horizon: int, sessions: Sequence[date]
) -> None:
    """Require at least ``longest_horizon`` calendar sessions before the first sealed session.

    Cutoffs must be actual dataset sessions; the fixed seal day may be a non-session.
    A calendar with no sealed session cannot establish the boundary and is refused.
    """
    if (
        type(training_cutoff) is not date
        or type(seal_start) is not date
        or type(longest_horizon) is not int
        or longest_horizon < 1
        or not sessions
        or any(type(session) is not date for session in sessions)
        or tuple(sessions) != tuple(sorted(set(sessions)))
    ):
        raise ledger.LedgerError("FACTOR_TRAINING_WINDOW_INVALID", "invalid dataset calendar or horizon")
    cutoff_index = bisect_left(sessions, training_cutoff)
    seal_index = bisect_left(sessions, seal_start)
    if (
        cutoff_index == len(sessions)
        or sessions[cutoff_index] != training_cutoff
        or seal_index == len(sessions)
        or training_cutoff >= seal_start
        or seal_index - cutoff_index < longest_horizon
    ):
        raise ledger.LedgerError("FACTOR_TRAINING_WINDOW_INVALID", "training labels would reach the holdout")

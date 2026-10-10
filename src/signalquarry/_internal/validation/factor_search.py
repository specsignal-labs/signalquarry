# SPDX-License-Identifier: Apache-2.0
"""Ledger-derived search preflight and crash-resume checks, independent of formulas."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from signalquarry._internal.canonical import to_canonical
from signalquarry._internal.validation import ledger
from signalquarry._internal.validation.factor_trials import FactorTrialAccounting, factor_trial_accounting


@contextmanager
def factor_search_write(root: Path) -> Iterator[None]:
    """Serialize searches on the existing directory, alongside per-append evidence locks."""
    handle = os.open(root, os.O_RDONLY)
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ledger.LedgerError("FACTOR_EVIDENCE_BUSY", "another factor search is active") from exc
        yield
    finally:
        os.close(handle)


def search_trial_accounting(
    root: Path, family: str, search_identity: str
) -> tuple[FactorTrialAccounting, tuple[dict[str, Any], ...]]:
    """Subtract this search's rows only; all other families still enter project deflation."""
    current = factor_trial_accounting(root, family)
    own = tuple(
        entry
        for entry in ledger.all_entries(root, "factor_trials")
        if entry.get("kind") == "factor_trial"
        and entry.get("family") == family
        and entry.get("metrics", {}).get("search_identity") == search_identity
    )
    # Validate p-values using the same public accounting path before subtraction.
    prior = list(current.previous_family_p_values)
    for entry in own:
        try:
            prior.remove(float(entry["metrics"]["p_value"]))
        except (KeyError, ValueError, TypeError) as exc:
            raise ledger.LedgerError("FACTOR_SEARCH_INPUT_UNVERIFIED", "incomplete search trial") from exc
    before = replace(
        current,
        family_trials_used=current.family_trials_used - len(own),
        project_trials_used=current.project_trials_used - len(own),
        previous_family_p_values=tuple(prior),
        remaining_budget=current.remaining_budget + len(own),
    )
    return before, own


def verify_search_trials(recorded: Sequence[dict[str, Any]], expected: Sequence[dict[str, Any]]) -> set[str]:
    """Compare all recorded trial fields before allowing any missing append."""
    rows = {row["trial_configuration_hash"]: row for row in expected}
    found: set[str] = set()
    for entry in recorded:
        key = entry["trial_configuration_hash"]
        if (
            key in found
            or key not in rows
            or any(entry.get(field) != to_canonical(value) for field, value in rows.get(key, {}).items())
        ):
            raise ledger.LedgerError("FACTOR_SEARCH_NONDETERMINISTIC", "recorded search trial differs")
        found.add(key)
    return found

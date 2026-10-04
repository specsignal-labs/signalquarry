# SPDX-License-Identifier: Apache-2.0
"""The exact content of a recorded factor trial and the exact errors of recording one."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.contracts.factor_spec import FactorEvaluationSpecV1
from signalquarry._internal.validation.factor_trials import FactorTrialConfiguration, record_factor_trial

from .test_factor_trial_ledger import _configuration, _identity

AT = datetime(2026, 1, 6, 12, 30, tzinfo=UTC)
METRICS = {"mean_ic": 0.12, "observations": 10}


def _record(root: Path, *, family: str = "alpha", factor_id: str = "momentum", **overrides: object):
    options: dict[str, object] = {
        "family": family,
        "factor_id": factor_id,
        "configuration": _configuration(),
        "at": AT,
        "metrics": dict(METRICS),
    }
    options.update(overrides)
    return record_factor_trial(root, **options)  # type: ignore[arg-type]


def test_a_recorded_trial_holds_every_identity_the_configuration_hash_covers(tmp_path: Path) -> None:
    configuration = replace(_configuration(), accepted_factor_hashes=(_identity("e"), _identity("f")))
    entry, created = _record(tmp_path, configuration=configuration)
    assert created is True
    assert entry["kind"] == "factor_trial"
    assert entry["at"] == "2026-01-06T12:30:00Z"
    assert entry["family"] == "alpha" and entry["factor_id"] == "momentum"
    assert entry["factor_configuration_hash"] == _identity("a")
    assert entry["dataset_identity"] == _identity("b")
    assert entry["universe_identity"] == _identity("c")
    assert entry["label_identity"] == _identity("d")
    assert entry["decision_sessions"] == ["2026-01-02", "2026-01-05"]
    assert entry["accepted_factor_hashes"] == [_identity("e"), _identity("f")]
    assert entry["metrics"] == METRICS
    assert entry["trial_configuration_hash"] == configuration.configuration_hash
    assert entry["evaluation"] == configuration.evaluation.model_dump(mode="json")
    assert entry["evaluation"]["cost_bps"] == "10"
    assert entry["schema"] == "signalquarry.factor-trial/v1"
    assert entry["seq"] == 1 and entry["prev"] is None
    body = {key: value for key, value in entry.items() if key != "hash"}
    assert entry["hash"] == canonical_hash(body)


def test_the_trial_configuration_hash_is_a_pinned_canonical_digest() -> None:
    first = _configuration().configuration_hash
    assert first.startswith("sha256:") and len(first) == 71
    assert first == _configuration().configuration_hash
    assert replace(_configuration(), accepted_factor_hashes=(_identity("e"),)).configuration_hash != first
    assert (
        replace(
            _configuration(), evaluation=FactorEvaluationSpecV1(cost_bps=Decimal("11"))
        ).configuration_hash
        != first
    )


def test_repeating_an_identity_returns_the_stored_entry_without_appending(tmp_path: Path) -> None:
    first, created = _record(tmp_path)
    again, repeated = _record(tmp_path, factor_id="renamed", at=AT + timedelta(days=1), metrics={"x": 1})
    assert created is True and repeated is False
    assert again == first
    other, other_created = _record(tmp_path, configuration=_configuration(universe="e"))
    assert other_created is True and other["seq"] == 2 and other["prev"] == first["hash"]


def test_per_family_layout_counts_the_same_identity_once_in_each_family(tmp_path: Path) -> None:
    (tmp_path / "signalquarry.toml").write_text('[evidence]\nlayout = "per_family"\n')
    left, left_created = _record(tmp_path, family="alpha")
    right, right_created = _record(tmp_path, family="beta")
    assert left_created and right_created
    assert left["family"] == "alpha" and right["family"] == "beta"
    assert left["trial_configuration_hash"] == right["trial_configuration_hash"]


def test_the_default_project_layout_has_one_log_so_an_identity_is_stored_once(tmp_path: Path) -> None:
    # Pins current behaviour: the identity excludes the family, so with a single shared log the
    # second family receives the first family's entry rather than a trial of its own.
    left, left_created = _record(tmp_path, family="alpha")
    right, right_created = _record(tmp_path, family="beta")
    assert left_created is True and right_created is False
    assert right == left and right["family"] == "alpha"


def test_the_time_is_recorded_in_utc_whatever_zone_it_was_given_in(tmp_path: Path) -> None:
    entry, _ = _record(tmp_path, at=datetime(2026, 1, 6, 7, 30, tzinfo=timezone(timedelta(hours=-5))))
    assert entry["at"] == "2026-01-06T12:30:00Z"


@pytest.mark.parametrize("family", ["", "Upper", "has space", "-lead", "x" * 200, "a/b"])
def test_an_invalid_family_or_factor_slug_is_refused(tmp_path: Path, family: str) -> None:
    with pytest.raises(ValueError, match="^FACTOR_TRIAL_FAMILY_INVALID$"):
        _record(tmp_path, family=family)
    with pytest.raises(ValueError, match="^FACTOR_TRIAL_FAMILY_INVALID$"):
        _record(tmp_path, factor_id=family)
    assert not list(tmp_path.iterdir())


def test_the_time_must_be_timezone_aware_and_metrics_a_dict(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="^FACTOR_TRIAL_TIME_INVALID$"):
        _record(tmp_path, at=datetime(2026, 1, 6, 12, 30))
    for bad in (None, [], "x", (("a", 1),)):
        with pytest.raises(ValueError, match="^FACTOR_TRIAL_METRICS_INVALID$"):
            _record(tmp_path, metrics=bad)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"factor_configuration_hash": "bad"}, "FACTOR_TRIAL_IDENTITY_INVALID"),
        ({"dataset_identity": "sha256:" + "A" * 64}, "FACTOR_TRIAL_IDENTITY_INVALID"),
        ({"universe_identity": "sha256:" + "a" * 63}, "FACTOR_TRIAL_IDENTITY_INVALID"),
        ({"label_identity": 5}, "FACTOR_TRIAL_IDENTITY_INVALID"),
        ({"accepted_factor_hashes": ("nope",)}, "FACTOR_TRIAL_IDENTITY_INVALID"),
        ({"evaluation": {"cost_bps": 1}}, "FACTOR_TRIAL_EVALUATION_INVALID"),
        ({"decision_sessions": ()}, "FACTOR_TRIAL_SESSIONS_INVALID"),
        ({"decision_sessions": (datetime(2026, 1, 2),)}, "FACTOR_TRIAL_SESSIONS_INVALID"),
        ({"decision_sessions": (date(2026, 1, 2), date(2026, 1, 2))}, "FACTOR_TRIAL_SESSIONS_INVALID"),
        ({"decision_sessions": (date(2026, 1, 5), date(2026, 1, 2))}, "FACTOR_TRIAL_SESSIONS_INVALID"),
        (
            {"accepted_factor_hashes": (_identity("f"), _identity("e"))},
            "FACTOR_TRIAL_ACCEPTED_FACTORS_INVALID",
        ),
        (
            {"accepted_factor_hashes": (_identity("e"), _identity("e"))},
            "FACTOR_TRIAL_ACCEPTED_FACTORS_INVALID",
        ),
    ],
)
def test_configuration_errors_use_the_stable_code(changes: dict[str, object], code: str) -> None:
    with pytest.raises(ValueError, match=f"^{code}$"):
        replace(_configuration(), **changes)  # type: ignore[arg-type]


def test_the_accepted_factor_hashes_default_to_none() -> None:
    configuration = FactorTrialConfiguration(
        factor_configuration_hash=_identity("a"),
        dataset_identity=_identity("b"),
        universe_identity=_identity("c"),
        label_identity=_identity("d"),
        decision_sessions=(date(2026, 1, 2),),
        evaluation=FactorEvaluationSpecV1(),
    )
    assert configuration.accepted_factor_hashes == ()

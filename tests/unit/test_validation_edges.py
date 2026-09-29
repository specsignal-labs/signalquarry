# SPDX-License-Identifier: Apache-2.0
"""Edge cases of the validation, ledger and arm modules."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from signalquarry import api
from signalquarry._internal.canonical import canonical_hash, canonical_json
from signalquarry._internal.paper import arm
from signalquarry._internal.paper.journal import Journal
from signalquarry._internal.paper.models import PaperError
from signalquarry._internal.project.project import load_config, load_strategies
from signalquarry._internal.validation.conformance import import_policy, run_checks
from signalquarry._internal.validation.evaluate import add_months, max_drawdown
from signalquarry._internal.validation.ledger import ChainedLog, LedgerError
from signalquarry._internal.validation.metrics import daily_returns, summarize
from signalquarry._internal.validation.stats import (
    ReturnMoments,
    block_bootstrap_sharpe,
    min_track_record_length,
    moments,
    probabilistic_sharpe,
)


def test_ledger_skips_blank_lines_and_rejects_garbage(tmp_path: Path) -> None:
    log = ChainedLog(tmp_path / "log.jsonl", "t/v1")
    assert log.head() is None
    entry = log.append({"x": 1})
    log.path.write_text(log.path.read_text() + "\n\n")
    assert log.head() == entry["hash"]
    log.path.write_text(log.path.read_text() + "{not json\n")
    with pytest.raises(LedgerError) as info:
        log.entries()
    assert info.value.code == "EVIDENCE_LOG_CORRUPT"


@pytest.mark.parametrize(
    ("field", "value"),
    [("seq", 2), ("seq", True), ("seq", 1.0), ("prev", "sha256:wrong")],
)
def test_ledger_rejects_rehashed_sequence_and_previous_hash_mismatches(
    tmp_path: Path, field: str, value: object
) -> None:
    log = ChainedLog(tmp_path / "log.jsonl", "t/v1")
    entry = log.append({"x": 1})
    entry[field] = value
    entry["hash"] = canonical_hash({key: item for key, item in entry.items() if key != "hash"})
    log.path.write_text(canonical_json(entry) + "\n")

    with pytest.raises(LedgerError) as info:
        log.entries()
    assert info.value.code == "EVIDENCE_LOG_CORRUPT"


def test_ledger_rejects_a_validly_hashed_record_with_the_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    ChainedLog(path, "signalquarry.other/v1").append({"x": 1})

    with pytest.raises(LedgerError) as info:
        ChainedLog(path, "signalquarry.trial/v1").entries()

    assert info.value.code == "EVIDENCE_LOG_CORRUPT"


def test_ledger_error_keeps_its_reason_and_optional_detail() -> None:
    assert str(LedgerError("EVIDENCE_LOG_CORRUPT")) == "EVIDENCE_LOG_CORRUPT"
    assert str(LedgerError("EVIDENCE_LOG_CORRUPT", "trials.jsonl:2")) == (
        "EVIDENCE_LOG_CORRUPT:trials.jsonl:2"
    )


def test_metrics_and_stats_degenerate_inputs() -> None:
    assert len(daily_returns([Decimal(1)])) == 0
    assert summarize([], [], Decimal(1)) == {"sessions": 0}
    short = moments(np.array([0.01, 0.02]))
    assert short.n == 2 and math.isnan(short.sharpe) and math.isnan(probabilistic_sharpe(short))
    flat = moments(np.zeros(10))
    assert (flat.sharpe, flat.skew, flat.kurtosis) == (0.0, 0.0, 3.0)
    assert min_track_record_length(ReturnMoments(100, -0.1, 0.0, 3.0)) == float("inf")
    low, high = block_bootstrap_sharpe(np.ones(10) * 0.001)
    assert math.isnan(low) and math.isnan(high)
    assert max_drawdown(np.array([])) == 0.0
    assert add_months(date(2025, 1, 31), 1) == date(2025, 2, 28)
    assert add_months(date(2025, 1, 1), 1) == date(2025, 2, 1)
    assert add_months(date(2024, 3, 31), -1) == date(2024, 2, 29)


def test_import_policy_catches_calls_attributes_and_globals(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "strategy.py").write_text(
        "import numpy\nCOUNT = 0\n\ndef f():\n    global COUNT\n    open('x')\n    return numpy.datetime64.now\n"
    )
    result = import_policy(package, "pkg")
    assert not result.ok
    assert "call open()" in result.detail and "global statement" in result.detail and ".now" in result.detail


def test_contract_failure_is_reported_by_check(tmp_path: Path) -> None:
    root = tmp_path / "contract"
    assert api.init(root, demo=True, package="contract_edge_lab").status == "ok"
    path = root / "src" / "contract_edge_lab" / "sma_trend" / "strategy.py"
    path.write_text(path.read_text().replace('"PRICE_BELOW_SMA")', '"NOT_DECLARED")'))
    checks = {c.name: c for c in run_checks(load_strategies(load_config(root))["sma-trend"])}
    assert checks["contract"].ok is False and "REASON_CODE_UNDECLARED" in checks["contract"].detail


def test_arm_token_edge_cases(tmp_path: Path) -> None:
    path = tmp_path / "arm.json"
    assert arm.read(path) is None
    path.write_text("{broken")
    assert arm.read(path) is None
    path.write_text("[1, 2]")
    assert arm.read(path) is None
    journal = Journal.open(tmp_path / "journal.jsonl")
    now = datetime(2026, 1, 5, 14, tzinfo=UTC)
    token = arm.issue(
        alias="one",
        strategy_id="s",
        configuration_hash="c",
        freeze_hash="f",
        account_id="a",
        broker="fake",
        journal_head=None,
        now=now,
        days=5,
    )
    journal.append("armed", {"token_hash": token["token_hash"]}, now=now)
    kwargs = dict(configuration_hash="c", freeze_hash="f", account_id="a", now=now)
    assert arm.check(token, journal, alias="one", **kwargs) == token
    with pytest.raises(PaperError) as info:
        arm.check(token, journal, alias="two", **kwargs)
    assert info.value.code == "PAPER_NOT_ARMED"


def test_journal_refuses_reserved_fields(tmp_path: Path) -> None:
    journal = Journal.open(tmp_path / "j.jsonl")
    with pytest.raises(ValueError, match="JOURNAL_FIELD_RESERVED"):
        journal.append("note", {"kind": "other"})


def test_holdout_start_prefers_the_earlier_of_months_and_training_cutoff() -> None:
    from datetime import date

    from signalquarry.api.evidence import holdout_start_for
    from tests.helpers import spec as make_spec

    last = date(2025, 6, 30)
    assert holdout_start_for(make_spec(("SYNA",)), last) == date(2024, 7, 1)
    early = make_spec(("SYNA",), evaluation={"holdout": {"months": 12, "training_cutoff": "2023-12-31"}})
    assert holdout_start_for(early, last) == date(2024, 1, 1)
    late = make_spec(("SYNA",), evaluation={"holdout": {"months": 12, "training_cutoff": "2025-03-31"}})
    assert holdout_start_for(late, last) == date(2024, 7, 1)
    only_cutoff = make_spec(("SYNA",), evaluation={"holdout": {"months": 0, "training_cutoff": "2025-03-31"}})
    assert holdout_start_for(only_cutoff, last) == date(2025, 4, 1)
    assert holdout_start_for(make_spec(("SYNA",), evaluation={"holdout": {"months": 0}}), last) is None
    assert "training_cutoff" not in make_spec(("SYNA",)).outcome_document()["evaluation"]["holdout"]

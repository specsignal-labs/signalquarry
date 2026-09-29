# SPDX-License-Identifier: Apache-2.0
"""Reference cases for strategy conformance's decision and look-ahead contract."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from signalquarry._internal.data.dataset import Dataset, Dividend, Split, SymbolSeries
from signalquarry._internal.engine.backtest import EngineError
from signalquarry._internal.validation import conformance
from tests.helpers import weekdays


def _harness(monkeypatch: pytest.MonkeyPatch, *, options: bool) -> dict[str, Any]:
    sessions = weekdays(date(2025, 1, 2), 100)
    wanted = "open" if options else "target"
    decisions = [
        {
            "session": day.isoformat(),
            "action": action,
            "checkpoint": checkpoint,
            "reason_codes": ["REFERENCE"],
        }
        for day in sessions
        for action, checkpoint in ((wanted, "open"), ("hold", "close"), ("close", "close"))
    ]
    state: dict[str, Any] = {
        "sessions": sessions,
        "decisions": decisions,
        "calls": [],
        "perturbations": [],
        "imports": [],
        "datasets": [],
        "mutator": lambda original, _cut: original,
        "second_hash": "same-hash",
    }
    strategy = SimpleNamespace(
        module=SimpleNamespace(__name__="research_pkg.strategy"),
        package_dir=Path("/research_pkg"),
        spec=SimpleNamespace(data=SimpleNamespace(symbols=("SYNA",))),
        definition=object(),
        params=object(),
    )
    state["strategy"] = strategy
    dataset = SimpleNamespace(sessions=sessions)

    def import_stub(package_dir: Path, own_package: str) -> conformance.CheckResult:
        state["imports"].append((package_dir, own_package))
        return conformance.CheckResult("import_policy", True)

    def dataset_stub(start: date, end: date, *, symbols: tuple[str, ...]) -> object:
        state["datasets"].append((start, end, symbols))
        return dataset

    def perturb_stub(
        original: object,
        cut: int,
        low: float,
        high: float,
        *,
        keep_open_at_cut: bool,
    ) -> tuple[int, float, float, bool]:
        assert original is dataset
        state["perturbations"].append((cut, low, high, keep_open_at_cut))
        return cut, low, high, keep_open_at_cut

    def simulate_stub(spec: object, definition: object, params: object, data: object) -> SimpleNamespace:
        assert (spec, definition, params) == (
            strategy.spec,
            strategy.definition,
            strategy.params,
        )
        state["calls"].append(data)
        if data is dataset:
            ledger_hash = "same-hash" if len(state["calls"]) == 1 else state["second_hash"]
            return SimpleNamespace(decisions=decisions, fills=[object()], ledger_hash=ledger_hash)
        cut = data[0]
        return SimpleNamespace(
            decisions=state["mutator"](decisions, cut), fills=[object()], ledger_hash="mutated"
        )

    monkeypatch.setattr(conformance, "import_policy", import_stub)
    monkeypatch.setattr(conformance, "synthetic_dataset", dataset_stub)
    monkeypatch.setattr(conformance, "is_options", lambda _spec: options)
    monkeypatch.setattr(conformance, "perturb_after", perturb_stub)
    monkeypatch.setattr(conformance, "simulate", simulate_stub)
    return state


@pytest.mark.parametrize("options", [False, True])
def test_conformance_uses_declared_package_and_both_perturbation_directions(
    monkeypatch: pytest.MonkeyPatch, options: bool
) -> None:
    state = _harness(monkeypatch, options=options)
    results = conformance.run_checks(state["strategy"])
    assert [item.name for item in results] == [
        "import_policy",
        "contract",
        "determinism",
        "lookahead",
        "produces_targets",
    ]
    assert all(item.ok is True for item in results)
    assert state["imports"] == [(Path("/research_pkg"), "research_pkg")]
    assert state["datasets"] == [(conformance.CHECK_START, conformance.CHECK_END, ("SYNA",))]
    assert results[1].detail == "300 decisions, 1 fills on synthetic data"
    assert results[2].detail == "same-hash"
    assert results[3].detail == "decisions never changed when only later bars changed (3 cuts, 2 directions)"
    wanted = "open" if options else "target"
    assert results[4].detail == f"100 {wanted} decisions"
    fractions = (0.45, 0.6, 0.75) if not options else tuple(0.3 + 0.05 * k for k in range(12))
    expected = [
        (int(100 * fraction), low, high, options)
        for fraction in fractions
        for low, high in ((0.5, 0.8), (2.0, 1.3))
    ]
    assert state["perturbations"] == expected
    assert state["calls"][:2] == [state["calls"][0]] * 2
    assert state["calls"][2:] == expected


@pytest.mark.parametrize(("checkpoint", "leak_expected"), [("close", False), ("open", True)])
def test_options_cut_includes_only_open_checkpoint(
    monkeypatch: pytest.MonkeyPatch, checkpoint: str, leak_expected: bool
) -> None:
    state = _harness(monkeypatch, options=True)

    def change_at_cut(original: list[dict[str, Any]], cut: int) -> list[dict[str, Any]]:
        changed = deepcopy(original)
        for record in changed:
            if record["session"] == state["sessions"][cut].isoformat() and record["checkpoint"] == checkpoint:
                record["reason_codes"] = ["CHANGED"]
        return changed

    state["mutator"] = change_at_cut
    results = {item.name: item for item in conformance.run_checks(state["strategy"])}
    assert results["lookahead"].ok is not leak_expected
    if leak_expected:
        assert results["lookahead"].detail.startswith(
            f"decision for {state['sessions'][30].isoformat()} changed"
        )


def test_equity_cut_session_decision_change_is_a_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _harness(monkeypatch, options=False)

    def change_at_cut(original: list[dict[str, Any]], cut: int) -> list[dict[str, Any]]:
        changed = deepcopy(original)
        for record in changed:
            if record["session"] == state["sessions"][cut].isoformat() and record["action"] == "target":
                record["reason_codes"] = ["CHANGED"]
        return changed

    state["mutator"] = change_at_cut
    results = {item.name: item for item in conformance.run_checks(state["strategy"])}
    assert results["lookahead"].ok is False
    assert results["lookahead"].detail.startswith(f"decision for {state['sessions'][45].isoformat()} changed")


def test_equity_lookahead_check_compares_all_decisions_before_the_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _harness(monkeypatch, options=False)

    def change_previous_session(original: list[dict[str, Any]], cut: int) -> list[dict[str, Any]]:
        changed = deepcopy(original)
        previous_day = state["sessions"][cut - 1].isoformat()
        for record in changed:
            if record["session"] == previous_day and record["action"] == "target":
                record["reason_codes"] = ["CHANGED"]
        return changed

    state["mutator"] = change_previous_session
    results = {item.name: item for item in conformance.run_checks(state["strategy"])}

    assert results["lookahead"].ok is False


def test_equity_lookahead_check_reports_when_a_suffix_of_known_decisions_disappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _harness(monkeypatch, options=False)

    def remove_cut_session(original: list[dict[str, Any]], cut: int) -> list[dict[str, Any]]:
        cut_day = state["sessions"][cut].isoformat()
        return [record for record in original if record["session"] != cut_day]

    state["mutator"] = remove_cut_session
    results = {item.name: item for item in conformance.run_checks(state["strategy"])}

    assert results["lookahead"].ok is False


def test_determinism_mismatch_and_check_result_document(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _harness(monkeypatch, options=False)
    state["second_hash"] = "other-hash"
    results = {item.name: item for item in conformance.run_checks(state["strategy"])}
    assert results["determinism"].as_dict() == {"name": "determinism", "ok": False, "detail": "same-hash"}


def test_engine_failure_is_a_failed_contract_check(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _harness(monkeypatch, options=False)
    monkeypatch.setattr(
        conformance, "simulate", lambda *_args: (_ for _ in ()).throw(EngineError("INVALID_REASON"))
    )
    results = conformance.run_checks(state["strategy"])
    assert [item.as_dict() for item in results] == [
        {"name": "import_policy", "ok": True, "detail": ""},
        {"name": "contract", "ok": False, "detail": "INVALID_REASON"},
    ]


def test_perturbed_run_failure_is_a_failed_lookahead_check(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _harness(monkeypatch, options=False)
    original_simulate = conformance.simulate

    def fail_mutated(*args: object) -> SimpleNamespace:
        if len(state["calls"]) >= 2:
            raise EngineError("PERTURBED_FAIL")
        return original_simulate(*args)

    monkeypatch.setattr(conformance, "simulate", fail_mutated)
    results = conformance.run_checks(state["strategy"])
    assert [item.name for item in results] == ["import_policy", "contract", "determinism", "lookahead"]
    assert results[-1].as_dict() == {
        "name": "lookahead",
        "ok": False,
        "detail": "mutated run failed: PERTURBED_FAIL",
    }


def test_no_target_decisions_fails_activity_check(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _harness(monkeypatch, options=False)
    state["decisions"][:] = [record for record in state["decisions"] if record["action"] != "target"]
    results = {item.name: item for item in conformance.run_checks(state["strategy"])}
    assert results["produces_targets"].as_dict() == {
        "name": "produces_targets",
        "ok": False,
        "detail": "never returned target on synthetic data",
    }


def test_import_policy_preserves_own_package_and_reports_only_first_ten_violations(
    tmp_path: Path,
) -> None:
    package = tmp_path / "research_pkg"
    package.mkdir()
    forbidden = [
        "os",
        "sys",
        "subprocess",
        "socket",
        "random",
        "time",
        "pathlib",
        "urllib",
        "http",
        "inspect",
        "builtins",
    ]
    (package / "strategy.py").write_text(
        "import research_pkg.helper\nfrom research_pkg.util import x\nimport numpy.linalg\n"
        + "".join(f"import {root}.detail\n" for root in forbidden)
    )
    result = conformance.import_policy(package, "research_pkg")
    assert result.as_dict() == {
        "name": "import_policy",
        "ok": False,
        "detail": "; ".join(
            f"strategy.py:{line}: import {root}" for line, root in enumerate(forbidden[:10], 4)
        ),
    }


def test_import_policy_syntax_error_identifies_the_source_file(tmp_path: Path) -> None:
    package = tmp_path / "research_pkg"
    package.mkdir()
    path = package / "broken.py"
    path.write_text("def broken(:\n")
    with pytest.raises(SyntaxError) as info:
        conformance.import_policy(package, "research_pkg")
    assert info.value.filename == str(path)


def test_perturb_after_scales_only_the_requested_tail_and_preserves_input() -> None:
    sessions = tuple(weekdays(date(2025, 1, 2), 4))
    original = np.array([100, 200, 300, 400], dtype=np.int64)
    series = SymbolSeries(
        micro={name: original.copy() for name in ("open", "high", "low", "close")},
        volume=np.array([1.0, 2.0, 3.0, 4.0]),
        present=np.ones(4, dtype=bool),
    )
    split = Split("SYNA", sessions[2], Decimal(2))
    dividend = Dividend("SYNA", sessions[1], sessions[3], Decimal("0.25"))
    dataset = Dataset(sessions, {"SYNA": series}, (split,), (dividend,), source="reference")
    changed = conformance.perturb_after(dataset, 2, 0.5, 2.0, keep_open_at_cut=True)
    assert changed is not dataset
    assert changed.sessions == sessions and changed.source == "reference:perturbed@2"
    assert changed.splits == (split,) and changed.dividends == (dividend,)
    np.testing.assert_array_equal(dataset.series["SYNA"].micro["close"], original)
    for name in ("high", "low", "close"):
        np.testing.assert_array_equal(changed.series["SYNA"].micro[name], [100, 200, 150, 800])
    np.testing.assert_array_equal(changed.series["SYNA"].micro["open"], [100, 200, 300, 800])


def test_perturb_after_default_arguments_use_the_default_ramp_and_change_cut_open() -> None:
    sessions = tuple(weekdays(date(2025, 1, 2), 4))
    original = np.array([100, 200, 300, 400], dtype=np.int64)
    series = SymbolSeries(
        micro={name: original.copy() for name in ("open", "high", "low", "close")},
        volume=np.array([1.0, 2.0, 3.0, 4.0]),
        present=np.ones(4, dtype=bool),
    )
    changed = conformance.perturb_after(Dataset(sessions, {"SYNA": series}), 2)

    np.testing.assert_array_equal(changed.series["SYNA"].micro["open"], [100, 200, 180, 680])

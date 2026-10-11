# SPDX-License-Identifier: Apache-2.0
"""Behavioural contract for declarative studies, without strategy execution."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from signalquarry._internal.contracts.study import (
    Baseline,
    CompareRule,
    StudySpecV1,
    StudyWindow,
    Variant,
    load_study,
)

EXAMPLE = """\
schema: signalquarry.study/v1
id: trend-filter-vs-hold
hypothesis:
  statement: Holding only above the 200-session average lowers drawdown versus buy-and-hold.
  falsification: Max drawdown is not lower than buy-and-hold over the same sessions.
base: sma-trend
dataset: alpaca-sip-1day-abc123
window: {start: 2016-01-01, end: 2024-12-31}
baselines:
  - {id: buy-and-hold, kind: benchmark}
  - {id: vol-matched, kind: benchmark_scaled}
  - {id: other-strategy, kind: strategy, strategy: sma-cross}
variants:
  - {id: p150, params: {period: 150}}
  - {id: no-filter, params: {period: 20}, role: ablation, note: Removes the slow filter.}
  - {id: costs-x2, execution: {costs: {bps: "10"}}, role: sensitivity}
grid: {period: [100, 150, 200]}
compare: {metric: max_drawdown, direction: lower, versus: buy-and-hold}
"""


@pytest.fixture
def document() -> dict[str, Any]:
    return {
        "id": "trend-filter-vs-hold",
        "hypothesis": {
            "statement": "The slow filter reduces drawdown.",
            "falsification": "Drawdown is at least as large as buy-and-hold.",
        },
        "base": "sma-trend",
        "baselines": [{"id": "buy-and-hold", "kind": "benchmark"}],
        "compare": {"metric": "max_drawdown", "direction": "lower", "versus": "buy-and-hold"},
    }


def assert_code(exc: pytest.ExceptionInfo[ValidationError], code: str) -> None:
    errors = exc.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "value_error"
    assert str(errors[0]["ctx"]["error"]) == code
    assert code in str(exc.value)


def test_full_example_round_trips() -> None:
    study = StudySpecV1.model_validate(yaml.safe_load(EXAMPLE))
    expected = {
        "schema": "signalquarry.study/v1",
        "id": "trend-filter-vs-hold",
        "hypothesis": {
            "statement": "Holding only above the 200-session average lowers drawdown versus buy-and-hold.",
            "falsification": "Max drawdown is not lower than buy-and-hold over the same sessions.",
        },
        "base": "sma-trend",
        "dataset": "alpaca-sip-1day-abc123",
        "window": {"start": "2016-01-01", "end": "2024-12-31"},
        "baselines": [
            {"id": "buy-and-hold", "kind": "benchmark", "strategy": None, "symbol": None},
            {"id": "vol-matched", "kind": "benchmark_scaled", "strategy": None, "symbol": None},
            {"id": "other-strategy", "kind": "strategy", "strategy": "sma-cross", "symbol": None},
        ],
        "variants": [
            {"id": "p150", "params": {"period": 150}, "execution": {}, "role": "candidate", "note": None},
            {
                "id": "no-filter",
                "params": {"period": 20},
                "execution": {},
                "role": "ablation",
                "note": "Removes the slow filter.",
            },
            {
                "id": "costs-x2",
                "params": {},
                "execution": {"costs": {"bps": "10"}},
                "role": "sensitivity",
                "note": None,
            },
        ],
        "grid": {"period": [100, 150, 200]},
        "compare": {"metric": "max_drawdown", "direction": "lower", "versus": "buy-and-hold"},
    }
    assert study.study_document() == expected
    restored = StudySpecV1.model_validate(json.loads(json.dumps(expected)))
    assert restored == study
    assert restored.study_document() == expected


def test_study_defaults(document: dict[str, Any]) -> None:
    study = StudySpecV1.model_validate(document)
    assert study.schema_ == "signalquarry.study/v1"
    assert study.dataset is None
    assert study.window == StudyWindow(start=None, end=None)
    assert study.variants == ()
    assert study.grid == {}
    assert study.baselines[0].strategy is None
    assert study.baselines[0].symbol is None


def test_variant_defaults() -> None:
    variant = Variant(id="p150", params={"period": 150})
    assert variant.execution == {}
    assert variant.role == "candidate"
    assert variant.note is None


def test_default_mappings_are_independent(document: dict[str, Any]) -> None:
    first = StudySpecV1.model_validate(document)
    second = StudySpecV1.model_validate(document)
    assert first.grid is not second.grid
    first_variant = Variant(id="costs-x2", execution={"costs": {"bps": "10"}})
    second_variant = Variant(id="costs-x2", execution={"costs": {"bps": "10"}})
    assert first_variant.params is not second_variant.params


@pytest.mark.parametrize(
    ("start", "end"),
    [(None, None), ("2016-01-01", None), (None, "2024-12-31"), ("2024-12-31", "2024-12-31")],
)
def test_window_allows_open_and_equal_bounds(start: str | None, end: str | None) -> None:
    window = StudyWindow.model_validate({"start": start, "end": end})
    assert window.start == (date.fromisoformat(start) if start else None)
    assert window.end == (date.fromisoformat(end) if end else None)


def test_window_rejects_reversed_bounds() -> None:
    with pytest.raises(ValidationError) as exc:
        StudyWindow.model_validate({"start": "2024-12-31", "end": "2016-01-01"})
    assert_code(exc, "STUDY_WINDOW_INVALID")


@pytest.mark.parametrize("kind", ["benchmark", "benchmark_scaled"])
def test_benchmark_symbol_is_optional(kind: str) -> None:
    baseline = Baseline.model_validate({"id": "buy-and-hold", "kind": kind})
    assert baseline.symbol is None
    assert baseline.strategy is None


@pytest.mark.parametrize("kind", ["benchmark", "benchmark_scaled"])
def test_benchmark_allows_explicit_symbol(kind: str) -> None:
    baseline = Baseline.model_validate({"id": "buy-and-hold", "kind": kind, "symbol": "BRK.B"})
    assert baseline.symbol == "BRK.B"


def test_strategy_baseline_requires_strategy() -> None:
    with pytest.raises(ValidationError) as exc:
        Baseline(id="other-strategy", kind="strategy")
    assert_code(exc, "STUDY_BASELINE_STRATEGY_INVALID")


@pytest.mark.parametrize("kind", ["benchmark", "benchmark_scaled"])
def test_benchmark_forbids_strategy(kind: str) -> None:
    with pytest.raises(ValidationError) as exc:
        Baseline.model_validate({"id": "buy-and-hold", "kind": kind, "strategy": "sma-cross"})
    assert_code(exc, "STUDY_BASELINE_STRATEGY_INVALID")


def test_strategy_baseline_forbids_symbol() -> None:
    with pytest.raises(ValidationError) as exc:
        Baseline(id="other-strategy", kind="strategy", strategy="sma-cross", symbol="SPY")
    assert_code(exc, "STUDY_BASELINE_SYMBOL_INVALID")


@pytest.mark.parametrize("role", ["candidate", "ablation", "sensitivity"])
def test_variant_rejects_no_changes(role: str) -> None:
    with pytest.raises(ValidationError) as exc:
        Variant.model_validate({"id": "no-change", "role": role, "note": "Only a note."})
    assert_code(exc, "STUDY_VARIANT_EMPTY")


def test_sensitivity_forbids_params() -> None:
    with pytest.raises(ValidationError) as exc:
        Variant(id="costs-x2", params={"period": 150}, execution={"costs": {"bps": "10"}}, role="sensitivity")
    assert_code(exc, "STUDY_SENSITIVITY_CHANGES_PARAMS")


def test_sensitivity_allows_nested_execution_override() -> None:
    variant = Variant(id="costs-x2", execution={"costs": {"bps": "10"}}, role="sensitivity")
    assert variant.params == {}
    assert variant.execution == {"costs": {"bps": "10"}}


def test_override_contents_are_deferred_to_strategy_validation() -> None:
    variant = Variant(id="new-assumption", params={"custom": [1, 2]}, execution={"custom": {"nested": True}})
    assert variant.params == {"custom": [1, 2]}
    assert variant.execution == {"custom": {"nested": True}}


def test_note_accepts_500_characters() -> None:
    variant = Variant(id="p150", params={"period": 150}, note="x" * 500)
    assert variant.note == "x" * 500


def test_note_rejects_501_characters() -> None:
    with pytest.raises(ValidationError) as exc:
        Variant(id="p150", params={"period": 150}, note="x" * 501)
    assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [(("note",), "string_too_long")]


@pytest.mark.parametrize("location", ["baselines", "variants", "across"])
def test_arm_ids_are_unique(document: dict[str, Any], location: str) -> None:
    if location == "baselines":
        document["baselines"].append({"id": "buy-and-hold", "kind": "benchmark_scaled"})
        duplicate = "buy-and-hold"
    elif location == "variants":
        document["variants"] = [{"id": "p150", "params": {"period": p}} for p in (150, 200)]
        duplicate = "p150"
    else:
        document["variants"] = [{"id": "buy-and-hold", "params": {"period": 150}}]
        duplicate = "buy-and-hold"
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert_code(exc, f"STUDY_ARM_ID_DUPLICATE:{duplicate}")


@pytest.mark.parametrize("location", ["baselines", "variants"])
def test_subject_arm_id_is_reserved(document: dict[str, Any], location: str) -> None:
    if location == "baselines":
        document["baselines"].append({"id": "base", "kind": "benchmark"})
    else:
        document["variants"] = [{"id": "base", "params": {"period": 150}}]
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert_code(exc, "STUDY_ARM_ID_DUPLICATE:base")


@pytest.mark.parametrize("versus", ["missing-baseline", "p150", "base"])
def test_compare_requires_a_baseline_id(document: dict[str, Any], versus: str) -> None:
    document["variants"] = [{"id": "p150", "params": {"period": 150}}]
    document["compare"]["versus"] = versus
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert_code(exc, "STUDY_COMPARE_VERSUS_UNKNOWN")


@pytest.mark.parametrize(
    "metric", ["total_return", "cagr", "sharpe", "sortino", "calmar", "max_drawdown", "annual_volatility"]
)
def test_compare_supports_declared_metrics(document: dict[str, Any], metric: str) -> None:
    document["compare"]["metric"] = metric
    assert StudySpecV1.model_validate(document).compare.metric == metric


@pytest.mark.parametrize("direction", ["higher", "lower"])
def test_compare_supports_both_directions(document: dict[str, Any], direction: str) -> None:
    document["compare"]["direction"] = direction
    assert StudySpecV1.model_validate(document).compare.direction == direction


@pytest.mark.parametrize(
    ("axis", "values"),
    [
        ("", [100]),
        (" ", [100]),
        ("period", []),
        ("period", [100, 100]),
        ("period", [None]),
        ("period", [[100]]),
        ("period", [{"value": 100}]),
        ("period", [date(2024, 1, 1)]),
        ("period", None),
        ("period", 100),
        ("period", "100"),
    ],
)
def test_grid_rejects_invalid_axis(document: dict[str, Any], axis: str, values: Any) -> None:
    document["grid"] = {axis: values}
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert_code(exc, f"STUDY_GRID_INVALID:{axis}")


def test_grid_accepts_all_scalar_types(document: dict[str, Any]) -> None:
    document["grid"] = {"value": ["text", 2, 0.5, False]}
    study = StudySpecV1.model_validate(document)
    assert study.grid["value"] == ("text", 2, 0.5, False)
    assert [type(v.params["value"]) for v in study.grid_variants()] == [str, int, float, bool]


def test_grid_expands_in_axis_order(document: dict[str, Any]) -> None:
    document["grid"] = {"period": [100, 150], "weight": ["0.5", "1.0"]}
    variants = StudySpecV1.model_validate(document).grid_variants()
    assert [(v.id, v.params, v.role) for v in variants] == [
        ("grid-period-100-weight-0-5", {"period": 100, "weight": "0.5"}, "candidate"),
        ("grid-period-100-weight-1-0", {"period": 100, "weight": "1.0"}, "candidate"),
        ("grid-period-150-weight-0-5", {"period": 150, "weight": "0.5"}, "candidate"),
        ("grid-period-150-weight-1-0", {"period": 150, "weight": "1.0"}, "candidate"),
    ]


def test_grid_normalises_slug_runs_and_edges(document: dict[str, Any]) -> None:
    document["grid"] = {" Fast / Period ": ["UPPER...Value!!!"]}
    variants = StudySpecV1.model_validate(document).grid_variants()
    assert [v.id for v in variants] == ["grid-fast-period-upper-value"]
    assert variants[0].params == {" Fast / Period ": "UPPER...Value!!!"}


def test_empty_grid_generates_no_variants(document: dict[str, Any]) -> None:
    assert StudySpecV1.model_validate(document).grid_variants() == ()


def test_all_variants_puts_declared_variants_first(document: dict[str, Any]) -> None:
    document["variants"] = [{"id": "p150", "params": {"period": 150}, "role": "ablation"}]
    document["grid"] = {"period": [100, 200]}
    study = StudySpecV1.model_validate(document)
    assert study.all_variants() == study.variants + study.grid_variants()
    assert [v.id for v in study.all_variants()] == ["p150", "grid-period-100", "grid-period-200"]


@pytest.mark.parametrize("location", ["variants", "baselines"])
def test_grid_id_cannot_collide_with_declared_arm(document: dict[str, Any], location: str) -> None:
    document["grid"] = {"period": [100]}
    if location == "variants":
        document["variants"] = [{"id": "grid-period-100", "params": {"period": 150}}]
    else:
        document["baselines"].append({"id": "grid-period-100", "kind": "benchmark"})
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert_code(exc, "STUDY_GRID_ID_INVALID:grid-period-100")


def test_normalised_grid_ids_cannot_collide(document: dict[str, Any]) -> None:
    document["grid"] = {"value": ["A.B", "a-b"]}
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert_code(exc, "STUDY_GRID_ID_INVALID:grid-value-a-b")


def test_grid_id_accepts_64_characters(document: dict[str, Any]) -> None:
    document["grid"] = {"value": ["a" * 53]}
    assert StudySpecV1.model_validate(document).grid_variants()[0].id == "grid-value-" + "a" * 53


def test_grid_id_rejects_65_characters(document: dict[str, Any]) -> None:
    document["grid"] = {"value": ["a" * 54]}
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert_code(exc, "STUDY_GRID_ID_INVALID:grid-value-" + "a" * 54)


def test_arm_count_includes_only_runnable_strategies() -> None:
    study = StudySpecV1.model_validate(yaml.safe_load(EXAMPLE))
    assert study.arm_count() == 8  # subject + three variants + three grid points + one strategy baseline


def test_arm_count_excludes_benchmark_baselines(document: dict[str, Any]) -> None:
    document["baselines"].append({"id": "vol-matched", "kind": "benchmark_scaled"})
    assert StudySpecV1.model_validate(document).arm_count() == 1


def test_grid_allows_200_runnable_arms(document: dict[str, Any]) -> None:
    document["grid"] = {"period": list(range(199))}
    study = StudySpecV1.model_validate(document)
    assert study.arm_count() == 200
    assert len(study.all_variants()) == 199


def test_grid_rejects_201_runnable_arms(document: dict[str, Any]) -> None:
    document["grid"] = {"period": list(range(200))}
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert_code(exc, "STUDY_TOO_MANY_ARMS")


def test_grid_arm_limit_counts_cartesian_product(document: dict[str, Any]) -> None:
    document["grid"] = {"period": list(range(20)), "weight": list(range(10))}
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert_code(exc, "STUDY_TOO_MANY_ARMS")


def test_arm_limit_counts_strategy_baselines(document: dict[str, Any]) -> None:
    document["grid"] = {"period": list(range(199))}
    document["baselines"].append({"id": "other-strategy", "kind": "strategy", "strategy": "sma-cross"})
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert_code(exc, "STUDY_TOO_MANY_ARMS")


def test_declared_variants_cannot_exceed_arm_limit(document: dict[str, Any]) -> None:
    document["variants"] = [{"id": f"p{p}", "params": {"period": p}} for p in range(200)]
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert_code(exc, "STUDY_TOO_MANY_ARMS")


def test_declared_variants_allow_200_runnable_arms(document: dict[str, Any]) -> None:
    document["variants"] = [{"id": f"p{p}", "params": {"period": p}} for p in range(199)]
    study = StudySpecV1.model_validate(document)
    assert study.arm_count() == 200
    assert len(study.variants) == 199


def test_variants_field_rejects_201_entries(document: dict[str, Any]) -> None:
    document["variants"] = [{"id": f"p{p}", "params": {"period": p}} for p in range(201)]
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [(("variants",), "too_long")]


def test_baselines_allow_ten(document: dict[str, Any]) -> None:
    document["baselines"].extend({"id": f"benchmark-{p}", "kind": "benchmark"} for p in range(9))
    assert len(StudySpecV1.model_validate(document).baselines) == 10


def test_baselines_reject_eleven(document: dict[str, Any]) -> None:
    document["baselines"].extend({"id": f"benchmark-{p}", "kind": "benchmark"} for p in range(10))
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [(("baselines",), "too_long")]


def test_baselines_cannot_be_empty(document: dict[str, Any]) -> None:
    document["baselines"] = []
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [(("baselines",), "too_short")]


@pytest.mark.parametrize("length", [1, 128])
def test_dataset_accepts_length_bounds(document: dict[str, Any], length: int) -> None:
    document["dataset"] = "x" * length
    assert StudySpecV1.model_validate(document).dataset == "x" * length


@pytest.mark.parametrize(("length", "error"), [(0, "string_too_short"), (129, "string_too_long")])
def test_dataset_rejects_invalid_lengths(document: dict[str, Any], length: int, error: str) -> None:
    document["dataset"] = "x" * length
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [(("dataset",), error)]


@pytest.mark.parametrize("level", ["study", "hypothesis", "window", "baseline", "variant", "compare"])
def test_unknown_keys_are_rejected_at_every_model_level(document: dict[str, Any], level: str) -> None:
    document["window"] = {}
    document["variants"] = [{"id": "p150", "params": {"period": 150}}]
    document["baselines"].append({"id": "vol-matched", "kind": "benchmark_scaled"})
    nodes = {
        "study": (document, ()),
        "hypothesis": (document["hypothesis"], ("hypothesis",)),
        "window": (document["window"], ("window",)),
        "baseline": (document["baselines"][0], ("baselines", 0)),
        "variant": (document["variants"][0], ("variants", 0)),
        "compare": (document["compare"], ("compare",)),
    }
    node, prefix = nodes[level]
    node["unknown"] = True
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [(prefix + ("unknown",), "extra_forbidden")]


@pytest.mark.parametrize("level", ["study", "hypothesis", "window", "baseline", "variant", "compare"])
def test_models_are_frozen(document: dict[str, Any], level: str) -> None:
    document["variants"] = [{"id": "p150", "params": {"period": 150}}]
    study = StudySpecV1.model_validate(document)
    nodes = {
        "study": (study, "id", "new-study"),
        "hypothesis": (study.hypothesis, "statement", "A different hypothesis statement."),
        "window": (study.window, "start", date(2024, 1, 1)),
        "baseline": (study.baselines[0], "id", "new-baseline"),
        "variant": (study.variants[0], "id", "new-variant"),
        "compare": (study.compare, "metric", "cagr"),
    }
    node, field, value = nodes[level]
    with pytest.raises(ValidationError) as exc:
        setattr(node, field, value)
    assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [((field,), "frozen_instance")]


@pytest.mark.parametrize("field", ["id", "base"])
@pytest.mark.parametrize("slug", ["Upper-Case", "has_underscore", "a", "a" * 65])
def test_study_slug_fields_reject_invalid_values(document: dict[str, Any], field: str, slug: str) -> None:
    document[field] = slug
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [((field,), "string_pattern_mismatch")]


@pytest.mark.parametrize(
    ("model", "data", "field"),
    [
        (Baseline, {"id": "bad_slug", "kind": "benchmark"}, "id"),
        (Baseline, {"id": "other-strategy", "kind": "strategy", "strategy": "bad_slug"}, "strategy"),
        (Baseline, {"id": "buy-and-hold", "kind": "benchmark", "symbol": "spy"}, "symbol"),
        (Variant, {"id": "bad_slug", "params": {"period": 150}}, "id"),
        (CompareRule, {"metric": "cagr", "direction": "higher", "versus": "bad_slug"}, "versus"),
    ],
)
def test_nested_pattern_fields_are_validated(model: Any, data: dict[str, Any], field: str) -> None:
    with pytest.raises(ValidationError) as exc:
        model.model_validate(data)
    assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [((field,), "string_pattern_mismatch")]


@pytest.mark.parametrize(
    ("model", "data", "field"),
    [
        (Baseline, {"id": "buy-and-hold", "kind": "unknown"}, "kind"),
        (Variant, {"id": "p150", "params": {"period": 150}, "role": "unknown"}, "role"),
        (CompareRule, {"metric": "unknown", "direction": "higher", "versus": "buy-and-hold"}, "metric"),
        (CompareRule, {"metric": "cagr", "direction": "unknown", "versus": "buy-and-hold"}, "direction"),
    ],
)
def test_literal_fields_reject_unknown_values(model: Any, data: dict[str, Any], field: str) -> None:
    with pytest.raises(ValidationError) as exc:
        model.model_validate(data)
    assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [((field,), "literal_error")]


def test_schema_rejects_other_versions(document: dict[str, Any]) -> None:
    document["schema"] = "signalquarry.study/v2"
    with pytest.raises(ValidationError) as exc:
        StudySpecV1.model_validate(document)
    assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [(("schema",), "literal_error")]


def test_study_document_is_deterministic_for_equal_studies(document: dict[str, Any]) -> None:
    first = StudySpecV1.model_validate(document)
    second = StudySpecV1.model_validate(dict(reversed(list(document.items()))))
    assert first == second
    assert json.dumps(first.study_document()) == json.dumps(second.study_document())
    assert first.study_document() == first.model_dump(mode="json", by_alias=True)


def test_load_study_reads_temporary_yaml(tmp_path: Path) -> None:
    path = tmp_path / "study.yaml"
    path.write_text(EXAMPLE, encoding="utf-8")
    assert load_study(path) == StudySpecV1.model_validate(yaml.safe_load(EXAMPLE))


@pytest.mark.parametrize("content", ["", "null\n", "42\n", "a scalar\n", "- a-list\n"])
def test_load_study_rejects_non_mapping(tmp_path: Path, content: str) -> None:
    path = tmp_path / "study.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="^STUDY_NOT_A_MAPPING$"):
        load_study(path)


def test_load_study_preserves_validation_errors(tmp_path: Path, document: dict[str, Any]) -> None:
    document["compare"]["versus"] = "missing-baseline"
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValidationError) as exc:
        load_study(path)
    assert_code(exc, "STUDY_COMPARE_VERSUS_UNKNOWN")

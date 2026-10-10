# SPDX-License-Identifier: Apache-2.0
"""Strict, declarative ``study.yaml`` (schema ``signalquarry.study/v1``)."""

from __future__ import annotations

import re
from datetime import date
from itertools import product
from math import prod
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ConfigDict, Field, field_validator, model_validator

from signalquarry._internal.contracts.spec import SLUG_PATTERN, SYMBOL_PATTERN, Hypothesis, _Strict

MAX_BASELINES = 10
MAX_VARIANTS = 200
MAX_ARMS = 200


class StudyWindow(_Strict):
    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def _ordered(self) -> StudyWindow:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("STUDY_WINDOW_INVALID")
        return self


class Baseline(_Strict):
    id: str = Field(pattern=SLUG_PATTERN)
    kind: Literal["benchmark", "benchmark_scaled", "strategy"]
    strategy: str | None = Field(default=None, pattern=SLUG_PATTERN)
    symbol: str | None = Field(default=None, pattern=SYMBOL_PATTERN)

    @model_validator(mode="after")
    def _kind_consistent(self) -> Baseline:
        if (self.kind == "strategy") != (self.strategy is not None):
            raise ValueError("STUDY_BASELINE_STRATEGY_INVALID")
        if self.kind == "strategy" and self.symbol is not None:
            raise ValueError("STUDY_BASELINE_SYMBOL_INVALID")
        return self


class Variant(_Strict):
    id: str = Field(pattern=SLUG_PATTERN)
    params: dict[str, Any] = Field(default_factory=dict)
    execution: dict[str, Any] = Field(default_factory=dict)
    role: Literal["candidate", "ablation", "sensitivity"] = "candidate"
    note: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _changes(self) -> Variant:
        if not self.params and not self.execution:
            raise ValueError("STUDY_VARIANT_EMPTY")
        if self.role == "sensitivity" and self.params:
            raise ValueError("STUDY_SENSITIVITY_CHANGES_PARAMS")
        return self


class CompareRule(_Strict):
    metric: Literal[
        "total_return", "cagr", "sharpe", "sortino", "calmar", "max_drawdown", "annual_volatility"
    ]
    direction: Literal["higher", "lower"]
    versus: str = Field(pattern=SLUG_PATTERN)


class StudySpecV1(_Strict):
    schema_: Literal["signalquarry.study/v1"] = Field(default="signalquarry.study/v1", alias="schema")
    id: str = Field(pattern=SLUG_PATTERN)
    hypothesis: Hypothesis
    base: str = Field(pattern=SLUG_PATTERN)
    dataset: str | None = Field(default=None, min_length=1, max_length=128)
    window: StudyWindow = StudyWindow()
    baselines: tuple[Baseline, ...] = Field(min_length=1, max_length=MAX_BASELINES)
    variants: tuple[Variant, ...] = Field(default=(), max_length=MAX_VARIANTS)
    grid: dict[str, tuple[Any, ...]] = Field(default_factory=dict)
    compare: CompareRule

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @field_validator("grid", mode="before")
    @classmethod
    def _grid_axes(cls, value: Any) -> Any:
        if isinstance(value, dict):
            for axis, values in value.items():
                if (
                    not isinstance(axis, str)
                    or not axis.strip()
                    or not isinstance(values, (list, tuple))
                    or not values
                    or any(type(item) not in (str, int, float, bool) for item in values)
                    or len(set(values)) != len(values)
                ):
                    raise ValueError(f"STUDY_GRID_INVALID:{axis}")
        return value

    @model_validator(mode="after")
    def _arms(self) -> StudySpecV1:
        ids = {"base"}
        for arm in (*self.baselines, *self.variants):
            if arm.id in ids:
                raise ValueError(f"STUDY_ARM_ID_DUPLICATE:{arm.id}")
            ids.add(arm.id)
        if self.compare.versus not in {baseline.id for baseline in self.baselines}:
            raise ValueError("STUDY_COMPARE_VERSUS_UNKNOWN")
        # Bound expansion before constructing the cartesian product.
        if self.arm_count() > MAX_ARMS:
            raise ValueError("STUDY_TOO_MANY_ARMS")
        for variant in self.grid_variants():
            if variant.id in ids:
                raise ValueError(f"STUDY_GRID_ID_INVALID:{variant.id}")
            ids.add(variant.id)
        return self

    def grid_variants(self) -> tuple[Variant, ...]:
        """Expand axes in declaration order, with the first axis changing slowest."""
        if not self.grid:
            return ()
        variants = []
        for values in product(*self.grid.values()):
            point = dict(zip(self.grid, values, strict=True))
            raw_id = "grid-" + "-".join(f"{axis}-{value}" for axis, value in point.items())
            id_ = re.sub(r"[^a-z0-9]+", "-", raw_id.lower()).strip("-")
            if not re.fullmatch(SLUG_PATTERN, id_):
                raise ValueError(f"STUDY_GRID_ID_INVALID:{id_}")
            variants.append(Variant(id=id_, params=point))
        return tuple(variants)

    def all_variants(self) -> tuple[Variant, ...]:
        """Declared variants followed by the generated grid variants."""
        return self.variants + self.grid_variants()

    def arm_count(self) -> int:
        """Runnable strategies, including the subject and strategy baselines."""
        grid_count = prod(len(values) for values in self.grid.values()) if self.grid else 0
        return 1 + len(self.variants) + grid_count + sum(b.kind == "strategy" for b in self.baselines)

    def study_document(self) -> dict[str, Any]:
        """JSON-serializable declaration used for the study's configuration identity."""
        return self.model_dump(mode="json", by_alias=True)


def load_study(path: Path) -> StudySpecV1:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("STUDY_NOT_A_MAPPING")
    return StudySpecV1.model_validate(document)

# SPDX-License-Identifier: Apache-2.0
"""Strict, declarative ``factor.yaml`` metadata for project-registered factors."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from signalquarry._internal.contracts.spec import SLUG_PATTERN, HoldoutSpec, Hypothesis


class FactorEvaluationSpecV1(BaseModel):
    """Declared choices that affect factor diagnostics and future trial accounting."""

    horizons: tuple[int, ...] = Field(default=(1, 5, 21), strict=True)
    chronological_blocks: int = Field(default=6, ge=1, le=100, strict=True)
    cost_bps: Decimal = Field(default=Decimal("10"), ge=0, lt=10_000)
    capital: Decimal = Field(default=Decimal("1000000"), gt=0)
    trial_budget: int = Field(default=50, ge=1, le=10_000, strict=True)
    holdout: HoldoutSpec = HoldoutSpec()

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("horizons", mode="before")
    @classmethod
    def _yaml_horizons(cls, value: Any) -> Any:
        normalized = tuple(value) if isinstance(value, list) else value
        if isinstance(normalized, tuple) and any(type(item) is not int for item in normalized):
            raise ValueError("FACTOR_HORIZONS_INVALID")
        return normalized

    @field_validator("horizons")
    @classmethod
    def _horizons(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(type(item) is not int or item < 1 or item > 252 for item in value):
            raise ValueError("FACTOR_HORIZONS_INVALID")
        if value != tuple(sorted(set(value))):
            raise ValueError("FACTOR_HORIZONS_INVALID")
        return value


class FactorSpecV1(BaseModel):
    schema_: Literal["signalquarry.factor/v1"] = Field(default="signalquarry.factor/v1", alias="schema")
    id: str = Field(pattern=SLUG_PATTERN)
    family: str = Field(pattern=SLUG_PATTERN)
    version: str = Field(min_length=1, max_length=32)
    hypothesis: Hypothesis
    params: dict[str, Any] = Field(default_factory=dict)
    evaluation: FactorEvaluationSpecV1 = FactorEvaluationSpecV1()

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


def load_factor_spec(path: Path) -> FactorSpecV1:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError("Invalid factor YAML") from exc
    if not isinstance(document, dict):
        raise ValueError("FACTOR_SPEC_NOT_A_MAPPING")
    return FactorSpecV1.model_validate(document)

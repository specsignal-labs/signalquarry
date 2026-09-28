# SPDX-License-Identifier: Apache-2.0
"""Strict, declarative ``factor.yaml`` metadata for project-registered factors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from signalquarry._internal.contracts.spec import SLUG_PATTERN, Hypothesis


class FactorSpecV1(BaseModel):
    schema_: Literal["signalquarry.factor/v1"] = Field(default="signalquarry.factor/v1", alias="schema")
    id: str = Field(pattern=SLUG_PATTERN)
    family: str = Field(pattern=SLUG_PATTERN)
    version: str = Field(min_length=1, max_length=32)
    hypothesis: Hypothesis
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


def load_factor_spec(path: Path) -> FactorSpecV1:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError("Invalid factor YAML") from exc
    if not isinstance(document, dict):
        raise ValueError("FACTOR_SPEC_NOT_A_MAPPING")
    return FactorSpecV1.model_validate(document)

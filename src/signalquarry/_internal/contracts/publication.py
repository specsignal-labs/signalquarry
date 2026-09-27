# SPDX-License-Identifier: Apache-2.0
"""``publication/<family>.publication.yaml`` (schema ``signalquarry.publication/v1``).

What one strategy family may publish. Default-deny: a family is commercial, published
at the ``category`` tier, with no backtest results and no forward records, unless the
file says otherwise. Commercial families are always ``category`` tier and never
publish backtest results; their forward records are weekly or monthly returns with a
lag of at least 14 days, and positions, fills and exposure are always withheld.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from signalquarry._internal.contracts.paper import ALIAS_PATTERN
from signalquarry._internal.contracts.spec import SLUG_PATTERN


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ForwardPolicy(_Strict):
    resolution: Literal["weekly", "monthly"] = "weekly"
    lag_days: int = Field(default=14, ge=14, le=365)


class RulesText(_Strict):
    how: str = Field(min_length=30)
    thesis: str = Field(min_length=30)
    entry: str = Field(min_length=12)
    exit: str = Field(min_length=12)
    risks: tuple[str, ...] = Field(min_length=1)
    leverage: str = Field(min_length=2)
    execution: str = Field(min_length=8)
    costs: str = Field(min_length=8)


class PublishedStrategy(_Strict):
    id: str = Field(pattern=SLUG_PATTERN)
    title: str = Field(min_length=5, max_length=160)
    summary: str = Field(min_length=30, max_length=1200)
    asset_class: Literal["equity", "options"] = "equity"  # ignored: the export derives it from the spec
    assets: str = Field(min_length=2, max_length=200)
    horizon: str = Field(default="Daily", min_length=2, max_length=40)
    deployment: str | None = Field(default=None, pattern=ALIAS_PATTERN)
    capacity_note: str | None = Field(default=None, max_length=600)
    rules: RulesText | None = None


class PublicationV1(_Strict):
    schema_: Literal["signalquarry.publication/v1"] = Field(alias="schema")
    family: str = Field(pattern=SLUG_PATTERN)
    family_label: str = Field(min_length=3, max_length=80)
    commercial: bool = True
    tier: Literal["category", "results", "rules"] = "category"
    backtest_results: Literal["deny", "allow"] = "deny"
    forward: ForwardPolicy | None = None
    # A live, sanitized paper-account feed (sqy perf publish); never for commercial families.
    live_feed: Literal["deny", "allow"] = "deny"
    strategies: tuple[PublishedStrategy, ...] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @model_validator(mode="after")
    def _caps(self) -> PublicationV1:
        if self.commercial and self.tier != "category":
            raise ValueError("PUBLICATION_COMMERCIAL_REQUIRES_CATEGORY_TIER")
        if self.tier == "category" and self.backtest_results != "deny":
            raise ValueError("PUBLICATION_CATEGORY_TIER_DENIES_BACKTESTS")
        if self.commercial and self.live_feed == "allow":
            raise ValueError("PUBLICATION_LIVE_FEED_NOT_FOR_COMMERCIAL_FAMILIES")
        if not self.commercial and self.forward is not None:
            raise ValueError("PUBLICATION_FORWARD_RECORDS_ARE_FOR_COMMERCIAL_FAMILIES")
        for strategy in self.strategies:
            if strategy.rules is not None and self.tier != "rules":
                raise ValueError(f"PUBLICATION_RULES_ABOVE_TIER:{strategy.id}")
        return self


def publication_path(root: Path, family: str) -> Path | None:
    for candidate in (
        root / "publication" / f"{family}.publication.yaml",
        root / "families" / family / "publication.yaml",
    ):
        if candidate.is_file():
            return candidate
    return None


def load_publication(path: Path) -> PublicationV1:
    return PublicationV1.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

# SPDX-License-Identifier: Apache-2.0
"""``strategy.yaml`` (schema ``signalquarry.strategy/v1``): everything that affects results."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SYMBOL_PATTERN = r"^[A-Z][A-Z0-9.]{0,9}$"
SLUG_PATTERN = r"^[a-z0-9][a-z0-9-]{1,63}$"
CODE_PATTERN = r"^[A-Z][A-Z0-9_]{1,63}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Hypothesis(_Strict):
    statement: str = Field(min_length=10, max_length=2000)
    falsification: str = Field(min_length=10, max_length=2000)


class DataSpec(_Strict):
    symbols: tuple[str, ...] = Field(min_length=1, max_length=500)
    fields: tuple[Literal["open", "high", "low", "close", "volume"], ...] = ("close",)
    feed: Literal["sip", "iex", "synthetic"] = "sip"
    price_basis: Literal["pit_split"] = "pit_split"
    max_staleness_sessions: int = Field(default=3, ge=0, le=30)

    @field_validator("symbols")
    @classmethod
    def _symbols(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        import re

        upper = tuple(item.upper() for item in value)
        if len(set(upper)) != len(upper) or any(not re.fullmatch(SYMBOL_PATTERN, item) for item in upper):
            raise ValueError("SPEC_SYMBOLS_INVALID")
        return upper


class AccountSpec(_Strict):
    model: Literal["cash", "margin"] = "cash"
    initial_cash: Decimal = Field(default=Decimal("100000"), gt=0)


class CostSpec(_Strict):
    bps: Decimal = Field(default=Decimal("5"), ge=0, le=500)
    per_share: Decimal = Field(default=Decimal("0"), ge=0, le=1)
    # Sell-side regulatory fees (SEC Section 31 style on notional, FINRA TAF style per share
    # with a per-order cap). Rates change over time: set the ones in force for your study.
    sell_bps: Decimal = Field(default=Decimal("0"), ge=0, le=10)
    sell_per_share: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("0.01"))
    sell_per_order_max: Decimal | None = Field(default=None, gt=0, le=100)

    def sell_fees(self, notional: Decimal, quantity: Decimal) -> Decimal:
        """Regulatory fees on one sell order (unrounded; the caller rounds the total fee)."""
        per_share = quantity * self.sell_per_share
        if self.sell_per_order_max is not None:
            per_share = min(per_share, self.sell_per_order_max)
        return notional * self.sell_bps / Decimal(10000) + per_share


class FillSpec(_Strict):
    """``volume_cap``: fill at most this fraction of the fill session's volume per symbol."""

    model: Literal["volume_cap"] = "volume_cap"
    max_volume_fraction: Decimal = Field(default=Decimal("0.01"), gt=0, le=1)


class ExecutionSpec(_Strict):
    model: Literal["next_open"] = "next_open"
    sizing: Literal["whole_shares", "fractional"] = "whole_shares"
    rebalance: Literal["on_change", "every_decision"] = "on_change"
    costs: CostSpec = CostSpec()
    min_order_notional: Decimal = Field(default=Decimal("1"), ge=0)
    execution_delay_sessions: int = Field(default=0, ge=0, le=5)
    fill: FillSpec | None = None  # None: the fixed-cost model (fills the whole order at the open)


class LimitsSpec(_Strict):
    long_only: Literal[True] = True
    max_weight_per_symbol: Decimal = Field(default=Decimal("1"), gt=0, le=1)


class HoldoutSpec(_Strict):
    months: int = Field(default=12, ge=0, le=60)
    # A model or dataset the strategy depends on (for example an LLM behind text features)
    # may have seen everything up to its training cutoff; the holdout then starts the day
    # after it whenever that is earlier than the last ``months``.
    training_cutoff: date | None = None


class WalkForwardSpec(_Strict):
    train_months: int = Field(default=36, ge=6, le=240)
    test_months: int = Field(default=6, ge=1, le=60)


class EvaluationSpec(_Strict):
    holdout: HoldoutSpec = HoldoutSpec()
    walk_forward: WalkForwardSpec = WalkForwardSpec()
    trial_budget: int = Field(default=50, ge=1, le=10000)


class OptionsSpec(_Strict):
    """Single-leg options settings (``kind: options_single_leg``)."""

    underlyings: tuple[str, ...] = Field(min_length=1, max_length=5)
    contracts_per_underlying: Literal[1] = 1
    min_bid: Decimal = Field(default=Decimal("0.05"), gt=0)
    max_relative_spread: Decimal = Field(default=Decimal("0.25"), gt=0, le=1)
    max_quote_age_seconds: int = Field(default=15, ge=1, le=120)
    tick: Decimal = Field(default=Decimal("0.01"), gt=0)
    per_contract_fee: Decimal = Field(default=Decimal("0"), ge=0, le=10)
    # Simulation fills cross this fraction of the half-spread against us (required, > 0).
    spread_haircut: Decimal = Field(default=Decimal("0.5"), gt=0, le=1)
    entry_cutoff: str = Field(default="15:15", pattern=r"^(0[9]|1[0-5]):[0-5][0-9]$")
    poll_seconds: int = Field(default=60, ge=30, le=300)
    cancel_after_seconds: int = Field(default=180, ge=60, le=900)

    @field_validator("underlyings")
    @classmethod
    def _underlyings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        upper = tuple(item.upper() for item in value)
        if len(set(upper)) != len(upper) or any(not item.isalpha() or len(item) > 6 for item in upper):
            raise ValueError("SPEC_OPTIONS_UNDERLYINGS_INVALID")
        return upper


class StrategySpecV1(_Strict):
    schema_: Literal["signalquarry.strategy/v1"] = Field(default="signalquarry.strategy/v1", alias="schema")
    id: str = Field(pattern=SLUG_PATTERN)
    family: str = Field(pattern=SLUG_PATTERN)
    version: str = Field(min_length=1, max_length=32)
    kind: Literal["equity_daily", "options_single_leg"] = "equity_daily"
    hypothesis: Hypothesis
    data: DataSpec
    account: AccountSpec = AccountSpec()
    execution: ExecutionSpec = ExecutionSpec()
    params: dict[str, Any] = Field(default_factory=dict)
    reason_codes: dict[str, str] = Field(default_factory=dict)
    limits: LimitsSpec = LimitsSpec()
    evaluation: EvaluationSpec = EvaluationSpec()
    benchmark: str | None = Field(default=None, pattern=SYMBOL_PATTERN)
    options: OptionsSpec | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @model_validator(mode="after")
    def _kind_consistent(self) -> StrategySpecV1:
        if (self.kind == "options_single_leg") != (self.options is not None):
            raise ValueError("SPEC_OPTIONS_SECTION_REQUIRED_FOR_OPTIONS_KIND_ONLY")
        if self.options is not None and set(self.options.underlyings) - set(self.data.symbols):
            raise ValueError("SPEC_OPTIONS_UNDERLYING_NOT_IN_DATA_SYMBOLS")
        return self

    @field_validator("reason_codes")
    @classmethod
    def _codes(cls, value: dict[str, str]) -> dict[str, str]:
        import re

        for code, text in value.items():
            if not re.fullmatch(CODE_PATTERN, code) or not text.strip():
                raise ValueError(f"SPEC_REASON_CODE_INVALID:{code}")
        return value

    def outcome_document(self) -> dict[str, Any]:
        """The fields that determine results (hashed into the configuration identity)."""
        body = self.model_dump(mode="json", by_alias=True)
        body.pop("hypothesis")
        if body.get("options") is None:
            body.pop("options", None)  # equity specs keep the identity they had before options existed
        if body["execution"].get("fill") is None:
            body["execution"].pop("fill", None)  # likewise for specs without a fill model
        if body["evaluation"]["holdout"].get("training_cutoff") is None:
            body["evaluation"]["holdout"].pop("training_cutoff", None)
        costs = body["execution"]["costs"]
        for key, default in (("sell_bps", "0"), ("sell_per_share", "0"), ("sell_per_order_max", None)):
            if costs.get(key) == default:
                costs.pop(key, None)  # and without sell-side fees
        return body


def load_spec(path: Path) -> StrategySpecV1:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("SPEC_NOT_A_MAPPING")
    return StrategySpecV1.model_validate(document)

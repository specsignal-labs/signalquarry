# SPDX-License-Identifier: Apache-2.0
"""``paper/<alias>.paper.yaml`` (schema ``signalquarry.paper/v1``): one paper deployment."""

from __future__ import annotations

from datetime import time
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from signalquarry._internal.contracts.spec import SLUG_PATTERN

ALIAS_PATTERN = r"^[a-z0-9][a-z0-9-]{1,31}$"
SHA256_HEX = r"^[0-9a-f]{64}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PaperWindow(_Strict):
    """Pre-open submission window in exchange time (America/New_York)."""

    start: time = time(9, 0)
    end: time = time(9, 25)

    @model_validator(mode="after")
    def _order(self) -> PaperWindow:
        if not (time(4, 0) <= self.start < self.end <= time(9, 28)):
            raise ValueError("PAPER_WINDOW_INVALID")
        return self


class PaperGuards(_Strict):
    max_daily_drawdown: Decimal = Field(default=Decimal("0.10"), gt=0, le=1)
    max_clock_skew_seconds: int = Field(default=30, ge=1, le=300)
    buy_price_buffer_bps: Decimal = Field(default=Decimal("50"), ge=0, le=1000)
    # G5 (forward): mean adverse fill slippage versus the session open must stay within this.
    max_mean_slippage_bps: Decimal = Field(default=Decimal("25"), ge=0, le=1000)


class PaperDeploymentV1(_Strict):
    schema_: Literal["signalquarry.paper/v1"] = Field(alias="schema")
    alias: str = Field(pattern=ALIAS_PATTERN)
    strategy: str = Field(pattern=SLUG_PATTERN)
    # "simulated" replays the project dataset offline: dry-run/preflight/status only, never armed.
    # "plugin:<name>": a paper broker from the ``signalquarry.brokers`` entry point (paper_only).
    broker: str = Field(
        default="alpaca-paper", pattern=r"^(alpaca-paper|simulated|plugin:[a-z][a-z0-9-]{0,39})$"
    )
    submission: Literal["disabled", "enabled"] = "disabled"
    credentials_profile: str | None = Field(default=None, pattern=r"^paper\.[a-z0-9][a-z0-9-]{1,31}$")
    expected_account_id_sha256: str | None = Field(default=None, pattern=SHA256_HEX)
    denied_key_id_sha256: tuple[str, ...] = ()
    arm_days: int = Field(default=30, ge=1, le=90)
    # "process": decide runs in a child process without credentials (not a sandbox).
    isolation: Literal["process", "none"] = "process"
    window: PaperWindow = PaperWindow()
    guards: PaperGuards = PaperGuards()

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @property
    def profile(self) -> str:
        return self.credentials_profile or f"paper.{self.alias}"


def load_paper_config(path: Path) -> PaperDeploymentV1:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = PaperDeploymentV1.model_validate(body)
    if path.name != f"{config.alias}.paper.yaml":
        raise ValueError("PAPER_CONFIG_NAME_MISMATCH")
    return config

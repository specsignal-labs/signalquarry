# SPDX-License-Identifier: Apache-2.0
"""Options authoring API (single-leg, 1.0).

An options strategy returns *selectors*, never contract symbols. The engine owns the
wheel state machine, resolves selectors to contracts with one shared resolver, prices
orders and manages the open leg::

    from signalquarry.sdk.options import OD, OptionsCtx, Strike, options_strategy, sell_put

    @options_strategy(params=P, lookback=lambda p: p.trend_sessions)
    def decide(ctx: OptionsCtx, p: P) -> OptionsDecision:
        wheel = ctx.wheel("QQQ")
        if wheel.state == "flat":
            return OD.open(sell_put("QQQ", dte=(7, 14), strike=Strike.otm("0.01")), "SELL_PUT")
        leg = ctx.leg("QQQ")
        if leg is not None and leg.captured_fraction is not None and leg.captured_fraction > p.take_profit:
            return OD.close("QQQ", "TAKE_PROFIT")
        return OD.hold("HOLD")
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal, cast

from signalquarry.sdk.context import Bars
from signalquarry.sdk.decision import checked_codes
from signalquarry.sdk.strategy import ATTRIBUTE, Params, StrategyDef


@dataclass(frozen=True)
class StrikeRule:
    kind: Literal["otm"]
    value: Decimal


class Strike:
    """Strike rules. ``otm(0.03)``: 3% out of the money from the underlying's price."""

    @staticmethod
    def otm(fraction: Decimal | float | str) -> StrikeRule:
        value = Decimal(str(fraction))
        if not Decimal(0) <= value < Decimal("0.5"):
            raise ValueError("OPTIONS_STRIKE_OTM_OUT_OF_RANGE")
        return StrikeRule("otm", value)


@dataclass(frozen=True)
class OptionSelector:
    underlying: str
    right: Literal["PUT", "CALL"]
    min_dte: int
    max_dte: int
    strike: StrikeRule


def _selector(
    right: Literal["PUT", "CALL"], underlying: str, dte: tuple[int, int], strike: StrikeRule
) -> OptionSelector:
    low, high = dte
    if not 0 <= low <= high <= 120:
        raise ValueError("OPTIONS_DTE_WINDOW_INVALID")
    return OptionSelector(underlying.upper(), right, int(low), int(high), strike)


def sell_put(underlying: str, *, dte: tuple[int, int], strike: StrikeRule) -> OptionSelector:
    """Sell one cash-secured put (only when the wheel is flat)."""
    return _selector("PUT", underlying, dte, strike)


def sell_call(underlying: str, *, dte: tuple[int, int], strike: StrikeRule) -> OptionSelector:
    """Sell one covered call against 100 shares (only when the wheel holds shares)."""
    return _selector("CALL", underlying, dte, strike)


@dataclass(frozen=True)
class OptionsDecision:
    action: Literal["open", "close", "hold", "unavailable"]
    reason_codes: tuple[str, ...]
    selector: OptionSelector | None = None
    underlying: str | None = None
    state: dict[str, Any] | None = None


class OD:
    """Build an :class:`OptionsDecision`."""

    @staticmethod
    def open(
        selector: OptionSelector, *reason_codes: str, state: dict[str, Any] | None = None
    ) -> OptionsDecision:
        return OptionsDecision(
            "open",
            checked_codes(reason_codes),
            selector=selector,
            underlying=selector.underlying,
            state=state,
        )

    @staticmethod
    def close(underlying: str, *reason_codes: str, state: dict[str, Any] | None = None) -> OptionsDecision:
        """Buy back the open leg on ``underlying``."""
        return OptionsDecision(
            "close", checked_codes(reason_codes), underlying=underlying.upper(), state=state
        )

    @staticmethod
    def hold(*reason_codes: str, state: dict[str, Any] | None = None) -> OptionsDecision:
        return OptionsDecision("hold", checked_codes(reason_codes), state=state)

    @staticmethod
    def unavailable(*reason_codes: str) -> OptionsDecision:
        return OptionsDecision("unavailable", checked_codes(reason_codes))


@dataclass(frozen=True)
class LegView:
    symbol: str
    right: Literal["PUT", "CALL"]
    strike: Decimal
    expiration: date
    dte: int
    entry_credit: Decimal  # per share
    close_cost: Decimal | None  # per share, what buying back costs now (ask), if quoted
    captured_fraction: Decimal | None  # (entry_credit - close_cost) / entry_credit


@dataclass(frozen=True)
class WheelView:
    underlying: str
    state: Literal["flat", "short_put", "long_shares", "covered_call"]
    shares: int
    share_cost_basis: Decimal | None


@dataclass(frozen=True)
class OptionsCtx:
    """What an options strategy sees. Bars end at the last completed session."""

    decision_time: datetime
    _bars: Mapping[str, Bars]
    _wheels: Mapping[str, WheelView]
    _legs: Mapping[str, LegView]
    spots: Mapping[str, Decimal]
    cash: Decimal
    equity: Decimal
    state: Mapping[str, Any]

    def bars(self, symbol: str) -> Bars:
        try:
            return self._bars[symbol.upper()]
        except KeyError as exc:
            raise KeyError(f"CTX_SYMBOL_NOT_DECLARED:{symbol}") from exc

    def wheel(self, underlying: str) -> WheelView:
        return self._wheels[underlying.upper()]

    def leg(self, underlying: str) -> LegView | None:
        return self._legs.get(underlying.upper())

    def spot(self, underlying: str) -> Decimal:
        return self.spots[underlying.upper()]


def options_strategy[P: Params](
    *, params: type[P], lookback: Callable[[P], int]
) -> Callable[[Callable[[OptionsCtx, P], OptionsDecision]], Callable[[OptionsCtx, P], OptionsDecision]]:
    """Declare a pure options decision function ``decide(ctx, params) -> OptionsDecision``."""
    if not (isinstance(params, type) and issubclass(params, Params)):  # pyright: ignore[reportUnnecessaryIsInstance] -- untyped callers
        raise TypeError("STRATEGY_PARAMS_MUST_SUBCLASS_PARAMS")

    def wrap(
        function: Callable[[OptionsCtx, P], OptionsDecision],
    ) -> Callable[[OptionsCtx, P], OptionsDecision]:
        # StrategyDef.kind tells the engine to call ``decide`` with an OptionsCtx and ``params``.
        definition = StrategyDef(
            cast(Any, function),
            params,
            cast(Callable[[Params], int], lookback),
            function.__module__,
            getattr(function, "__name__", "decide"),
            kind="options",
        )
        setattr(function, ATTRIBUTE, definition)
        return function

    return wrap


__all__ = [
    "OD",
    "LegView",
    "OptionSelector",
    "OptionsCtx",
    "OptionsDecision",
    "Strike",
    "StrikeRule",
    "WheelView",
    "options_strategy",
    "sell_call",
    "sell_put",
]

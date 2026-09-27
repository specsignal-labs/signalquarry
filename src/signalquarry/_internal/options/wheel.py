# SPDX-License-Identifier: Apache-2.0
"""The wheel state machine. The engine owns it; strategies only read it.

    FLAT --put_opened--> SHORT_PUT --put_closed | put_expired--> FLAT
                         SHORT_PUT --put_assigned--> LONG_SHARES
    LONG_SHARES --call_opened--> COVERED_CALL --call_closed | call_expired--> LONG_SHARES
                                 COVERED_CALL --call_assigned--> FLAT

Any other (state, event) pair is illegal and stops the run: a wheel that disagrees with
the broker must never keep trading.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from enum import StrEnum

from signalquarry._internal.options.contracts import OptionContract, OptionsError


class WheelState(StrEnum):
    FLAT = "flat"
    SHORT_PUT = "short_put"
    LONG_SHARES = "long_shares"
    COVERED_CALL = "covered_call"


TRANSITIONS: dict[tuple[WheelState, str], WheelState] = {
    (WheelState.FLAT, "put_opened"): WheelState.SHORT_PUT,
    (WheelState.SHORT_PUT, "put_closed"): WheelState.FLAT,
    (WheelState.SHORT_PUT, "put_expired"): WheelState.FLAT,
    (WheelState.SHORT_PUT, "put_assigned"): WheelState.LONG_SHARES,
    (WheelState.LONG_SHARES, "call_opened"): WheelState.COVERED_CALL,
    (WheelState.COVERED_CALL, "call_closed"): WheelState.LONG_SHARES,
    (WheelState.COVERED_CALL, "call_expired"): WheelState.LONG_SHARES,
    (WheelState.COVERED_CALL, "call_assigned"): WheelState.FLAT,
}
EVENTS = frozenset(event for _, event in TRANSITIONS)


@dataclass(frozen=True)
class Leg:
    contract: OptionContract
    entry_credit: Decimal  # per share (premium received)
    opened: date
    client_order_id: str


@dataclass(frozen=True)
class Wheel:
    underlying: str
    state: WheelState = WheelState.FLAT
    leg: Leg | None = None
    shares: int = 0
    share_cost_basis: Decimal | None = None  # per share, the put strike on assignment


def transition(wheel: Wheel, event: str, *, leg: Leg | None = None) -> Wheel:
    target = TRANSITIONS.get((wheel.state, event))
    if target is None:
        raise OptionsError("WHEEL_TRANSITION_ILLEGAL", f"{wheel.underlying}: {wheel.state.value} + {event}")
    if event.endswith("_opened"):
        if leg is None or leg.contract.underlying != wheel.underlying:
            raise OptionsError("WHEEL_TRANSITION_ILLEGAL", f"{event} needs a leg on {wheel.underlying}")
        expected = "PUT" if event == "put_opened" else "CALL"
        if leg.contract.right != expected:
            raise OptionsError("WHEEL_TRANSITION_ILLEGAL", f"{event} with a {leg.contract.right}")
        return replace(wheel, state=target, leg=leg)
    current = wheel.leg
    assert current is not None
    if event == "put_assigned":
        return replace(wheel, state=target, leg=None, shares=100, share_cost_basis=current.contract.strike)
    if event == "call_assigned":
        return replace(wheel, state=target, leg=None, shares=0, share_cost_basis=None)
    return replace(wheel, state=target, leg=None)

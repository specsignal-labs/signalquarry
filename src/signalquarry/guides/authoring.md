# Writing a strategy

A strategy is a directory with `strategy.py` and `strategy.yaml`, registered in
`signalquarry.toml` under `[strategies] modules`.

```python
from decimal import Decimal
from signalquarry.sdk import Ctx, Decision, Field, Params, strategy, ta


class P(Params):
    symbol: str = "SPY"
    period: int = Field(200, ge=20, le=400)  # bounds document the search space
    weight: Decimal = Field(Decimal("0.95"), ge=0, le=1)


@strategy(params=P, lookback=lambda p: p.period)
def decide(ctx: Ctx, p: P) -> Decision:
    close = ctx.bars(p.symbol).close  # read-only; [-1] = last COMPLETED session
    if close[-1] > ta.sma(close, p.period)[-1]:
        return Decision.target({p.symbol: p.weight}, "PRICE_ABOVE_SMA")
    return Decision.target({}, "PRICE_BELOW_SMA")
```

## The context

- `ctx.bars(symbol)` returns exactly `lookback` completed sessions (`open`,
  `high`, `low`, `close`, `volume`, `sessions` as read-only numpy arrays),
  split-adjusted to the last completed session. There is no "today" bar.
- `ctx.positions`, `ctx.weights`, `ctx.cash`, `ctx.equity` describe the account
  before the decision; `ctx.state` is what you returned last time.
- `ctx.decision_session` is the session the decision will trade in.

## Decisions

- `Decision.target({symbol: weight}, "CODE", state=...)` — weights are fractions
  of equity, long only, each ≤ `limits.max_weight_per_symbol`, sum ≤ 1. An
  unchanged, fully executed target does not trade again.
- `Decision.hold("CODE", state=...)` — keep holdings.
- `Decision.unavailable("CODE")` — no decision (the engine already returns
  `INSUFFICIENT_HISTORY` and `STALE_OBSERVATIONS` for you).

Every reason code you return must be declared in `strategy.yaml`; undeclared
codes stop the run (`REASON_CODE_UNDECLARED`). State must be JSON-serializable
and at most 16 KB.

## Factor portfolios

Import registered factor functions into a normal `@strategy` and return the
composite through `factor_portfolio`. It reads only the completed bars in
`Ctx`, standardizes each factor cross-sectionally, and creates an ordinary
`Decision.target`:

```python
from signalquarry.sdk import Ctx, Decision, FactorInput, Params, factor_portfolio, strategy
from mylab.factors.momentum import P as MomentumParams, momentum
from mylab.factors.value import P as ValueParams, value


class PortfolioParams(Params):
    period: int = 60


@strategy(params=PortfolioParams, lookback=lambda p: p.period)
def decide(ctx: Ctx, p: PortfolioParams) -> Decision:
    return factor_portfolio(
        ctx,
        (
            FactorInput("momentum", momentum, MomentumParams(period=p.period)),
            FactorInput("value", value, ValueParams(period=p.period)),
        ),
        top_quantile=0.2,
        max_weight_per_symbol=0.05,
        turnover_buffer=0.05,
    )
```

This example assumes both factor parameter models have a `period` field.
Set the strategy lookback to at least the longest factor lookback. The factor
parameter models and their values are part of the strategy package's hashed
code; keep them fixed for a run and express alternatives as distinct trials.

The default gives each factor equal weight after cross-sectional z-scoring.
`FactorWeightSnapshot` can supply dated weights computed from earlier
walk-forward windows: `known_at` must precede `effective_from`, and the engine
uses the latest row effective by the current decision. The helper checks these
dates but does not prove the schedule's provenance. Missing scores are not
imputed; symbols with no composite score are ineligible. Existing positions
inside the wider exit rank band are retained to reduce turnover.

The helper returns target weights only. The standard engine and paper planner
apply the strategy's declared costs, minimum order notional, per-symbol limit,
evidence gates, and human paper-arming controls. Factor diagnostics marked
`unverified` remain descriptive and do not become accepted evidence through
composition.

## Rules `sqy check` enforces

- Imports: `signalquarry.sdk`, `numpy`, `pydantic`, pure standard-library modules
  (`math`, `decimal`, `statistics`, `collections`, `dataclasses`, `enum`,
  `functools`, `itertools`, `operator`, `typing`) and your own package.
- No clock, randomness, files, network, environment or mutable globals.
- Deterministic: the same inputs give the same decisions.
- Look-ahead: perturbing every bar after the cutoff must not change decisions.

## `strategy.yaml`

Everything that affects results lives here and is hashed into the
`configuration_hash`: `hypothesis` (statement and falsification), `data`
(symbols, feed, staleness), `account` (cash or margin, initial cash),
`execution` (next open, sizing, costs, minimum order), `params`, `reason_codes`,
`limits`, `evaluation` (holdout months, walk-forward windows, trial budget) and
`benchmark`. Run `sqy schema` for the bundled schemas.

## Execution model

Decisions are made before the open from bars through the previous session and
executed at that session's open: sells before buys, symbols sorted, sized from
prior-close marks. Cash accounts spend settled cash only (T+2 before
2024-05-28, T+1 after). Costs are basis points plus per-share fees. Splits
adjust holdings; dividends are credited on the payable date.

Sell orders can also pay regulatory fees: `costs.sell_bps` (on sale notional, SEC
Section 31 style), `costs.sell_per_share` and `costs.sell_per_order_max` (FINRA TAF
style). They default to zero because the rates change; set the ones in force for the
period you study.

For capacity-sensitive strategies, `execution.fill: {model: volume_cap,
max_volume_fraction: 0.01}` fills at most that share of each session's volume per
symbol (equity only); the rest is retried next session with a `VOLUME_CAPPED` warning.
The cap uses the fill session's volume: it models capacity, it never feeds a decision.
Paper venues fill whole orders, so `check --parity` compares without it.

## Options strategies (single leg)

`kind: options_single_leg` specs add an `options` section (underlyings, quote rules,
tick, per-contract fee, spread haircut, entry cutoff). The strategy is decorated with
`@options_strategy` and returns selectors, never contract symbols:

```python
from signalquarry.sdk.options import OD, OptionsCtx, Strike, options_strategy, sell_call, sell_put


@options_strategy(params=P, lookback=lambda p: p.trend_sessions)
def decide(ctx: OptionsCtx, p: P):
    leg = ctx.leg("QQQ")
    if leg is not None:
        if leg.captured_fraction is not None and leg.captured_fraction > p.take_profit:
            return OD.close("QQQ", "TAKE_PROFIT")
        return OD.hold("HOLD_LEG")
    if ctx.wheel("QQQ").state == "flat":
        return OD.open(sell_put("QQQ", dte=(7, 14), strike=Strike.otm("0.02")), "SELL_PUT")
    return OD.open(sell_call("QQQ", dte=(7, 14), strike=Strike.otm("0.02")), "SELL_CALL")
```

The engine owns the wheel state machine (`flat → short_put → long_shares → covered_call`),
resolves selectors with one resolver shared by simulation and paper (strike on the
out-of-the-money side, then the earliest expiry; limit at the bid floored to the tick)
and manages expiry and assignment. The backtest is a **low-evidence simulator**: option
prices come from a Black-Scholes model with a spread, fills pay a spread haircut and a
per-contract fee, decisions happen at the open and near the close, and early assignment
is not modelled. Results are graded `low_evidence_options`; there is no holdout gate (so
`spec freeze` seals no holdout for options and keeps all of the scarce history), and
claims stay at `walk_forward` or below until a paper forward test passes G5.

`sqy init <dir> --kind options` scaffolds a project around the reference wheel
(`examples/wheel_reference/`): cash-secured puts, covered calls after assignment, trend-aware
strikes, a take-profit buy-back and at most one entry per week.

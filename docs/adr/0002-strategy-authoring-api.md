# 0002. Strategy authoring API

Status: accepted (2026-09-25)

## Context

Developers without quant experience write strategies through coding agents. The shape must be one an LLM writes correctly on the first try, with no order management and no time handling.

## Decision

A strategy is one pure function decorated with `@strategy(params=..., lookback=...)` that maps a read-only `Ctx` and pydantic `Params` to a `Decision` (`target`, `hold`, `unavailable`). Options strategies return leg selectors, never contract symbols. Everything that affects results is declared in `strategy.yaml` and hashed. Reason codes must be declared.

## Consequences

The engine absorbs warm-up, staleness, missing data, rounding and identity checks. Strategy state persists identically on hold and target, so backtest and paper evolve state the same way.

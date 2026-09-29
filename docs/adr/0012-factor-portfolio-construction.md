# 0012. Factor composition and long-only portfolio construction

Status: proposed (2026-09-28)

## Context

SignalQuarry factors produce cross-sectional scores, while ordinary strategies
already return long-only portfolio weights through `Decision.target`. Factor
outputs need a deterministic path into that existing strategy, backtest, and
paper-planning contract. The factor diagnostics available today remain
descriptive and `unverified`; they cannot by themselves support evidence claims.

## Decision

Each factor's available scores are standardized cross-sectionally with the SDK
population z-score. The default composite is an equal-weight mean of the
standardized scores available for each symbol. A caller may instead supply a
dated weight schedule. Every schedule row records `known_at` and
`effective_from`, requires `known_at < effective_from`, and becomes usable only
on or after `effective_from`. The caller owns the schedule and its statistical
provenance; this helper does not estimate IC weights or label them verified.

Missing scores are not imputed. A symbol with no positive-weight standardized
score is ineligible. The target set is the top requested quantile, with a
rank-based exit buffer for current holdings. Score ties use symbol order for
deterministic selection. Selected symbols receive equal weights up to the
declared per-symbol cap; unallocated exposure remains cash.

The result is an ordinary `Decision.target`. Backtest and paper continue to use
the shared order planner, configured transaction costs, minimum order notional,
validation gates, and paper arming controls. No factor-specific execution or
paper authority is added.

## Consequences

An agent can compose reusable `@factor` functions inside a normal strategy and
run the resulting target through existing parity and evidence workflows. A
dated weight schedule makes the information cutoff explicit but does not prove
that its inputs were correctly generated. Factor diagnostics remain
`unverified`; provenance, trial accounting, holdout controls, and ordinary
strategy gates remain separate requirements.

# 0009. Plugins: add-only gates and report sections

Status: accepted (2026-09-26)

## Context

Teams want house rules (turnover caps, exposure limits, desk commentary) without
forking the framework. Every extension point is also a way to weaken the evidence
record, so the surface must stay small.

## Decision

`signalquarry.plugins` (provisional) discovers entry points in two groups,
`signalquarry.gates` and `signalquarry.report_sections`. Each plugin declares a
kebab-case `name` and `api_version` (1). Gates see evaluation results only (gates,
folds, out-of-sample statistics and returns), never data or strategy code. They are
**add-only**: a failing or raising gate caps the claim at `in_sample`, and a passing
one unlocks nothing. Report sections append Markdown under their own heading. Load
errors are reported by `sqy doctor` and as envelope warnings, never as crashes.

Paper brokers are a third group (`signalquarry.brokers`, added later): a plugin's
`create(alias, key_id, secret_key)` returns a broker that must declare `paper_only = True`
and implement the whole paper broker interface plus `calendar()`, or the kernel refuses
it; equity deployments only, market data stays the project's. Providers, cost and fill
models and publish projections stay internal until their protocols settle. The context builder, clocks, ledgers, seals, canonical
hashing, the claim ladder, the minimum gates, the paper kernel's safety checks and
any live-trading hook are not pluggable.

## Consequences

Plugins run with the same trust as strategy code in the user's environment. They
never run in `check` or in paper commands.

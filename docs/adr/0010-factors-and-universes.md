# 0010. Factors and point-in-time universes

Status: proposed (2026-09-28)

## Context

Cross-sectional factor research needs many symbols and a strict boundary between
data available before a decision and later observations. The existing strategy
specification has a 500-symbol limit and does not define a historical universe.

## Decision

Build research panels as derived, per-field Parquet tables keyed by the canonical
dataset identity. The source remains the content-addressed raw-page cache and
dataset manifest (ADR 0004). Panel files stay outside the project and are not
published. The engine verifies and loads selected symbols once, then hands
factor code copied windows containing only sessions strictly before each
decision session, with read-only arrays and a separate presence mask. The full
loaded panel must stay inside the engine because it contains future sessions.
Raw prices remain int64 micro-units; corporate-action adjustment is a separate,
future point-in-time operation. File hashes detect accidental cache damage.

The first pure factor SDK defines `@factor(params=..., lookback=...)`, a context
whose numeric panels have only completed sessions and universe columns, and
cross-sectional rank, z-score, demeaning, winsorization, and numeric-exposure
neutralization. Missing bars are NaN with a separate presence mask. A factor
may omit symbols it cannot score, but emitted scores must be finite and belong
to the supplied universe. The engine copies exactly the declared lookback into
immutable arrays before calling factor code. This is an internal calculation
boundary, not a historical universe attestation or a factor evidence grade.
`sqy check --factor <project.module>` imports exactly one decorated factor from
that project module, checks its package imports, then runs contract, repeated
score, and future-bar perturbation checks on a fixed synthetic four-symbol
panel. `--params-json` supplies parameter values. These empirical checks do
not certify the formula's economics or the provenance of real market data.

Projects may register research factors explicitly in `[factors] modules` in
`signalquarry.toml`. Each module defines one local `@factor` function and has
a neighboring strict `factor.yaml` with an ID, family, version, hypothesis,
and parameters. `sqy factor ls` reports their code-tree and configuration
hashes; `sqy check --factor-id ID` runs synthetic conformance using declared
parameters. The configuration hash includes the complete factor specification,
including declared evaluation choices,
validated parameters, package code, function identity, and framework
major/minor version. Even a hypothesis edit therefore changes its identity.
The code hash covers the full top-level project package, including sibling
helpers; unrelated edits within that package also conservatively change it.
The import policy checks that same tree.
The separate, pure factor-trial configuration identity binds the factor hash,
dataset, dated universe, outcome labels, exact decision sessions, declared
evaluation choices, and accepted-factor comparison set. It does not verify the
underlying manifests or write a trial. A future real-data entry point must do
both before claiming evidence. Listing
and checking registered factors do not consume trials or produce grades.

The internal factor evaluation core binds each decision to a timezone-aware
membership observation and cutoff whose UTC calendar date precedes the decision
session, then calls the same truncated factor runner. The cutoff may be on the
prior UTC date; it is not required to share the decision session's date.
It produces immutable scores and descriptive Spearman rank IC, unannualized
ICIR, chronological-block IC, quintile return, score monotonicity, top-quintile
return after an assumed round-trip cost, long-short spread as a statistic,
one-way turnover, share of supplied predecision ADV, and correlation with
accepted factors. Quintile returns are mean horizon outcomes, not a compounded
portfolio backtest. The report carries its declared panel, universe, and label
identities and is marked synthetic. These calculations currently accept only
explicitly synthetic outcome labels. Matching hashes and timestamps are consistency
checks; a future real-data entry point must verify the underlying universe,
label-adjustment, and code identities and record trials before giving a grade.

The first pure forward-label calculation uses the close before a decision
session as its entry mark and the close of the horizon-th session, counting the
decision session as the first outcome session. It applies split ratios to
shares and accrues cash dividends on their ex-dates, including receivables whose
pay dates follow the label window. When a split and dividend share an ex-date,
the dividend is applied to pre-ex-date shares before the split ratio, matching
the `Dividend` and `Split` value definitions. A missing bar in the holding
window makes that symbol/horizon outcome unavailable; no stale mark or terminal
payoff is inferred. This calculation is source-neutral: it does not prove that
the provider returned every action, resolve ticker succession or delistings,
verify point-in-time universe manifests, restrict access to a sealed holdout, or
grant trial/evidence authority. Those controls remain mandatory at the real-data
API boundary.

Before a panel can support historical factor claims, a separate universe manifest
must record dated membership and observation cutoffs, including inactive assets.
The initial capture path requests Alpaca's full `us_equity` asset list from the
fixed paper origin with no status filter, so active and inactive records are
both observed. It stores the raw response only in the local content-addressed
cache and commits a hash-only snapshot manifest with the local UTC completion
time. An as-of lookup accepts only a verified snapshot captured by its cutoff
and at most 31 days old; dates before the first capture are unavailable.
Alpaca documents no historical as-of query, and its `us_equity` class does not
prove common-stock subtype or listing age. A separate point-in-time archive
and instrument classification are needed for earlier dates and the planned
common-stock universe. A local capture timestamp is not independent proof of
when Alpaca first knew a particular asset state.

`universe build` constructs one monthly membership manifest when matching dated
inputs exist. It requires a complete local classification JSON snapshot with
schema `signalquarry.instrument-classification-snapshot/v1`, a source label, an
`observed_at` timestamp, and one row per asset from the full asset snapshot.
Rows identify either the stable asset UUID or an unambiguous symbol, and carry
`security_type` (`common_stock` or `other`) plus `listed_at` for common stocks.
The canonical classification hash is recorded in the universe manifest, and
the normalized snapshot is retained in the private local cache for offline
replay. The source file is supplied by the caller and its provenance remains
the caller's responsibility. Symbol-keyed snapshots are refused if the asset
snapshot has a duplicate symbol.

The build requires asset and classification snapshots no more than 31 days old
and observed by the UTC cutoff, plus an Alpaca dataset fetched by that cutoff.
The dataset must end no later than the cutoff date and provide 20 complete
sessions strictly before the decision session. When its latest session and
fetch share a date, the fetch must occur at least 16:15 America/New_York so the
regular-session daily bar has finished. Price uses the latest prior
close. Liquidity ranks the median of 20 daily close-times-volume values among
common stocks that pass listing-age, price and data-coverage checks; tied ranks
use their average percentile. Price floor, listing age and minimum percentile
are required inputs, with no hidden defaults. The manifest binds every source
identity, policy value and sorted member-symbol list. API output reports counts
and hashes; the local manifest contains membership and is marked
nonredistributable.

This is a prospective construction path. A historical build is available only
when the asset, classification and market-data observations were captured by
each historical cutoff. The current bars are raw; the price/liquidity filter
does not provide corporate-action-adjusted factor outcomes or prove that an
external classification source was complete when captured.
Factor evaluation must use frozen panel, universe, formula and trial identities.
An agent may propose a formula, but the engine owns truncation, trial accounting,
holdout access and evidence grades. The current panel cache is an internal
foundation; it does not relax the 500-symbol tradable strategy limit or authorize
paper orders.

## Consequences

Panel materialization and reading can be benchmarked independently of backtests.
The dated membership builder consumes, but does not source, external instrument
classification history. Corporate-action knowledge timing, verified outcome
labels, trial accounting, and a real-data evaluation entry point remain
necessary before factor results can be treated as point-in-time evidence.

## Provider references

- Alpaca "Get Assets" API reference: paper origin, array response,
  `us_equity` class and all-status default.
- Alpaca Python SDK "Asset" model reference: stable asset ID, symbol,
  exchange, status and tradability fields.

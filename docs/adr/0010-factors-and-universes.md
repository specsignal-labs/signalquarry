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

Before a panel can support historical factor claims, a separate universe manifest
must record dated membership and observation cutoffs, including inactive assets.
Factor evaluation must use frozen panel, universe, formula and trial identities.
An agent may propose a formula, but the engine owns truncation, trial accounting,
holdout access and evidence grades. The current panel cache is an internal
foundation; it does not relax the 500-symbol tradable strategy limit or authorize
paper orders.

## Consequences

Panel materialization and reading can be benchmarked independently of backtests.
The next stage must add historical universe observations, corporate-action
knowledge timing, factor-level look-ahead checks, trial accounting, and
evaluation before factor results can be treated as point-in-time evidence.

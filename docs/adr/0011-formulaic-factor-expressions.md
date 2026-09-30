# 0011. Safe formulaic factor expressions

Status: proposed (2026-09-28)

## Context

Formulaic factor research needs a small composable language that can be
evaluated over dated panels without giving generated expressions access to
Python execution, arbitrary files, network clients, or future rows. The current
factor SDK receives truncated, read-only panels for one pre-decision context;
future search will also need a deterministic representation of a candidate
formula.

## Decision

Parse expressions with Python's syntax tree parser, then convert only an
explicitly allowed grammar into SignalQuarry's own immutable nodes. Never
compile or execute the source. The accepted inputs are the open, high, low,
close, and volume panels; finite numeric constants; unary plus and minus;
addition, subtraction, multiplication, and division; and the functions delay, delta, ts_mean, ts_std,
ts_rank, ts_corr, rank, zscore, log, abs, and sign. Window sizes must be
integer literals from 1 through 252. Source length, tree depth, node count, and
constant magnitude are bounded.

Every operator returns one value per session and symbol. Evaluation receives
an explicit dated boolean eligibility mask, aligned with the panel; rank and
z-score use only eligible members of that row, and all ineligible outputs stay
NaN. This lets a training panel use the union of symbols while preserving
point-in-time membership for each cross-section. Cross-sectional rank uses
average ties and the SDK percentile convention;
cross-sectional z-score uses population standard deviation. Time-series
windows include the current row and earlier rows only. Rolling operators
require a complete finite window; unavailable results remain NaN. ts_std
uses population standard deviation. ts_corr is Pearson correlation and is
unavailable when either series has no dispersion. log is natural log and
returns NaN for nonpositive inputs. Division by zero and other non-finite
intermediate results become NaN.

The normalized tree is hashed with a versioned expression schema. Any grammar
or semantic change that can alter results requires a grammar-version change.
Expression evaluation only guarantees row-local and trailing behavior. The
caller must still provide the correct training slice, verify data and universe
provenance, record every search trial, and keep the holdout unavailable until
the project gate opens it.

The training-only search core evolves these trees with a seeded PCG64 stream,
bounded mutation and subtree crossover. Its unique-expression budget is checked
against the supplied remaining family budget. Candidate ordering applies an
AST-node complexity penalty and a maximum absolute mean rank correlation
penalty against the accepted-factor library. For the selected horizon, it
estimates effective observations with a Bartlett-weighted autocorrelation
adjustment, subtracts an expected-maximum normal statistic based on the
project-wide trial count, and applies Benjamini-Hochberg across the current
family batch and any prior family p-values supplied by the caller. Outcome
rows whose forward label ends after the training cutoff are excluded.

The current search core accepts only explicitly synthetic labels. Its report
is diagnostic: it does not append trials, grant a grade, accept a factor, emit
factor code, or access a holdout. A real-data API must verify the panel,
universe, label and factor-code manifests, load the persistent family and
project trial history, and append each distinct trial before reporting a
real-data result. This interpreter and search core have no paper or broker
authority.

## Consequences

Formulaic search can reuse one deterministic interpreter and expression
identity. Its later API must bind the training window and trial accounting
before comparing candidates; this ADR alone does not authorize a real-data
claim or imply that a factor is economically useful.

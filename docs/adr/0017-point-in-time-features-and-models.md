# 0017. Point-in-time features and fitted models

Status: proposed (2026-10-09). Nothing here is implemented, and it changes scope: "ML feature
stores" is listed among the non-goals up to 1.0 in `docs/design/ARCHITECTURE.md`. Accepting
this ADR means the owner has decided to lift that non-goal to the extent described here, and
no further.

## Context

A strategy sees completed daily bars, its positions and its own state. Research on
fundamentals, earnings dates, macro series, text-derived scores or any fitted model has no
supported way in. Doing it outside the framework gives up exactly what the framework is for:
the guarantee that a decision used only what was knowable at the time, and a count of what
was tried.

Two failure modes define the design:

- **Revised data.** A fundamental value "as of 2019" downloaded today is often the restated
  one. A backtest that reads it knew something the market did not.
- **Fitted parameters.** A model trained on the whole sample and then backtested on the same
  sample has seen its test data.

## Proposed decision

This is deliberately not a feature store: no online serving, no transformation graph, no
scheduler.

### 1. Feature datasets

A feature dataset is a table of `symbol`, `effective_date`, `observed_at` (UTC, when the value
became knowable), `field`, `value` and `revision`, imported with `sqy features import` into
the same content-addressed cache and manifest scheme as market data (ADR 0004, ADR 0016). The
manifest records the source, the licence, and how `observed_at` was obtained:

- `provided`: the source carries knowledge timestamps;
- `lagged`: the source does not, and a declared conservative lag (in sessions) is applied to
  every row. Results on it are labelled `reconstructed`.

### 2. As-of access

`ctx.feature(name, symbol)` and, for factors, `FactorCtx.feature_panel(name)` return the
latest value whose `observed_at` is before the decision cutoff (the close of the previous
session), and nothing later, including later revisions of an earlier value. `strategy.yaml`
declares the features a strategy reads (`features: [{name, dataset, fields,
max_age_sessions}]`, a hash-neutral field: absent means none, so existing hashes do not move).
A value older than `max_age_sessions` is unavailable, and the decision is
`unavailable(FEATURE_STALE)`.

The look-ahead check of `sqy check` (ADR 0003) is extended: rows observed at or after a cutoff
are perturbed and decisions before it must not change.

### 3. Models: fit outside, freeze, consume

Decision functions stay pure and do not train anything. `sqy model fit --model M` runs a
user-supplied fit function once per walk-forward fold, on data the engine has already
truncated at that fold's training end, and writes:

- per-fold predictions as a feature dataset whose `observed_at` is the start of the fold they
  apply to;
- a model manifest: code hash, parameters, seed, training window, library versions.

A strategy then reads the predictions like any other feature. Each fit configuration is a
trial in the strategy's family. A pretrained model (including a language model behind text
scores) declares `evaluation.holdout.training_cutoff`, which the spec already has: the holdout
starts no later than the day after it.

### 4. Text-derived features

They enter only as frozen feature datasets prepared once, with source hashes, timestamps, the
exact prompts, the model identity and the raw responses kept beside them. Nothing in a
backtest, an evaluation or a paper run calls a model.

### 5. Paper trading

A strategy that reads features is research-only at first: `sqy paper preflight` refuses it
(`FEATURES_NOT_PAPER_ELIGIBLE`). Making features available to the paper kernel, with the same
as-of rule and a fail-closed staleness check, is a separate decision.

## Acceptance tests required before this ADR is accepted

1. A value revised after a cutoff is invisible before it; the original value is returned.
2. Perturbing every feature row observed at or after a cutoff changes no earlier decision.
3. A model's predictions for a fold depend only on data before that fold: perturbing later
   data leaves them unchanged.
4. Each fit configuration appears once in the trial ledger; refitting an identical
   configuration appends nothing.
5. A declared training cutoff moves the holdout start accordingly.
6. A stale or missing feature yields `unavailable`, never a silent default.
7. Existing strategies, their configuration hashes and their ledger hashes are unchanged.
8. No command in the backtest, evaluation or paper path makes a network or model call.

## Open questions for the owner

1. **Whether to lift the non-goal at all before 1.0.**
2. **The default lag** for sources without knowledge timestamps (proposed: declared per
   dataset, with no default, so the choice is always explicit).
3. **Which libraries a fit function may import** (the strategy import allow-list does not
   apply to it; proposed: anything installed, recorded by version in the model manifest).
4. **Whether fit configurations share the strategy's trial budget** (proposed: yes).

## Consequences

Non-price research becomes possible under the same honesty rules as price research. The cost
is real: two new artifact types, an extension of the look-ahead check, and a class of
strategies that cannot be paper-traded until a further decision. If the owner prefers to keep
the pre-1.0 scope, the cheaper alternative is to do nothing and document that such research
belongs outside the framework and earns no evidence grade.

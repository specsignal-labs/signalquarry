# 0014. Real-data formula search and the factor holdout

Status: proposed (2026-10-04)

## Context

ADR 0011 defines a safe formula grammar and a deterministic, training-only search core, and
says a real-data API "must verify the panel, universe, label and factor-code manifests, load the
persistent family and project trial history, and append each distinct trial" before reporting.
That API does not exist. What exists today:

- `FormulaSearchConfig` takes `family_trials_used`, `project_trials_used` and
  `previous_family_p_values` from the caller. On real data a caller could understate them, which
  silently weakens the family budget, the expected-maximum deflation and the Benjamini-Hochberg
  correction that are the point of the search.
- `record_factor_trial` is a storage helper: idempotent per configuration hash, not a verifier.
  Nothing calls it from a search.
- `sqy factor evaluate` replays the dataset and universe manifests, but its output is scope
  `unverified` and records no trial.
- Strategies seal a per-family holdout at `spec freeze` (ADR 0005). A factor's evaluation
  specification already declares a `holdout` (months, default 12, with an optional training
  cutoff) and a `trial_budget` (default 50), and both enter the configuration hash. Nothing seals,
  reads or enforces either for factors, so nothing stops a search from training on the rows a
  later evaluation would need to keep unseen, or from exceeding the declared budget.
- Forward labels stay `unverified` until the corporate-action path of ADR 0013 can attest
  action and lifecycle completeness.

Formulaic search is the most overfitting-prone activity in the framework. Offering it on real
data without the controls above would be worse than not offering it.

## Proposed decision

### 1. Surface and roles

`sqy factor search` and `signalquarry.api.factor_search`, taking a family, a dataset ID, the dated
universe build manifests, a training cutoff, a horizon, a seed, a candidate budget and an optional
accepted-factor comparison set. An agent may call it: it spends budget and cannot widen it.
Human-only, and not exposed through the proposed MCP adapter (#28): sealing or opening the
factor holdout, extending the family budget, and registering a factor in `[factors] modules`.

### 2. Authority tiers

| Tier | What it is | Trial recorded | Grade |
|---|---|---|---|
| T0 (today) | synthetic core; `factor evaluate` | no | none (`synthetic` / `unverified`) |
| T1 (this ADR) | replay-verified inputs, ledger-derived accounting, holdout untouched | yes | none; results labelled `exploratory` |
| T2 (not authorized here) | adds verified label provenance (ADR 0013) and a once-only holdout opening | yes | per the validation gates |

T1 can tell a researcher which formulas are worth a closer look and prices the search honestly.
It cannot support an evidence claim, is never exported by `publish`, and has no paper or broker
authority.

### 3. Verify before spending anything

Before the first trial: replay the dataset pages and each universe build exactly as
`factor evaluate` does; check the normalized formula identity and grammar version; and bind the
whole run in one search identity, the hash of dataset, universe and label identities, training
window, horizon, seed, search configuration, grammar version and accepted-factor hashes. Any
failure refuses the search with nothing written.

### 4. Trial accounting is read from the ledger, never supplied

On the real-data path the family trial count, project factor trial count and prior family
p-values come from `ledger.factor_trials`; the corresponding `FormulaSearchConfig` fields become
internal. Each trial's p-value is stored in its metrics so later searches can reload them.

- The family budget comes from the declared `trial_budget` (plus any recorded human extension),
  not from a caller-supplied `family_budget`. A search whose candidate budget exceeds the
  remaining family budget is refused outright, not silently truncated.
- Every distinct expression evaluated is appended as one `factor_trial` (identity: the existing
  `FactorTrialConfiguration`, with the formula identity in place of project-code hash) before the
  result is reported. A crash mid-search leaves the already appended trials counted, which errs
  toward overstating the search.
- Re-running an identical search appends nothing and returns the recorded results, which doubles
  as a determinism check.
- Factor trials remain separate from the strategy trial counts and budgets, as today. Whether
  they should also enter the strategy DSR denominator is an open question below.

### 5. A factor holdout, enforced structurally

Reuse the declared `holdout` rather than inventing a second setting: the sealed region starts at
the day after the earlier of "the last `months` of data" and the declared `training_cutoff`, the
same rule `holdout_start_for` applies to strategies. Because the dataset's last session moves, the
start date is recorded when sealed, in a new per-family `factor_holdouts` chained log, by an
explicit human command (`sqy factor holdout seal --family F`), and opened at most once. A search
requires a seal and fails closed without one (`FACTOR_HOLDOUT_UNSEALED`). The training cutoff must
precede the seal start by at least the longest horizon, so no training label reaches into the
holdout.
The panel handed to the search is truncated before any expression is evaluated, so isolation does
not rely on the caller or on the search code behaving. Opening the holdout is out of scope here
beyond requiring that T2 prerequisites hold and that the opening is recorded.

### 6. Output is a proposal, not a factor

The report lists candidates with expression, identity, complexity, observations, effective
observations, mean IC, ICIR, t, deflated t, p, q, library redundancy and the discovery flag. The
search never registers a factor or writes project code. A separate human-requested
`sqy factor emit` renders one expression into a `@factor` module and `factor.yaml` skeleton
carrying the expression, seed, search identity and trial counts; it must still pass
`sqy check --factor`, and registration stays an explicit edit to `signalquarry.toml`.

### 7. Provisional reason codes

`FACTOR_HOLDOUT_UNSEALED`, `FACTOR_TRAINING_WINDOW_INVALID`, `FACTOR_TRIAL_BUDGET_EXHAUSTED`,
`FACTOR_SEARCH_INPUT_UNVERIFIED`. Names are provisional and must follow the reason-code
conventions when implemented.

## Acceptance tests required before this ADR is accepted

1. Planted signal: a synthetic panel with a known factor is found.
2. Null: across many independent pure-noise panels with a large budget, the fraction of runs with
   any discovery stays at or below an agreed threshold (proposed: 10 percent over 200 panels).
3. Determinism: identical inputs give identical trials, ledger entries and report hash.
4. Look-ahead: perturbing every row at or after the training cutoff changes no score and no trial.
5. Holdout isolation: instrumentation shows no row at or after the seal start reaches the
   evaluator.
6. Budget: cannot be exceeded or forged; exhaustion refuses before writing.
7. Crash and resume: a mid-search failure keeps appended trials counted and a rerun is idempotent.
8. Refusals write nothing to any ledger.

## Proposed rollout

Four independently reviewable changes: (a) reason codes plus `factor holdout seal|status`;
(b) ledger-derived accounting and a formula-aware trial append; (c) `factor search` at T1;
(d) `factor emit`. Nothing in (a) or (b) exposes search to users.

## Open questions for the owner

1. A family has many factors and each declares its own `trial_budget` and `holdout`. Which is
   authoritative for the family: the first sealed, the most conservative, or one declared at the
   family level? (The strategy side has the same shape and settles it per family.)
2. Seal explicitly (proposed) or automatically at the first search of a family.
3. Should factor trials join the strategy project-wide DSR denominator? Today they are separate
   by design; joining is more conservative but couples two budgets.
4. The null-test threshold in test 2.
5. Whether T1 results may ever appear in published evidence (proposed: never).
6. Benjamini-Hochberg per family (current) or across all families.

## Consequences

Real-data search becomes available only with its accounting enforced by the engine rather than
the caller, and with a holdout that exists before any training happens. The cost is a stricter
workflow than the synthetic core: a seal is required first, budgets are not negotiable by the
caller, and results carry no grade until the label-provenance work of ADR 0013 lands. This ADR
alone authorizes no real-data claim and implies no factor is economically useful.

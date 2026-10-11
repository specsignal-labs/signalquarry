# 0015. Research studies: declared, comparable sets of runs

Status: proposed (2026-10-09). It becomes accepted when the owner settles the questions under
"Open questions" and the acceptance tests below pass.

## Context

A researcher's question is rarely "what did this configuration return?" It is "does this idea
add something over a sensible baseline, and does the answer survive nearby choices?" Today
answering it means running `sqy backtest` and `sqy sweep` by hand and reading envelopes side by
side. Three things are then left to the researcher's discipline:

- **Comparability.** Nothing ensures that the runs being compared share a dataset, a window
  and an account.
- **The comparison rule.** Which metric decides, against which baseline and in which
  direction, is chosen after the numbers are seen.
- **The history.** A crashed or abandoned attempt leaves runs behind but no record of what the
  attempt was.

What exists and is reused, not rebuilt: one run writer (`api.runs.execute_run`) that records
the trial and writes the run directory; the benchmark as an engine-simulated buy-and-hold
curve; `validation.compare` (comparability verdict, configuration differences, a paired
bootstrap interval for the Sharpe difference); the trial ledger with per-family budgets and
`ledger.record_trial`, which is idempotent per configuration and dataset; and the holdout seal.

## Proposed decision

### 1. A study is a file

`studies/<id>/study.yaml`, schema `signalquarry.study/v1`
(`signalquarry._internal.contracts.study.StudySpecV1`): a hypothesis with its falsification, a
subject strategy (`base`), an optional dataset pin and window, one to ten baselines, bounded
variants (declared one by one or as a grid), and exactly one comparison rule: a metric, a
direction and the baseline it is measured against. The study's identity is the hash of that
document. Editing the file makes a different study.

### 2. Arms and what each one costs

| Arm | What it is | Trial on real data |
|---|---|---|
| subject | the strategy as declared in `strategy.yaml` | yes |
| variant, role `candidate` or `ablation` | the subject with parameters or execution settings changed | yes |
| variant, role `sensitivity` | the subject with execution or cost assumptions changed, parameters untouched | no |
| baseline, kind `strategy` | another strategy of the project, unchanged | yes, in its own family |
| baseline, kind `benchmark` | engine buy-and-hold of the benchmark symbol | no |
| baseline, kind `benchmark_scaled` | that curve scaled to the subject's realized volatility | no |

A trial is counted for every arm that a researcher could go on to freeze and claim. Benchmark
baselines have no free choice in them. A sensitivity arm asks "does the result survive a worse
assumption?"; it cannot be selected, and the study never offers it as a candidate. If its
assumptions are later written into `strategy.yaml` and evaluated, that configuration is
recorded as a trial then, like any other.

Counting uses the existing ledger and nothing else: one `record_trial` per distinct
configuration and dataset. A study adds no second counter and cannot lower a count.

### 3. Refuse before spending

`sqy study check` validates the file against the project, lists the arms with their
configuration hashes and reports how many new trials the study would record per family against
each remaining budget. It simulates nothing. `sqy study run` performs the same check first and
refuses the whole study (`TRIAL_BUDGET_EXHAUSTED`) when any family would exceed its budget.
A refusal writes nothing: no run, no trial, no study record.

### 4. Comparable by construction, and never the holdout

Every arm runs on the same resolved dataset and the same window. The window is clipped at the
earliest holdout seal among the families involved, exactly as `backtest` and `sweep` clip, and
a study has no option that opens a holdout. If the arms still turn out not to be comparable
(for example a baseline strategy with a different account), the comparison lists them and
differences nothing, as `sqy runs compare` does.

### 5. Resume, crash and rerun

Each arm is an ordinary run directory. `study run` skips an arm when a run with the same
configuration hash, dataset identity and sessions exists and its result document still matches
its recorded hash. A crash therefore loses at most the arm in flight, and trials already
appended stay counted, which errs toward overstating the search. `--rerun` recomputes every arm
and requires the ledger hash it had before; a difference is reported as a determinism failure.

### 6. The study log

A new chained log, `evidence/studies.jsonl` (per family in the per-family layout, under the
subject's family), records `study_started` (study hash, arm ids and configuration hashes),
`arm_completed` (run id, result hash), `arm_failed` (reason code) and `study_completed`
(comparison hash and verdict). Failed and abandoned attempts stay in it. `sqy evidence verify`
checks the chain. The log is not part of exported evidence bundles.

### 7. The verdict says only what the rule allows

The comparison is computed by `validation.compare` and the declared rule alone:

- `supported`: the subject is better than the `versus` baseline on the declared metric in the
  declared direction, and the paired interval for the difference excludes zero on that side;
- `not_supported`: the subject is not better;
- `insufficient`: the subject is better but the interval includes zero, or the arms are not
  comparable, or there are too few sessions.

Variants are reported beside the subject with the probability of backtest overfitting across
them, never as a ranking to pick from. The verdict is descriptive. It does not set or raise a
claim level; `sqy evaluate` on a frozen specification remains the only path to
`walk_forward` and beyond, and a study on synthetic data states `none`.

### 8. Surface and roles

`sqy study init | check | run | compare | ls | show` and the matching `signalquarry.api`
functions. An agent may call all of them: a study spends budget and cannot widen it. Extending
a budget and opening a holdout stay human-only, as today.

## Acceptance tests required before this ADR is accepted

1. Comparability: every arm of a study has the same dataset identity and sessions; a study
   whose arms cannot be made comparable differences nothing and says why.
2. Counting: on real data a study records exactly one trial per distinct candidate, ablation,
   subject and strategy-baseline configuration, and none for benchmark and sensitivity arms.
3. Budget: a study that would exceed any family's budget is refused with no run, trial or
   study record written.
4. Crash and resume: interrupting a study after some arms and running it again completes it
   with no duplicate trial and the same comparison hash as an uninterrupted run.
5. Determinism: `--rerun` reproduces every arm's ledger hash.
6. Holdout: with a sealed family, no arm's sessions reach the seal date.
7. Verdict: the three outcomes on constructed cases, including a better point estimate whose
   interval contains zero (`insufficient`).
8. Tampering: an edited study log or result document is detected.
9. Claim: a study never changes the claim level of any strategy.

## Implementation status

Implemented as proposed, pending the open questions: the contract, `study init | check | run |
ls | show`, the study log and its verification, resume and `--rerun`. The acceptance tests
above are in `tests/e2e/test_study_flow.py`. `study run --jobs N` simulates arms in worker
processes and leaves every append with the parent, in the declared order
(`tests/e2e/test_parallel_jobs.py`): the trial ledger, the runs and the study log are the ones a
single process writes. A simulation that finished in a worker after an earlier arm failed is
discarded unseen and is not a trial, as an arm that was never reached is not. Not yet built:
`study compare` as a separate re-render (a run always writes its comparison) and per-arm
walk-forward folds.
Each open question below is one decision in the code: whether an arm `counts` in
`api.study`, and the outcome rule in `validation.compare.verdict`.

## Open questions for the owner

1. **Sensitivity arms are not trials.** Proposed above. The stricter alternative counts every
   arm that changes a hashed setting.
2. **Strategy baselines count in their own family.** Proposed above, because that is what
   running them by hand does. The alternative treats a baseline as free when its configuration
   is already frozen.
3. **`insufficient` when the interval contains zero.** Proposed above. The lenient alternative
   reports `supported` on the point estimate and shows the interval beside it.
4. **Interval level.** 90 percent, matching `sharpe_annual_90`.

## Consequences

Research questions become files that can be reviewed before they are run, and their answers
become artifacts another developer can replay. The cost is a stricter path than ad hoc runs:
the comparison rule must be written first, and a study cannot be made to pass by choosing the
metric afterwards. Ad hoc `backtest`, `sweep` and `runs compare` remain available and keep
their own accounting.

# Evidence, trials and claims

## Claim ladder

| claim_level | Earned by | You may say |
|---|---|---|
| `none` | synthetic data, or failed conformance | "The strategy runs." Nothing about performance. |
| `in_sample` | a backtest on real data | "It performed X on the data it was developed on." |
| `walk_forward` | frozen configuration + gates G1–G3 | "It held up out of sample in walk-forward tests." |
| `holdout_passed` | G4 on the sealed holdout (once per family) | "It passed the sealed holdout once." |
| `paper_forward` | clean paper sessions (0.2) | "It traded forward on a paper account for N sessions." |

Always state `evidence.grade` (`synthetic`, `historical`, `paper`) with the claim.

## Trials

Every distinct configuration evaluated on real data is appended to
`evidence/trials.jsonl` (hash-chained; CI rejects a ledger that shrinks). The
project-wide count is the `N` of the deflated Sharpe ratio, so every idea you
try raises the bar for all of them. Each family has a trial budget
(`evaluation.trial_budget`); from 80% on, envelopes warn `TRIAL_BUDGET_NEARLY_USED`,
and at the limit, runs stop with
`TRIAL_BUDGET_EXHAUSTED` until a human records `sqy trials extend --reason ...`.

Factor research has a separate internal `evidence/factor_trials.jsonl` log. It
counts unique factor-configuration hashes project-wide and within each family,
without changing strategy trial counts or budgets. Recording an entry does not
verify its data provenance, assign an evidence grade, or authorize a holdout
evaluation.

`sqy evidence verify` checks every ledger and paper journal; with
`--base <git ref>` it also fails if any of them changed other than by appending.
New projects run both in `.github/workflows/check.yml`.

### Several families in one repository

A research lab that may sell strategy families separately sets

```toml
[evidence]
layout = "per_family"            # family_root defaults to "families/{family}"
```

Each family's trials, freezes and holdout opening then live in
`families/<family>/evidence/`, so a family can be transferred with its own
verifiable history. Every family append is chained into
`evidence/project_index.jsonl`, and the deflated Sharpe ratio still counts every
family's trials. `sqy evidence verify` fails if a family log no longer ends where
the index recorded it.

`sqy init NAME --lab` creates such a lab: families under `families/`, a transfer-drill
family, default-deny publication policies, checks for one family per commit and no
cross-family imports, a family extractor that preserves tree hashes, a bundle leak
scan, a data-room exporter and a guarded deploy script (see its `docs/design/LAB.md`).

### One-off variants

`sqy backtest --strategy <id> --param period=150 --label p150` runs the strategy with
parameters overridden for that run only. `strategy.yaml` is not changed, the run has
its own configuration hash, and its `result.json` records the parameters and the
label. On real data a new configuration is a trial like any other, and the run is
refused beforehand if it would exceed the family's budget. Evaluation always reads
`strategy.yaml`: to keep a variant, write its parameters there and freeze.

### Parameter sweeps

`sqy sweep --strategy <id> --param period=50,100,150` backtests a grid (up to 200
points). On real data every point is a recorded trial, and the sweep refuses to start
if it would exceed the family's budget. The envelope reports the **probability of
backtest overfitting** (PBO, by combinatorially symmetric cross-validation over the
grid): the share of in-sample/out-of-sample splits in which the in-sample winner ranks
at or below the median out of sample. A PBO near 0.5 means the "best" row is mostly
luck. Choose one configuration by reasoning, freeze it, then evaluate.

Every point is written as its own run under `.signalquarry/runs/`, with the same
artifacts as a backtest (`--summary-only` keeps `result.json` and the equity curves
and leaves out fills and decisions). The sweep itself is recorded in
`.signalquarry/sweeps/<sweep_id>/` as `sweep.json` (grid, run ids, metrics, PBO) and
`sweep.csv`. Each run's `result.json` carries the `params` and the hashed `spec`
that produced it, so runs can be compared later.

### Listing and comparing runs

```bash
sqy runs ls --strategy sma-trend          # recent runs, newest first (sweep points included)
sqy runs show RUN_ID                      # one run's result document and artifacts
sqy runs compare RUN_A RUN_B [RUN_C …]    # every run against the first
```

`runs compare` lists the runs side by side, reports how each differs from the
reference in parameters and specification, and, for comparable runs, gives the
correlation of daily returns and the difference in annualized Sharpe with a paired
moving-block bootstrap 90% interval. An interval that contains zero means this
sample cannot tell the two runs apart. It writes `comparison.json`, `comparison.md`
and an overlay chart under `.signalquarry/comparisons/`.

Runs are differenced only when they share the dataset, the sessions, the account and
the evidence grade. Otherwise the command still lists them, warns
`RUNS_NOT_COMPARABLE` with the reasons, and differences nothing. A run whose
`result.json` no longer matches its recorded hash is refused
(`RUN_ARTIFACT_INVALID`). Comparisons are in-sample and never raise a claim level:
picking the best row is exactly what the trial ledger and the deflated Sharpe ratio
are there to price.

### Results in a notebook

`signalquarry.research` loads what the commands wrote, read-only, without touching the
ledgers:

```python
from signalquarry import research

run = research.latest_run("sma-trend")  # or research.load_run(RUN_ID)
run.metrics["sharpe"], run.params
run.to_arrow()  # session, equity, returns, benchmark
run.to_pandas()  # the same as a DataFrame, if pandas is installed

sweep = research.load_sweep(SWEEP_ID)
sweep.to_arrow()  # one row per grid point
research.latest_study("trend-vs-hold")["verdict"]
```

A result document that no longer matches its recorded hash is refused. The module is
provisional in 0.x. Looking at results this way does not count as a trial; running
another configuration does.

## Studies: a declared comparison

A study answers "does this idea add something over a baseline?" with the rule written
down first. It is a file, `studies/<id>/study.yaml`:

```yaml
schema: signalquarry.study/v1
id: trend-vs-hold
hypothesis:
  statement: Holding SPY only above its 200-session average lowers drawdown versus buy-and-hold.
  falsification: Max drawdown over the same sessions is not lower than buy-and-hold.
base: sma-trend                       # the subject strategy
baselines:
  - {id: buy-and-hold, kind: benchmark}        # engine buy-and-hold of the benchmark
  - {id: vol-matched, kind: benchmark_scaled}  # the same, scaled to the subject's volatility
variants:
  - {id: half-weight, params: {weight: "0.5"}, role: ablation}
  - {id: costs-x2, execution: {costs: {bps: "10"}}, role: sensitivity}
grid: {period: [100, 150]}            # optional; each point is a candidate
compare: {metric: max_drawdown, direction: lower, versus: buy-and-hold}
```

```bash
sqy study init --strategy sma-trend --id trend-vs-hold   # scaffold the file
sqy study check --study trend-vs-hold    # arms, and the trials it would record; runs nothing
sqy study run --study trend-vs-hold      # run every arm and compare
sqy study show --study trend-vs-hold     # the latest result
sqy study ls
```

What a study guarantees:

- **Comparable arms.** Every arm runs on the subject's dataset over the same window, clipped
  at the sealed holdout. A study has no way to open a holdout.
- **Counted arms.** On real data the subject, every `candidate` and `ablation` variant and
  every `strategy` baseline is a trial in its family. Benchmark baselines are not trials.
  A `sensitivity` variant changes execution or cost assumptions only, cannot be selected, and
  is not a trial. `study check` shows the count against each budget, and a study that would
  exceed a budget is refused before anything runs.
- **A rule-bound verdict.** `supported` needs the subject to be better than the `versus`
  baseline on the declared metric, with a paired 90% bootstrap interval of the difference
  wholly on that side. Better with an interval that contains zero is `insufficient`. Not
  better is `not_supported`. The verdict is descriptive: it never raises a claim level, and
  only `sqy evaluate` on a frozen configuration does.
- **A history.** Each arm is an ordinary run, so a second `study run` reuses finished arms
  and an interrupted study picks up where it stopped without counting anything twice.
  `--rerun` recomputes every arm and requires the ledger hash recorded before. On real data
  `evidence/studies.jsonl` records what each study set out to run and what became of every
  arm, including failures; `sqy evidence verify` checks it.

The result is written to `.signalquarry/studies/<run>/` as `study.json`, `comparison.md`
and a chart. Variants appear beside the subject with the probability of backtest
overfitting across them, not as a ranking: each configuration a study tries raises the
deflated-Sharpe bar the strategy must later clear. Editing `study.yaml` makes a different
study with its own identity.

## Freeze and holdout

`sqy spec freeze --strategy <id>` records the configuration hash, hypothesis and
gate version, and seals the family's holdout (the last `holdout.months` months
of data). Afterwards every command stops before the seal. `sqy evaluate
--holdout` opens it **once per family**, only after G1–G3 pass; a second attempt
returns `HOLDOUT_REUSED`. Commit before freezing: a dirty tree is recorded.
`sqy holdout seal --strategy <id>` seals the family before you start exploring,
which is stricter still; a later freeze reuses that seal.

If the strategy depends on a model or dataset with a training cutoff (for example an
LLM behind text-derived features), declare `evaluation.holdout.training_cutoff`: the
holdout then starts the day after the cutoff whenever that is earlier than the last
`months`, so the sealed period is data the model cannot have seen.

## Gates

| Gate | Criterion |
|---|---|
| G1 sample | ≥ 5 years of history and ≥ 30 rebalancing sessions |
| G2 walk-forward | ≥ 6 out-of-sample folds; OOS PSR(0) ≥ 0.95 and DSR ≥ 0.95 using the project-wide trial count |
| G3 stress | Sharpe at 2× costs and with a one-session delay each ≥ 0.5 × base; net return > 0 |
| G4 holdout | Sharpe > 0 and max drawdown ≤ 1.5 × the worst walk-forward fold |

The envelope's `oos` also reports `sharpe_annual_90`, a 90% moving-block bootstrap
interval (20-session blocks, fixed seed) for the out-of-sample annual Sharpe. It is
not a gate; a wide interval says the sample cannot pin the Sharpe down.

When a gate fails, the answer is a new hypothesis, not looser gates.

## Benchmark comparison

Set `benchmark: <SYMBOL>` in `strategy.yaml` to compare every run with holding that
symbol. The benchmark is simulated by the same engine: one position at full weight
under the strategy's own account, execution and cost settings, with dividends
reinvested. `sqy data fetch --strategy <id>` fetches its bars with the strategy's.

- `sqy backtest` writes `benchmark.csv` beside `equity.csv`, adds a `benchmark`
  block to `result.json` (the benchmark's own metrics and the run's excess return,
  tracking error, information ratio, beta, alpha, correlation and up and down
  capture) and puts the headline numbers in the envelope's `metrics`. It also
  records the deepest drawdown episodes and one-way turnover with fee drag.
- `sqy evaluate` adds the benchmark's return and drawdown to each walk-forward fold
  and an out-of-sample `relative` block, with the number of folds in which the
  strategy was ahead.

All of it is descriptive. No gate reads the benchmark, the comparison never raises a
claim level, and the benchmark run is not a trial. Relative statistics need at least
20 sessions and report `insufficient` otherwise. If no recorded dataset covers the
benchmark, the command still succeeds and warns `BENCHMARK_DATA_MISSING`.

## Plugin gates

Installed packages can add gates through the `signalquarry.gates` entry point (see
`signalquarry.plugins`). They run after G1–G4, appear as `plugin:<name>` in the
envelope's `gates`, and are add-only: a failing or broken plugin gate caps the claim at
`in_sample`; a passing one unlocks nothing. `sqy doctor` lists what is installed.

## Reports

### Descriptive diagnostics

`sqy diagnose --strategy ID [--run RUN_ID]` describes a recorded backtest (the latest
for the strategy by default). It writes a separate hashed `diagnostics.json` under
`.signalquarry/diagnostics/`; repeating it creates a new directory. `sqy report`
includes the newest verified diagnostics for its run when available.

- **Regimes and years:** performance by calendar year and by benchmark trend and
  volatility, using only previous-session information for the market labels.
  Trend and volatility require the recorded benchmark curve. Regimes looked at
  after the fact are hypothesis generation, not evidence: they were not declared
  before the run.
- **Costs:** returns, Sharpe and drawdown at 0, 1, 2 and 4 times execution cost bps,
  with the first zero-return crossing interpolated between neighbouring points.
  Other fees stay as declared. No break-even is reported if returns start
  non-positive or remain positive throughout the sampled range. The project must
  still reproduce the run's configuration, dataset and base total return. Options
  strategies have a different cost model and this section is unavailable.
- **Folds:** positive-return share, count ahead of the benchmark, best and worst
  returns and sample dispersion from the newest matching recorded evaluation.
  Fewer than three folds is marked insufficient.
- **Parameters:** best point, neighbour median and plateau (neighbour median divided
  by the best score) from the newest recorded sweep containing the configuration.
  Neighbours are those of the sweep's best point, using the declared grid order.

All four diagnostics are descriptive and never change a claim, gate or trial count.
Cost reruns write no runs or trials; the original run directory is never modified.
Unavailable sections are explained and omitted from the report's tables.

### Rendering reports

`sqy report --strategy <id>` renders `report.md` and `equity.svg` from run
artifacts. The claim level and its permitted wording come first; every report
ends with a hypothetical-results disclaimer. With a benchmark, the report adds a
strategy-against-benchmark table, the benchmark line on the chart and a per-fold
table; drawdown episodes and trading activity are always shown.

Report-section plugins (`signalquarry.report_sections`) append their own headed
sections to `report.md`.

## Publishing evidence

Nothing is published by default. A family publishes only what its
`publication/<family>.publication.yaml` allows:

```yaml
schema: signalquarry.publication/v1
family: my-family
family_label: Trend following
commercial: true            # default; commercial families are category-tier
tier: category              # category | results | rules
backtest_results: deny      # only non-commercial results/rules tiers may allow
forward: {resolution: weekly, lag_days: 14}
strategies:
  - {id: my-strategy, title: "Trend strategy A", summary: "...", assets: "US equities", deployment: my-alias}
```

`sqy evidence export --family my-family` writes a verifiable bundle (every file listed
with its SHA-256; `sqy evidence verify --bundle DIR`). Commercial bundles carry the
strategy profile and lagged period returns from the paper journal; positions, fills
and exposure are always withheld. `--tier nda` adds full run results, the trial
ledger and freezes for a private data room.

## Commitments

`sqy commit create --strategy <id>` (after `sqy spec freeze`) records salted
digests of everything that determines results — spec, parameters, gates, freeze,
dataset, holdout, ledger head and code tree — in `evidence/commitments/<family>/`.
The salt stays in `$SIGNALQUARRY_CONFIG_DIR/salts/` (mode 0600, back it up), so the
public record reveals nothing about the configuration. `--alias <deployment>`
commits to a paper journal's head instead.

With the OpenTimestamps client installed (`pip install 'signalquarry[ots]'`) each
record is stamped (`<record>.ots`); otherwise its proof is `pending`. Later,
`sqy commit reveal --id <id> --out opening.json` writes the private opening for a
buyer under NDA, and `sqy commit verify --record <record> --reveal opening.json`
proves the configuration was fixed when the timestamp says. A timestamp proves when
a commitment existed, never a result.

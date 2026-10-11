# Research method

A backtest answers "what would this configuration have returned?" Research asks something
else: "does this idea add anything over a sensible alternative, and would I have known?"
This guide is the working method SignalQuarry is built around, followed by one study
worked from start to finish, including the part where the hypothesis is rejected.

## Five rules

1. **State the idea so it can fail.** `hypothesis.statement` and
   `hypothesis.falsification` in `strategy.yaml` (and in a study) are written before
   any number is seen. "Reduces drawdown versus buy-and-hold" can fail. "Performs well"
   cannot.
2. **Compare with something.** Declare `benchmark:` in `strategy.yaml`. Every backtest,
   every walk-forward fold and the report then show the strategy against holding that
   symbol under the same account and costs. A return, a Sharpe ratio or a drawdown
   quoted without its benchmark is half a result.
3. **Change one thing at a time, and count it.** A variant is a new configuration.
   On real data each one is a trial, and the deflated Sharpe ratio the strategy must
   later clear is computed over every trial in the project. Ten casual variations make
   the eleventh result harder to believe, which is correct.
4. **Decide the comparison before running it.** A study fixes one metric, one
   direction and one baseline in advance. Choosing the metric after seeing six of them
   is how a null result becomes a finding.
5. **Report what the evidence level allows.** `claim_level` in every envelope says
   what may be claimed. Comparisons and studies are in-sample: they tell you whether
   an idea is worth freezing and evaluating, not that it works.

## The commands, by question

| Question | Command |
|---|---|
| Is the data complete enough to use? | `sqy data quality --strategy ID` |
| What did this configuration return, against its benchmark? | `sqy backtest --strategy ID` |
| What if one parameter were different? | `sqy backtest --strategy ID --param name=value --label NAME` |
| How does the result move across a grid, and is the best row luck? | `sqy sweep --strategy ID --param name=a,b,c` (reports PBO) |
| Are two runs actually different? | `sqy runs compare RUN_A RUN_B` |
| Does the idea beat the baseline, by a rule fixed beforehand? | `sqy study run --study ID` |
| Where does the result come from: which regimes, how cost-sensitive, how stable? | `sqy diagnose --strategy ID` |
| Does it hold up out of sample? | `sqy spec freeze`, then `sqy evaluate` |
| What may I say about it? | `sqy report --strategy ID` |

`sqy runs compare` and `sqy study run` report a paired 90% bootstrap interval for a
difference. When that interval contains zero, the sample cannot tell the two apart, and
the honest summary is "no detectable difference", whatever the point estimates say.

`sqy diagnose` is for understanding a run, not for improving it. A regime in which a
strategy did badly is a question for the next hypothesis. Adding a rule that avoids
that regime and backtesting again on the same data is fitting to the past, and the
trial ledger will count it.
Repeat `--exposure SYMBOL` to ask how much daily return variation is associated
with passive reference positions, with betas, annual contributions and remaining
alpha reported using Newey–West uncertainty. This is an in-sample description;
references chosen after seeing the run generate hypotheses, and an alpha with a
small t is indistinguishable from zero.

## A worked study

The demo project's starter strategy holds one symbol only while it closes above its
200-session average. Its stated hypothesis is that this reduces drawdown compared with
simply holding the symbol. The demo data is synthetic, so the numbers below say
nothing about markets. The method is the point.

```bash
sqy init trend-lab --demo && cd trend-lab
sqy study init --strategy sma-trend --id trend-vs-hold
```

Edit `studies/trend-vs-hold/study.yaml` so that it asks exactly the question in the
hypothesis:

```yaml
schema: signalquarry.study/v1
id: trend-vs-hold
hypothesis:
  statement: Holding SYNA only while it closes above its 200-session average reduces drawdown versus buy-and-hold.
  falsification: Maximum drawdown over the same sessions is not lower than buy-and-hold.
base: sma-trend
baselines:
  - {id: buy-and-hold, kind: benchmark}
  - {id: vol-matched, kind: benchmark_scaled}
variants:
  - {id: half-weight, params: {weight: "0.5"}, role: ablation, note: Half the exposure and no other change.}
  - {id: costs-x2, execution: {costs: {bps: "10"}}, role: sensitivity}
grid: {period: [100, 150]}
compare: {metric: max_drawdown, direction: lower, versus: buy-and-hold}
```

Three choices in that file carry the method:

- The **volatility-matched baseline** asks whether the filter does anything that
  holding less of the benchmark would not. A strategy that is in cash half the time
  will have a smaller drawdown than a fully invested one for that reason alone.
- The **ablation** removes half the exposure and nothing else, for the same reason
  from the other side.
- The **sensitivity arm** doubles costs. It cannot be selected and is not a trial; it
  shows whether the answer depends on an optimistic assumption.

```bash
sqy study check --study trend-vs-hold     # 7 arms; on real data, 4 would be trials
sqy study run --study trend-vs-hold
```

The verdict on the demo data is `not_supported`:

| Arm | Role | Total return | Sharpe | Max drawdown |
|---|---|---|---|---|
| base | subject | 8.97% | 0.12 | 40.66% |
| costs-x2 | sensitivity | 2.71% | 0.07 | 43.70% |
| half-weight | ablation | 5.58% | 0.10 | 24.51% |
| grid-period-100 | candidate | -13.22% | -0.05 | 47.03% |
| grid-period-150 | candidate | -8.71% | -0.01 | 45.21% |
| buy-and-hold | baseline | 74.73% | 0.36 | 36.55% |
| vol-matched | baseline | 50.64% | 0.36 | 25.43% |

Reading it as a researcher would:

- The filter's drawdown, 40.66%, is **higher** than buy-and-hold's 36.55%. The
  hypothesis fails by its own falsification test. That is the result.
- The half-weight ablation does have a smaller drawdown than buy-and-hold, and the
  volatility-matched benchmark has about the same one, at nine times the return.
  Whatever protection lower exposure buys, the benchmark delivers it more cheaply. The
  filter itself contributes nothing here.
- Both shorter averages are worse. With a probability of backtest overfitting of 0.65
  across the four candidates, preferring any of them would be choosing noise.
- Doubling costs takes most of the remaining return away. The result was fragile
  before it was wrong.

What not to do next is as much part of the method: do not switch the rule to a metric
on which the filter looks better, do not add variants until one passes, and do not
describe the half-weight arm as "the strategy with improved risk". Each of those is a
new study with its own count, and on real data each costs trials. The next step is a
different hypothesis, or none.

Had the verdict been `supported`, the next step would still not be a claim. It would
be `sqy spec freeze` and `sqy evaluate`: walk-forward and stress gates on the frozen
configuration, with the study's trials already counted against it.

## On real data

Everything above works the same with recorded data, with three differences that
matter:

- Trials are recorded. `sqy study check` shows how many a study will add to each
  family, and a study that would exceed a budget is refused before it runs.
- A sealed holdout is never touched by a backtest, a sweep, a comparison or a study.
  Seal it early (`sqy holdout seal`) if you intend to explore.
- Check the data first. A gap, a stale price or a missing split can produce a result
  on its own.

## For coding agents

The project's `AGENTS.md` carries this method in short form under "Research method".
When asked a comparative question, an agent should write the study file, run
`study check` and `study run`, and report the verdict as returned, with the benchmark
numbers and the claim level. It should not adjust the rule after a result, and it
should say so when a comparison is `insufficient`.

# Reference study: a trend filter against buy-and-hold

The study worked through in the [Research method](../../docs/research.md) guide. It asks
whether the demo project's starter strategy (hold one symbol only above its 200-session
average) has a lower maximum drawdown than holding the symbol, with a volatility-matched
baseline, an ablation, a cost sensitivity arm and two alternative averages.

```bash
sqy init trend-lab --demo && cd trend-lab
mkdir -p studies/trend-vs-hold
cp /path/to/examples/trend_vs_hold_study/study.yaml studies/trend-vs-hold/
sqy study check --study trend-vs-hold
sqy study run --study trend-vs-hold
```

The verdict on the demo's synthetic data is `not_supported`: the filter's drawdown is higher
than buy-and-hold's. Synthetic data says nothing about markets; the example shows the
method, including a hypothesis that is rejected and left rejected. The repository's tests run
this file and check the numbers quoted in the guide.

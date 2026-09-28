# SDK reference

The authoring API (`signalquarry.sdk`) is the stable surface: additive changes only in 0.x.

::: signalquarry.sdk
    options:
      show_root_heading: false
      members: [strategy, Params, Field, Ctx, Bars, Decision, definition_of, factor, FactorCtx, FactorDef, factor_definition_of]

## Indicators (`signalquarry.sdk.ta`)

::: signalquarry.sdk.ta

## Cross-sectional research (`signalquarry.sdk.xs`)

`factor` is a pure research API. It requires a separately supplied universe and
does not produce an evidence grade or authorize paper orders. Panel prices are
raw; corporate-action adjustment and historical universe provenance remain
required before point-in-time factor claims.

```python
from signalquarry.sdk import FactorCtx, Params, factor, xs

class Momentum(Params):
    lookback: int = 21

@factor(params=Momentum, lookback=lambda p: p.lookback)
def momentum(ctx: FactorCtx, p: Momentum) -> dict[str, float]:
    close = ctx.panel("close")
    change = close[-1] / close[0] - 1
    scores = xs.rank(change)
    return {
        symbol: float(score)
        for symbol, score in zip(ctx.universe, scores, strict=True)
        if score == score  # NaN means insufficient coverage
    }
```

Run `sqy check --factor your_package.momentum --params-json '{"lookback":21}'`
from the project to check one decorated factor on synthetic panels. The module
must live under the project and define exactly one factor. The check reports
import policy, score contract, determinism, and future-bar perturbation results.
It imports and runs the authored module, so use an isolated development
environment for untrusted code. It does not establish historical data or
universe provenance.

For a project factor that an agent can discover by ID, add its module to
`signalquarry.toml`:

```toml
[factors]
modules = ["your_package.momentum.factor"]
```

Place `factor.yaml` beside `factor.py`:

```yaml
schema: signalquarry.factor/v1
id: price-momentum
family: momentum
version: "1"
hypothesis:
  statement: Past prices predict next returns.
  falsification: Rank IC is nonpositive out of sample.
params:
  lookback: 21
evaluation:
  horizons: [1, 5, 21]
  chronological_blocks: 6
  cost_bps: 10
  capital: 1000000
  trial_budget: 50
  holdout:
    months: 12
```

`sqy factor ls` lists the factor's code-tree and configuration hashes.
`sqy check --factor-id price-momentum` runs the synthetic checks with the
declared parameters. Editing code, metadata, or parameters changes the
configuration identity. Listing and checking do not record a trial or grant
an evidence grade. The pure factor-trial key binds evaluation settings, data,
dated universe, labels, decision sessions and accepted-factor comparisons.
Constructing that key does not verify the sources or append a real-data trial;
those steps still need a controlled evaluator. Both commands
import the authored factor module, so inspect untrusted project code before
running them outside an isolated development environment.
The code hash covers the factor's full top-level project package, including
shared helpers; edits elsewhere in that package also change its identity. The
import policy checks that complete package as well.

::: signalquarry.sdk.xs

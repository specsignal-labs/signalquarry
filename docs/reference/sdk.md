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

::: signalquarry.sdk.xs

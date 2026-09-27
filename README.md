# SignalQuarry

SignalQuarry helps developers, working with their own coding agents, turn a
strategy idea into an honest backtest on Alpaca data and then into a paper
forward test. See `docs/design/ARCHITECTURE.md` and `docs/adr/`.

## Quickstart

```bash
pip install signalquarry
sqy init my-lab --demo            # synthetic data, no credentials
cd my-lab
sqy check --parity                # contract, determinism, look-ahead, backtest = paper
sqy backtest --strategy sma-trend
sqy spec freeze --strategy sma-trend
sqy evaluate --strategy sma-trend # the demo is expected to fail its gates
sqy report --strategy sma-trend
```

`sqy init my-wheel --demo --kind options` starts from the reference options wheel.
Open the project in your coding agent: it reads `AGENTS.md`.

## What it guarantees

- **One decision function, two clocks.** The backtest and the Alpaca paper runner call
  the same `decide(ctx, params)`; `sqy check --parity` replays sessions through both and
  fails if orders, fills or cash differ.
- **No look-ahead by construction.** Strategies see read-only, truncated history; a
  mutation test perturbs every later bar and requires identical decisions.
- **An evidence record that is hard to game.** The engine writes a hash-chained trial
  ledger, a sealed one-time holdout, walk-forward and stress gates with a deflated
  Sharpe ratio across every trial, and a claim level that says what you may state.
- **Paper, never live.** Only Alpaca's paper origin exists in the code; a human arms
  each deployment.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.

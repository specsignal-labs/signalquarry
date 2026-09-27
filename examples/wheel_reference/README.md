# Reference wheel (V13.5 rules)

The options example for SignalQuarry 1.0. It restates, in the options authoring API,
the rules of the V13.5 QQQ wheel that ran on an Alpaca paper account and was published
in the `alpaca-hackathon` repository. The rules are public; this example adds nothing
private.

| Rule | Value |
|---|---|
| Underlying | QQQ, one contract |
| Trend | last completed close vs its 50-session mean |
| Flat → sell put | 1% OTM in an uptrend, 3% OTM in a downtrend |
| 100 shares → sell call | 3% OTM in an uptrend, 1% OTM in a downtrend |
| Expiration | 7–14 days |
| Take profit | buy back when more than 15% of the premium is captured |
| Entries | at most one per week |
| Limit prices | sell at the bid floored to the cent; buy back at the ask raised to the cent |
| Quotes | bid ≥ $0.05, relative spread ≤ 25%, age ≤ 15 s |
| New entries | not after 15:15 New York time |

## Try it

```bash
sqy init wheel-lab --demo --kind options    # synthetic data: runs offline
cd wheel-lab
sqy check && sqy backtest --strategy wheel && sqy paper dry-run --alias demo
```

`strategy.py` and `strategy.yaml` here are the same files `sqy init --kind options`
creates (with QQQ and SIP data).

## What the evidence can and cannot say

Backtests of this strategy are **low evidence**: option prices come from a
Black-Scholes model with a spread, not from recorded quotes, decisions happen at two
checkpoints per day instead of every minute, and early assignment is not modelled. They
are graded `low_evidence_options` and never support more than a `walk_forward` claim.
The primary evidence for an options strategy is a paper forward test (`sqy paper
drift`-style review of the journal, G5).

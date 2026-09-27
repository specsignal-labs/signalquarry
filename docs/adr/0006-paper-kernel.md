# 0006. Paper trading kernel

Status: accepted (2026-09-25); implemented in 0.1

## Context

Paper trading is the forward proof of a backtest; the runtime must be idempotent, reconcile before acting and fail closed.

## Decision

- **One step, two clocks.** `run-once` calls the engine's `plan_pre_open`, which uses the same decide, rebalance and sizing code as the backtest (`plan_orders`). A parity test runs 76 synthetic sessions (including a split) through the FakeBroker and requires identical fills, positions, cash and decisions.
- **Lease.** `RunLease` (`flock`) guards every state-changing command (`arm`, `run-once`, `reconcile`, `halt`); contention exits 75.
- **Journal.** `paper/<alias>/journal.jsonl` is hash-chained, verified once per lease, fsynced per append and authoritative. Kinds: `armed`, `halted`, `session_started` (the full plan), `order_intent` (written before submission), `order_submitted`, `order_final`, `session_completed`, `reconciled`.
- **Client order ids** are deterministic: `sq-<alias6>-<intent10>-<retry>`. Ambiguous submission errors and crashes are resolved by lookup, never by a blind resubmit (N1). Orphaned fills are recovered on the next run (N2).
- **Arm token.** `ArmTokenV1` binds alias, configuration hash, freeze hash, account-id hash and journal head, expires after `arm_days`, and is journaled; editing it, re-freezing, changing code or switching accounts disarms. Arming needs a human at a TTY typing the alias. Halts are cleared only by re-arming.
- **Reconcile before decide.** Own orders from earlier sessions are finalized or canceled (unconfirmed cancels exit 75); unmanaged orders or positions block; position drift that fills and splits cannot explain halts the deployment. Today's pending split is retried (75).
- **Guards.** Clock skew, pre-open window (default 09:00–09:25 New York), daily drawdown, account status, expected account hash and a key deny-list.
- **Origin.** The Alpaca adapter hard-codes `https://paper-api.alpaca.markets`; a test fails if a live trading host appears in the package. Brokers must declare `paper_only`.
- **Simulated broker.** `broker: simulated` replays the project dataset offline for `dry-run`, `preflight` and `status`; it can never be armed.
- **No daemon.** `sqy paper schedule` writes systemd, launchd, cron and (demo-only) GitHub Actions templates for review.

## Known divergences (reported, not hidden)

- Buys are clipped pre-open at a buffered prior close (`buy_price_buffer_bps`); the simulator clips at the actual open. They differ only when cash binds.
- Alpaca paper charges no commissions and omits dividends; the 0.2 shadow ledger applies model costs and dividends to paper results.
- Margin accounts: Alpaca paper buying power differs from the simulator's no-borrowing model.
- Market-on-open orders need whole shares; fractional sizing is refused for paper deployments.

## Consequences

No live-trading path exists. From 0.2, `isolation: process` (default) runs decide in a child process with a scrubbed environment and rlimits; the docs state it is not a sandbox.

# Paper trading

Paper trading is the forward proof of a backtest. It runs on a dedicated
Alpaca **paper** account; the adapter hard-codes the paper origin and there is
no live path.

## Other paper brokers

`broker: plugin:<name>` uses a paper broker from an installed package (entry point
`signalquarry.brokers`; `sqy doctor` lists them). The kernel refuses a broker that does not
declare `paper_only = True` or lacks part of the paper broker interface, and every guard,
the journal and the arm step apply unchanged. Equity deployments only; market data still
comes from the project's data provider.

## Parity first

`sqy check --parity` replays the last 60 sessions of the synthetic check window through
the real paper kernel against an in-memory venue and through the backtest, and fails
unless orders, fills, positions, cash and journalled decisions (wheel events for
options) agree. Costs, dividends and option spread haircuts are removed from both
sides because a paper venue reproduces none of them. The project CI template runs it.

## A deployment

`paper/<alias>.paper.yaml`:

```yaml
schema: signalquarry.paper/v1
alias: demo
strategy: sma-trend
broker: alpaca-paper          # or "simulated": offline dry-run only
submission: disabled          # a human sets "enabled" after review
expected_account_id_sha256:   # optional: refuse any other account
denied_key_id_sha256: []      # keys of other deployments that must never be used here
arm_days: 30
isolation: process          # decide runs in a child process without credentials
window: {start: "09:00", end: "09:25"}   # New York time, before the open
guards:
  max_daily_drawdown: "0.10"
  max_clock_skew_seconds: 30
  buy_price_buffer_bps: "50"
```

Keys go in `credentials.toml` under `[paper.<alias>]` (or
`SIGNALQUARRY_PAPER_KEY_ID` / `SIGNALQUARRY_PAPER_SECRET_KEY`). Use one paper
account per deployment.

## Commands

| Command | Who | What |
|---|---|---|
| `sqy paper preflight --alias A` | anyone | Read-only readiness checks. |
| `sqy paper dry-run --alias A` | anyone | Decide and size the next session; submits nothing. |
| `sqy paper arm --alias A --reason ...` | **human** | Type the alias to confirm; binds the frozen configuration and account. |
| `sqy paper run-once --alias A` | scheduler | Reconcile, decide, submit market-on-open orders. Idempotent. |
| `sqy paper status --alias A` | anyone | Journal state, sessions, pending orders. |
| `sqy paper reconcile --alias A` | anyone | Finalize journaled orders and check positions. |
| `sqy paper halt --alias A --reason ...` | anyone | Stop and cancel open orders; a human re-arms. |
| `sqy paper schedule --alias A --target systemd` | anyone | Write scheduler templates for review. |
| `sqy paper drift --alias A` | anyone | Replay sessions, measure slippage, shadow costs and dividends; gate G5. |
| `sqy paper run --alias A` | service manager | Run `run-once` until done, retrying while busy or unavailable. |
| `sqy paper backup --alias A` | anyone | Archive the journal and arm token with a hash manifest. |
| `sqy paper verify-continuity --alias A --backup F` | anyone | Prove the live journal only grew since a backup. |

## How a session runs

1. Take the deployment lease; verify the journal's hash chain.
2. Check the submission switch, the account, the arm token and the broker clock.
3. Load bars through the previous session.
4. Reconcile: finalize own orders from earlier runs, refuse unmanaged orders or
   positions, halt on drift that fills and splits cannot explain.
5. Decide and size with the backtest's own code; clip buys to settled cash.
6. Journal each order **before** submitting it, with a deterministic client
   order id, so a crash or network error is resolved by lookup, never by a
   duplicate order.

`run-once` exits 0 when done (or nothing to do), 75 to retry later, 69 when the
broker or data is unavailable, and 2 or 78 when a human is needed.

## Isolation

With `isolation: process` (the default) the decide-and-size step runs in a child
process that receives only the data it needs, with no broker or data credentials, an
empty config directory, `SIGNALQUARRY_OFFLINE=1`, and CPU and memory limits; a
strategy that exceeds them halts the deployment (`PAPER_STRATEGY_TIMEOUT`). This keeps
keys out of strategy code's process, but it is **not a sandbox**: code running as the
same OS user can read that user's files. Run paper deployments under a dedicated user.

## Scheduling

`sqy paper schedule` writes systemd (recommended, on a small VM), launchd,
cron or GitHub Actions (demo only) templates that call `run-once` every 5
minutes from 09:00 to 09:25 New York time. Review and install them yourself.
For a local scheduler, add `--notify-command /absolute/path/to/executable` to
`paper schedule` to include a failure hook in the generated `run-once` command.
The same option works on a direct `paper run-once` invocation. The executable is
called once for every non-zero exit, including retryable 69 and 75, with only
`HOME`, `PATH`, `LANG`, `SIGNALQUARRY_ALIAS`, and `SIGNALQUARRY_EXIT_CODE` in its
environment. It receives no broker credentials. It must get any notification
credentials from its own configuration. The hook has a 10-second limit; its
failure does not replace the original `run-once` exit code. Use the hook's own
deduplication if repeated retryable exits should produce one alert.
The systemd target also writes a post-close timer (16:30 New York, weekdays) that runs
`sqy commit create --alias A` to commit to the journal head, stamped with OpenTimestamps
when the `ots` client is installed, and a weekly timer that runs `ots upgrade` on pending
proofs. Allow egress to the OpenTimestamps calendars for those two units.

## Drift and the forward gate (G5)

`sqy paper drift` replays every journaled session through the engine with the
inputs recorded that day and requires the same decision and orders (data that the
provider has since revised is reported as unverifiable). It measures each fill
against that session's open (adverse slippage in basis points) and keeps a shadow
ledger: the model's costs and the dividends Alpaca paper does not credit, applied to
the broker's equity. **G5** passes with at least 20 clean sessions (parity, orders
final, no halt), no parity mismatch and mean adverse slippage within
`guards.max_mean_slippage_bps` (default 25). With G5 passed for the current frozen
configuration and a prior claim of `walk_forward` or better, exports report
`paper_forward`.

## A live feed for published strategies

Non-commercial families whose publication policy sets `live_feed: allow` can publish
a sanitized snapshot of a paper account: `sqy perf publish --alias A --out
site/public/feeds/paper` writes `A/latest.json` (`signalquarry-public-performance/v1`,
identified by its `snapshot_hash` and chained to the file it replaces through
`previous_snapshot_hash`). It carries equity history, positions and fills
without any account or order identifiers; figures that would need cash-flow data are
marked unavailable. Commercial families never get a live feed; they publish lagged
exports only.

`sqy perf capture --alias A` records the same snapshot privately under
`paper/A/captures/`, chained by `previous_snapshot_hash`; no publication policy is
needed because nothing leaves the project. The NDA export (`sqy evidence export --tier
nda`) carries the captures and the paper journal, with broker order ids withheld and the
journal head and length kept so a recipient can match published commitments.

## Options deployments

A deployment of an `options_single_leg` strategy polls instead of running once before
the open: `sqy paper run-once` performs one poll, and `sqy paper schedule` writes
per-minute templates for the regular session (systemd, launchd or cron; not GitHub
Actions). Each poll reconciles its own limit orders (cancelling entries left unfilled
after `options.cancel_after_seconds`), applies assignment and expiration activities to
the wheel state machine, checks broker positions against the wheels (a leg awaiting its
expiration activity is retried until the close of the next weekday; anything else halts),
then decides once per underlying with the same context and resolver as the simulator.
Every entry plan must satisfy the plan invariants (strike side, DTE window, limit at the
bid floored to the tick, collateral) before it is journaled and submitted. Quotes come
from Alpaca's data API (the stock's latest quote and the option chain snapshots).
`sqy perf capture` and `sqy perf publish` work for options deployments too (positions
include the open leg and any assigned shares); `sqy paper reconcile` is equity-only.

`sqy paper drift` evaluates **G5 for options** from the journal alone: option quotes are
not journaled, so decisions cannot be replayed as they are for equities (the report says
`decision_replay: not_available_for_options`). A session is clean when it was not halted
and every order it placed reached a final state (plan invariants are enforced before any
order is journaled). G5 passes with at least 20 clean sessions **and** at least one
expiration or assignment applied from broker activities. Model costs are the
per-contract fees.

## Known differences from the backtest

- Buys are clipped pre-open at a buffered prior close; the simulator clips at
  the actual open. They differ only when cash binds.
- Alpaca paper charges no commissions and omits dividends.
- Market-on-open orders need whole shares.

<!-- signalquarry:begin project/intro v1 -->
# Working in this project (for people and coding agents)

This project uses SignalQuarry to research trading strategies honestly. The
framework owns data, time, fills and the evidence record; you write strategy
logic only. Every `sqy` command prints one JSON envelope on stdout when not in a
terminal (or with `--json`). Read `status`, `reason_codes` and `next_actions`.

Sections between `signalquarry:begin`/`end` markers are refreshed by
`sqy init --upgrade-agents-md`; keep your own notes outside them.
<!-- signalquarry:end project/intro -->

<!-- signalquarry:begin project/claim-ladder v1 -->
## Claim ladder — say only what the evidence level allows

| claim_level | You may say |
|---|---|
| `none` | "The strategy runs." Nothing about performance. |
| `in_sample` | "It performed X on the data it was developed on." Not predictive. |
| `walk_forward` | "It held up out of sample in walk-forward tests." |
| `holdout_passed` | "It passed the sealed holdout once." |
| `paper_forward` | "It traded forward on a paper account for N sessions." |

Always state `evidence.grade` too. Synthetic data proves nothing about markets.
<!-- signalquarry:end project/claim-ladder -->

<!-- signalquarry:begin project/golden-path v2 -->
## Golden path

1. Write the idea in `strategy.yaml` → `hypothesis` (statement and falsification).
2. Edit `strategy.py`: one pure `decide(ctx, params)` function.
3. `sqy check` — conformance, determinism and look-ahead checks must pass
   (`--parity` also replays 60 sessions through the paper kernel; CI runs it).
4. `sqy data fetch --strategy <id>` — records a dataset manifest (needs Alpaca keys;
   demo projects use synthetic data and skip this).
5. `sqy backtest --strategy <id>` — exploratory run (claim at most `in_sample`).
6. `sqy spec freeze --strategy <id>` — freezes the configuration and seals the
   holdout (the most recent months). Commit before freezing.
7. `sqy evaluate --strategy <id>` — walk-forward and stress gates (G1–G3).
8. Only if G1–G3 pass: `sqy evaluate --strategy <id> --holdout` — **once per family, ever**.
9. `sqy report --strategy <id>` — writes `report.md` with the claim level stated first.
10. Paper forward test (after the evidence, on a dedicated Alpaca **paper** account):
    `sqy paper preflight --alias <a>` and `sqy paper dry-run --alias <a>` are safe to run.
    A human reviews, sets `submission: enabled` in `paper/<a>.paper.yaml` and runs
    `sqy paper arm`. After that, `sqy paper run-once` (scheduled with
    `sqy paper schedule`) is idempotent; `sqy paper status` shows the journal.

Every new configuration you evaluate on real data is a recorded trial and makes
the deflated Sharpe bar higher. Try fewer, better-reasoned ideas.
<!-- signalquarry:end project/golden-path -->

<!-- signalquarry:begin project/authoring-rules v2 -->
## Authoring rules

- `decide(ctx, params) -> Decision` must be pure: no clock, randomness, files,
  network or global state. Allowed imports: `signalquarry.sdk`, `numpy`,
  `pydantic`, `math`, `decimal`, `statistics`, `collections`, `dataclasses`,
  `enum`, `functools`, `itertools`, `operator`, `typing`, and this package.
- `ctx.bars(symbol)` holds exactly `lookback` completed sessions; `[-1]` is the
  last completed session. There is no "today" bar.
- Return `Decision.target({symbol: weight}, "CODE")`, `Decision.hold("CODE")`
  or `Decision.unavailable("CODE")`. Weights are fractions of equity, sum ≤ 1,
  long only. Declare every reason code in `strategy.yaml`.
- Everything that affects results (symbols, costs, account, parameters) lives
  in `strategy.yaml`, not in code.
- Options strategies (`kind: options_single_leg`) use `@options_strategy` and return
  `OD.open(sell_put(u, dte=(lo, hi), strike=Strike.otm(f)), "CODE")` (or `sell_call`),
  `OD.close(u, "CODE")` or `OD.hold("CODE")`: selectors, never contract symbols. The
  engine owns the wheel state (`ctx.wheel(u).state`) and open legs (`ctx.leg(u)`).
  Options results are `low_evidence_options`, have no holdout step, and reach
  `paper_forward` only through a paper forward test (G5).
<!-- signalquarry:end project/authoring-rules -->

<!-- signalquarry:begin project/stop-conditions v2 -->
## Stop conditions

- Exit status 2 (`blocked`): stop the research workflow and report the reason codes.
  After a blocked evaluation, `sqy report --strategy <id>` may render the existing
  evidence before you stop. Do not retry evaluation, access holdout, change the
  configuration to evade the failure, or work around a gate.
- Exit status 75 (`busy`): retry later (lease held, outside the window, data not ready).
- Exit status 78 (`disabled`): a human decision is needed (not armed, halted, submission disabled).
- Never loosen gates, edit `evidence/`, or re-run a sealed holdout.
- If the trial budget is exhausted, stop and ask a human.
<!-- signalquarry:end project/stop-conditions -->

<!-- signalquarry:begin project/never v1 -->
## Never

- Commit market data, credentials or `.signalquarry/`.
- Run `sqy paper arm`, set `submission: enabled`, or edit `paper/<alias>/` files — humans only.
- Touch live trading. SignalQuarry has no live path; do not add one.
- Place orders through other tools on an account a SignalQuarry deployment uses.

`sqy explain <CODE>` explains any reason code. `sqy doctor` checks the environment.
<!-- signalquarry:end project/never -->

<!-- signalquarry:begin project/files v2 -->
## Files

- `signalquarry.toml` — project settings (data provider, strategy modules)
- `src/{{package}}/<strategy>/strategy.py` and `strategy.yaml` — one strategy each
- `tests/test_conformance.py` — runs `sqy check` in your test suite
- `evidence/` — trial ledger and commitments (engine-written; never edit)
- `data/manifests/` — dataset hashes (no prices)
- `data/universe/snapshots/` — hash-only prospective asset-list observations; raw pages stay in the local cache
- `paper/<alias>.paper.yaml` — a paper deployment; `paper/<alias>/journal.jsonl` is its
  hash-chained record (engine-written; never edit)
<!-- signalquarry:end project/files -->

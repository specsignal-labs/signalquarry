# SignalQuarry framework: architecture

Status: accepted design for 0.1 → 1.0. Decisions are recorded as ADRs in
`docs/adr/`. When a decision changes, update the ADR and this document.

## 1. Framework design

**Principles**
- **Honest by construction.** The engine owns time, data, fills and the ledger. A strategy only maps a truncated context to a decision.
- **One `step()`, two clocks.** Backtest and paper run the same step with a different clock and broker.
- **Declare, hash, freeze.** Everything that affects results is declared in `strategy.yaml`, hashed and frozen.
- **One obvious way.** A small public surface, a reason code for every error, one JSON envelope per command.
- **Fail closed.**
- **Local-first.** Users bring their own keys. No data redistribution, no telemetry, no hosted component.
- **Solo-maintainable.** Four runtime dependencies, a stdlib CLI, and about 300 lines of in-house statistics.

**Non-goals (up to 1.0)**
- Live trading. No live URL exists anywhere, and brokers must declare `paper_only`.
- Intraday equities, crypto, futures, multi-leg options (these come in 1.x).
- ML feature stores, optimizers, a hosted cloud, a GUI.
- Generic order and account tools (Alpaca's MCP and CLI already do this).
- Windows (the paper kernel relies on POSIX `flock`).

**Layout and layering** (enforced by import-linter):
```
src/signalquarry/
  sdk/       authoring API: strategy, Params, Field, Ctx, Decision, ta, options   (stable)
  api/       facade: check/backtest/evaluate/data/paper_* -> Envelope            (provisional)
  plugins/   protocols: provider, broker, cost/fill model, gate, report section (provisional)
  testing/   pytest plugin: conformance(), FakeBroker, fixtures
  cli/       argparse adapter over api; command registry; exit codes
  schemas/v1/  templates/
  _internal/{canonical, contracts, data, project, engine, validation, evidence, publish, paper}
```

Dependency rules:
- **Layer order:** cli → api → {paper | evidence | publish} → validation → engine → {data | project} → {sdk | plugins} → contracts → canonical.
- **No network in pure code:** `sdk`, `engine`, `validation`, `contracts` and `canonical` must not import network, subprocess or paper modules.
- **Stdlib-only verifiers:** `canonical` and `evidence.verify` use the stdlib only, so the showcase can vendor them.
- **Network allowlist:** only `data.providers.*`, `paper.brokers.*` and `publish` may use `urllib.request`.
- The future MCP server wraps `api`, so no logic is duplicated.

**Stability in 0.x**
- **Additive-only changes:** `sdk`, `strategy.yaml` v1, CLI names and flags, envelope `signalquarry.cli/v1`, exit codes, reason codes (never renamed or removed), and the ledger, seal, manifest and bundle formats.
- **Breaking one of these** requires a new schema version, a reader for the old version kept for at least 2 minors (ledger readers forever), and `sqy migrate`.
- **`api` and `plugins`** may break after one minor of deprecation warnings.
- **`_internal`** has no guarantees.
- A griffe API snapshot in CI fails on unlabeled public-surface changes.

**Authoring API.** One pure function plus pydantic params, and no order management. This is the shape coding agents write most reliably.
```python
from decimal import Decimal
from signalquarry.sdk import Ctx, Decision, Field, Params, strategy, ta

class P(Params):
    symbol: str = "SPY"
    period: int = Field(200, ge=20, le=400)       # bounds double as the sweep space
    weight: Decimal = Field(Decimal("0.95"), ge=0, le=1)

@strategy(params=P, lookback=lambda p: p.period)
def decide(ctx: Ctx, p: P) -> Decision:
    close = ctx.bars(p.symbol).close             # read-only; [-1] = last COMPLETED session
    if close[-1] > ta.sma(close, p.period)[-1]:
        return Decision.target({p.symbol: p.weight}, "PRICE_ABOVE_SMA")
    return Decision.target({}, "PRICE_BELOW_SMA")
```

- **Options (1.0)** use `@options_strategy` with an `OptionsCtx`:
  - `decide` returns `D.open(sell_put(sym, dte=(21,45), strike=Strike.otm(f)), code)`, `D.close(leg, code)` or `D.hold(code)`;
  - the engine owns the wheel state machine (`ctx.wheel(sym)`) and the managed legs (`ctx.leg(sym).captured_fraction`);
  - strategies emit selectors, never OCC symbols.
- **`strategy.yaml` (`signalquarry.strategy/v1`):**
  - id, family, version, kind
  - `hypothesis{statement, falsification}`
  - `data{symbols, fields, feed, price_basis, max_staleness_sessions}`
  - `account{model, initial_cash}`, `execution{model, sizing, costs}`
  - frozen params, declared reason codes, limits
  - `evaluation{holdout, walk_forward, trial_budget}`, benchmark
- **Registration:** `signalquarry.toml` lists strategy modules (`[strategies] modules`); the registry finds the decorated function and the YAML next to it. This works without installing the project, and the lab's `tools/new_family.py` maintains the list. (Entry points were the first design; the explicit list replaced them so one file is the source of truth.)
- **Boilerplate the engine absorbs:** warm-up, staleness, missing bars, unchanged targets, identity checks and rounding. These take about 70 of the 117 lines in today's `sma_trend_demo.py`.
- **State:** canonical JSON, at most 16 KB, persisted identically on hold and target.
- **Reason codes:** undeclared codes fail with `REASON_CODE_UNDECLARED` (exit 65).

**Look-ahead is impossible by structure**
1. The context builder exposes read-only arrays truncated at `available_at ≤ cutoff`. `decide` never sees the dataset, the clock or any I/O.
2. Prices are adjusted point-in-time: raw bars plus corporate actions, back-adjusted only with ex-dates ≤ the session. Alpaca's `adjustment=all` would leak future dividends into past prices.
3. Holdout rows are clipped out before the engine starts.
4. `check` enforces an AST import policy: `sdk`, numpy, pure stdlib and the user's own package are allowed. `now`/`today`, `time`, `random`, `os`, `open` and mutated globals are rejected.
5. A mutation test perturbs or drops every bar after the cutoff and requires identical decisions and state.

**Engine**
- **Session phases:** PRE_OPEN (decide; cutoff 09:00 ET, bars through D−1) → OPEN (fills) → CLOSE (marks, expiries) → SETTLE.
- **Fixed event order:** splits → ex-dividends → decide → fills → marks → option expiry/assignment → settlement and payable dividends.
- **Clocks:** `HistoricalClock` walks Alpaca's `/v2/calendar`; `WallClock` gives one tick per `run-once`. Options poll every `poll_seconds` and on lifecycle events.
- **Context:** built in O(1) per session, so a whole run is O(N).
- **Execution:**
  - `next_open` is the default, using `opg` whole-share orders; the simulator fills at the official open. `next_close` uses `cls`.
  - Size at prior-close marks; sells before buys; sorted symbols; deterministic retry ordinals.
  - `cash` accounts spend settled cash only; `margin` accounts use buying power.
  - `execution_delay_sessions` exists for stress runs.
- **Costs:** bps + per-share + SEC/TAF fees. Fill models `fixed_bps` and `volume_cap`. Options: per-contract fee plus a required spread haircut.
- **Corporate actions:** splits adjust lots and pending orders; dividends accrue at the ex-date and are credited at the payable date; spin-offs and mergers stop the run with `CORPORATE_ACTION_UNSUPPORTED`.
- **Settlement:** T+2 before 2024-05-28, T+1 after; options T+1.
- **Options:**
  - The engine owns the wheel state machine.
  - The simulator assigns contracts ITM by ≥ $0.01 at expiry; early assignment is not modelled.
  - Paper transitions come from the account-activities sweep.
  - Sim and paper share one resolver: nearest expiry in the DTE window, then nearest OTM strike.
  - The `delta` strike rule is paper-only.
- **Determinism:** prices stored as int64 micro-units; a Decimal ledger (precision 28, HALF_EVEN); float64 signals; PCG64 randomness seeded from `configuration_hash`.
- **Performance targets** (CI benchmark warns at 1.5×, fails at 2×):

| Workload | Target |
|---|---|
| 10 years × 1 symbol | < 1 s |
| 10 years × 100 symbols | < 10 s |
| 10 years × 500 symbols | < 60 s, < 1 GB |
| 10-fold walk-forward | < 15 s |

**Data**
- **`MarketDataProvider` protocol:** capabilities, bars, corporate_actions, calendar, assets, option_contracts (incl. expired), option_bars, option_snapshot, normalize. Every call returns `RawPage{endpoint, params, bytes, sha256}`. Providers register under `signalquarry.providers`, and capabilities gate features.
- **Alpaca adapter** (seeded from `research_data/client.py`):
  - explicit `feed`, `adjustment=raw`, `asof`, `limit=10000`, at most 100 symbols per request;
  - a token bucket at 180/min (configurable), honoring `Retry-After`;
  - **SIP by default for both backtest and paper.** At the 09:00 cutoff, D−1 bars are more than 15 minutes old, so free plans can use SIP and both modes share one feed;
  - corporate actions with payable-date enrichment, ported from `collector_v4`;
  - expired option contracts are probed in Phase 7 (`sqy data probe options-coverage`). If they're missing, the simulator is disabled and `sqy data record options` snapshots chains going forward.
- **Credentials:** `APCA_*` environment variables, or a 0600 `credentials.toml` with separate `data` and `paper:<alias>` profiles.
- **Cache (Parquet; pyarrow is a core dependency):**
  - Raw pages are stored as gzip JSON, content-addressed by sha256, and are the source of truth.
  - Per-symbol Parquet files are derived; their identity is a canonical row hash.
  - The cache lives in `$SIGNALQUARRY_CACHE_DIR` (default `~/.cache/signalquarry`), never inside the repo.
- **`DatasetManifestV1`:** committed under `data/manifests/`; it contains no prices and is marked `redistributable:false`. Fetches are incremental with a 5-session overlap, and a provider revision creates a new manifest.
- **Synthetic provider:** seeded regime-switching `SYN_*` symbols, split and dividend fixtures, and Black–Scholes chains. Its grade is `synthetic`. `SIGNALQUARRY_OFFLINE=1` forbids network access.

**Validation and evidence**
- **Hashes:**
  - `configuration_hash` = canonical(params, data/execution/account/limits/codes, the strategy's `code_tree_hash`, framework major.minor).
  - `freeze_hash` adds the hypothesis, gates and holdout policy.
  - A dirty working tree caps the claim at `in_sample`.
- **Trial ledger (`evidence/trials.jsonl`):**
  - Appended by the engine, once per distinct (configuration, dataset, window), and hash-chained.
  - Each line records n_obs, Sharpe, skew, kurtosis and `returns_sha256`; the return series are stored per run for PBO.
  - Trial budget: a warning at 80%; runs are blocked at 100% until `sqy trials extend --reason`, which is itself a ledger entry.
  - CI rejects a ledger that shrinks or whose chain breaks.
- **Holdout:**
  - `spec freeze` writes `holdout.seal.json` covering the last 12 months, or from a declared model training cutoff if earlier.
  - Every command clips data at the seal, except a single `evaluate --holdout --freeze <hash>` once the walk-forward gates have passed.
  - Reuse returns `HOLDOUT_REUSED` and caps the claim at `walk_forward` for good.
  - This is a commitment protocol; it becomes tamper-evident once anchored with OpenTimestamps.

**Default gates.** Projects may make them stricter; loosening them is recorded as `gates: custom` and shown on every report.

| Gate | Criterion | Unlocks |
|---|---|---|
| G0 integrity | manifest verified, `check` passes, ledger intact | `in_sample` |
| G1 sample | ≥ 5 years of history, ≥ 30 round trips or rebalances | prerequisite |
| G2 walk-forward | ≥ 6 out-of-sample folds; OOS PSR(0) ≥ 0.95; DSR ≥ 0.95 using the project-wide trial count | `walk_forward` (together with G3) |
| G3 stress | Sharpe at costs ×2 and at a one-session delay each ≥ 0.5 × base; net return > 0 | |
| G4 holdout | Sharpe > 0; max drawdown ≤ 1.5 × walk-forward max drawdown | `holdout_passed` |
| G5 forward | ≥ 20 clean paper sessions within drift tolerance | `paper_forward` |

- **Options:** no G4, always graded `low_evidence_options`, and capped at `walk_forward` until G5.
- **Statistics:** written in-house (PSR, DSR, MinTRL, block bootstrap, and CSCV PBO in 0.2) using numpy + `statistics.NormalDist`, with no SciPy. Goldens reproduce the published numeric examples.
- **Outputs:** each run writes `result.json` (the single source of numbers), `report.md` rendered from it, and in-house SVGs (no matplotlib). `evidence export --tier public|nda` produces an `EvidenceBundleV1` `.tar.gz` with a stdlib verifier.

**Paper kernel**
- `RunLease` (flock) on every state-changing command.
- A hash-chained JSONL journal is authoritative (SQLite is dropped) and is verified once per lease.
- `ArmTokenV1` binds alias, `freeze_hash`, account-id hash, manifest, ledger head and a validity window.
- **Reconcile before decide:**
  - own-prefix orders;
  - account activities (OPASN/OPEXP/OPEXC, splits, dividends) with a grace window;
  - halt only on drift those can't explain.
- **Guards:** clock skew, daily drawdown, unmanaged positions or orders, the market-phase window.
- Client order IDs `sq-<alias6>-<intent10>-<retry>`; `status`, `halt`, `backup`.
- 0.2 adds a shadow ledger that applies model costs and the dividends Alpaca paper omits.
- **Scheduling:**
  - No daemon in 0.1; `run-once` is idempotent and refuses to run outside its window.
  - `sqy paper schedule --target systemd|launchd|cron|github-actions` writes templates that retry every 5 minutes within 09:00–09:25 ET, stop on exit 2, skip on 75 and back off on 69.
  - systemd on a small VM is recommended. GitHub Actions is labelled demo-only (concurrency group + compare-and-swap push to `paper-state`).
- **Credentials:**
  - the paper origin is hard-coded, and CI forbids any non-paper Alpaca host;
  - expected account-id hash;
  - a deny-list of other deployments' key hashes;
  - 0600 file checks;
  - `APCA_*` values redacted from output.
- **Isolation (0.2):** `decide` runs in a spawned child with a scrubbed environment and rlimits, before credentials load. The docs say honestly that strategy code cannot read the keys, and that this is not a sandbox.

**Extensibility.** Entry-point groups, each with an `api_version`: `signalquarry.strategies`, `.providers`, `.brokers` (must declare `paper_only=True`), `.cost_models`, `.fill_models`, `.gates` (add-only), `.report_sections`, `.publish_projections`.

Deliberately not pluggable: the context builder and clock, ledger and seal writers, canonical hashing, the claim ladder and minimum gates, kernel safety, the envelope and exit codes, and any live-broker hook.

**Repo, packaging, release, docs, governance**
- **Layout:** `src/`, `tests/{unit,property,golden,conformance,parity,lookahead,kernel_faults,e2e}`, `evals/`, `examples/{sma_trend,wheel_reference}`, `docs/`, `tools/{leak_scan,check_headers}.py`. Root files: AGENTS.md, CLAUDE.md, CONTRIBUTING, GOVERNANCE, SECURITY, CODE_OF_CONDUCT, LICENSE, NOTICE, TRADEMARKS, CHANGELOG, REUSE.toml, uv.lock.
- **Packaging:**
  - one pure wheel (hatchling) exposing `signalquarry` and `sqy`;
  - Python 3.12–3.14 (the three latest CPython minors);
  - runtime dependencies are only `pydantic`, `numpy`, `pyarrow` and `pyyaml` (no pandas, no HTTP library);
  - extras `[pandas]`, `[keyring]`, `[ots]`;
  - adding a runtime dependency requires an ADR.
- **CI:**
  - Ubuntu + macOS arm64 × 3.12/3.13/3.14, plus a lowest-direct resolution job;
  - fresh-venv wheel smoke test (`init --demo → check → backtest → evaluate`);
  - import-linter, ruff, pyright (strict on sdk/api/plugins), leak scan, gitleaks, API snapshot, `mkdocs build --strict`, REUSE lint, benchmarks.
- **Release:**
  - signed tags → `uv build` → PyPI Trusted Publishing from a reviewer-gated environment;
  - PEP 740 attestations and build provenance;
  - a CycloneDX SBOM, plus the schemas and verifier with checksums, attached to the GitHub Release;
  - Keep-a-Changelog; after 1.0, deprecations last at least 2 minors or 6 months.
- **Docs:** MkDocs Material + mkdocstrings. The toolchain is Python-only, and the content is plain Markdown so the generator can be swapped. `llms.txt` and `llms-full.txt` are generated from the docs, CLI catalog and schemas; `sqy docs --llms` serves them offline.
- **Governance:**
  - BDFL; DCO, no CLA;
  - CODEOWNERS on validation, paper, schemas, templates and the leak scan;
  - issue templates (the bug template asks for `sqy doctor --json`); a PR checklist including AI-assistance disclosure;
  - GitHub private vulnerability reporting; **no telemetry**; SPDX headers enforced.
  - `init` templates are CC0/MIT-0, so user strategies carry no license obligations.

**Testing**
- **Hypothesis properties:** cash + marks = equity; no negative settled cash; splits preserve value; canonical round-trip; resolver bounds; state-machine legality; reconcile idempotence; deterministic client IDs.
- **Goldens:** synthetic-symbol engine fixtures, canonical v2 vectors, the published statistics examples, envelope snapshots.
- **Conformance meta-tests** must catch deliberately leaky strategies and an injected off-by-one in the context builder.
- **Backtest↔paper parity:** 60 synthetic sessions against the FakeBroker must produce identical orders, positions and cash.
- **Kernel faults:** duplicate submit, orphan after a fill, a 5xx on submit, clock skew, unconfirmed cancel, delayed assignment, lease contention.
- **Coverage:** ≥ 85% overall; ≥ 95% line and branch on the engine, validation, paper kernel and canonical.

## 2. Agent surface

**Commands by release:**
- **0.1:** `init [--demo]`, `doctor`, `check`, `data fetch|verify|ls`, `spec freeze`, `backtest`, `evaluate [--walk-forward|--stress|--holdout]`, `holdout seal|status`, `trials ls|show|extend`, `report`, `paper preflight|dry-run|arm|run-once|status|reconcile|halt|schedule`, `schema`, `explain <CODE>`, `commands`, `version`, `docs --llms`.
- **0.2:** `sweep`, `check --parity`, `paper run|backup|verify-continuity`, `perf capture|publish`, `evidence export|verify`, `commit create|verify`.
- **1.0:** options paths in every command, plus `init --kind options`.

**CLI:** stdlib argparse with a declarative command registry, which drives `commands`, the docs and the error handling.

**Envelope** (`signalquarry.cli/v1`): `{schema, command, framework_version, run_id, duration_ms, status, reason_codes, summary, data, metrics, evidence{grade, claim_level, holdout, trial{family,index,count,project_count,ledger_head}}, artifacts[{path,sha256,kind}], warnings, next_actions[{command,why}]}`.
- `data` is capped at 64 KB; anything larger becomes an artifact (`--detail full` lifts the cap).
- Progress goes to stderr as NDJSON. Stdout carries exactly one envelope.

**Exit codes:**

| Status | Exit | Notes |
|---|---|---|
| ok | 0 | |
| error | 1 | includes a `trace_id` |
| blocked | 2 | also covers gate failure, holdout reuse and trial budget |
| usage | 64 | |
| invalid | 65 | spec or data |
| provider unavailable | 69 | |
| busy | 75 | |
| disabled / not armed | 78 | |

**`init` layout:** `AGENTS.md CLAUDE.md pyproject.toml (signalquarry pin) signalquarry.toml (strategy modules) src/<pkg>/<strategy>/{strategy.py,strategy.yaml} tests/test_conformance.py data/manifests/ evidence/{trials.jsonl,holdout.seal.json,commitments/} paper/paper.yaml (submission disabled) .github/workflows/{check.yml (offline, synthetic, ledger append-only check), paper.yml (opt-in)}`. The cache lives outside the repo.

**AGENTS.md** (at most 150 lines, in versioned blocks that `sqy init --upgrade-agents-md` updates):
1. The claim ladder and the wording each level permits.
2. The golden-path commands.
3. Authoring rules.
4. How to read envelopes, `next_actions` and `explain`.
5. Stop conditions: exit 2, `HOLDOUT_REUSED`, trial budget, failed gates. Write a new hypothesis; never relax gates.
6. Never: edit `evidence/`, commit data, arm paper, touch live endpoints.
7. Use Alpaca's MCP or CLI for read-only lookups.
8. A file map.

`CLAUDE.md` contains `@AGENTS.md`. The guardrail text from the predecessor project's agent guidance is reused, rewritten for users.

**Agent eval (`evals/`):**
- YAML tasks, including **temptation tasks** ("make it pass the gates").
- Headless `claude -p` and `codex exec` runs in a fresh container from the PyPI install, using synthetic data plus a recorded Alpaca cassette.
- Scored on artifacts: an envelope was reached; the ledger is intact and monotonic; conformance passes; the count of 64/65 errors; wall time; tokens.
- Cadence: per release candidate and weekly. Every PR runs a deterministic command-script replay.
- Bar: 3/3 runs reach `evaluate` in ≤ 15 min, with zero ledger or gate tampering.

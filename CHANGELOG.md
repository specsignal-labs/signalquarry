# Changelog

All notable changes are recorded here ([Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/), with 0.x minors allowed to
change provisional surfaces after deprecation).

## [Unreleased] — 0.2.0

### Added
- An internal, pure data-quality assessment of a daily dataset: calendar coverage, gaps,
  OHLC consistency, non-positive prices, zero volume, stale closes and large moves that no
  recorded split explains. Its report holds dates and counts only, never a price, volume or
  return. No command uses it yet.
- Broader validation mutation coverage with independent reference cases for equity metrics,
  walk-forward and holdout gates, evidence history, and strategy conformance.
- Engine mutation coverage with hand-worked order-planning, affordability, settlement, and
  pre-open paper-parity cases.
- Options-simulator mutation coverage with short wheel, quote, expiry-boundary, and
  action-refusal reference cases.
- Formula-search mutation coverage with independent references for the significance arithmetic,
  configuration and input contracts, report assembly, and genetic operators.
- Factor-evaluation mutation coverage: forward-return labels against a day-by-day reference with
  corporate-action boundaries, rank-IC/quintile/turnover/capacity diagnostics against hand-worked
  and per-day references, and score-panel timing and universe-identity checks.
- Point-in-time universe and research-panel mutation coverage: cutoff, listing-age, price and
  percentile boundaries, classification and asset snapshot contracts, the private classification
  cache, and panel integrity and windowing rules.
- Calendar and factor-integrity mutation coverage: Easter and weekday arithmetic against independent
  references across the tracked range, the exact report of the synthetic factor checks, and the exact
  content of recorded factor trials.
- Evidence mutation coverage: every defect code of the standalone bundle verifier and its command
  line, and run identity, artifact layout, cleanup after failure and result documents.
- Paper-state mutation coverage: the arm token's issue/check order and boundaries, journal chaining and
  durability, the single-writer lease, the isolated strategy child's exact environment and failure
  reporting, and the scheduler templates' times, commands and quoting.
- Paper-runner mutation coverage with exact snapshot, drift, and deterministic
  order-intent reference cases.
- An internal, hash-chained factor-trial log with separate project/family configuration counts; recording does not certify provenance or change strategy trial budgets.
- A versioned internal factor-expression interpreter for a bounded, trailing-only AST grammar.
  Rolling mean, standard deviation and correlation use centred windows (grammar version 2), so
  constant windows have exactly zero dispersion instead of rounding noise.
- A deterministic genetic formula-search core over synthetic training inputs, with family-budget
  checks, complexity and library-correlation penalties, and deflated-t/BH diagnostics. It writes no
  trial ledger, issues no evidence grade, and does not access a holdout.
- Optional local `sqy-mcp` adapter over the command API for coding agents;
  human-only paper arming and trial-budget extension remain terminal-only.
- Internal, dataset-identity keyed Parquet research panels derived from verified cached pages,
  with explicit pre-decision windows, missing-bar masks, and a load-once path for factor research.
- `sqy universe build` creates a hashed, replay-verifiable common-stock membership manifest from
  dated asset and caller-supplied classification snapshots, explicit price/listing-age/liquidity
  filters, and 20 completed prior sessions. It fails closed when required as-of inputs are missing.
- Disk-backed equity backtest decisions and fills in the `backtest` API, driven by a pure
  per-session engine iterator. Run artifacts are written incrementally; failed streams remove
  partial run directories. Artifact formats and ledger hashes are unchanged.
- Developer tooling for private, read-only Alpaca HTTP cassette capture and strict offline replay;
  response fixtures are source-labeled, credential-redacted, hash-verified, and stored outside Git worktrees.
- Synthetic-only asset-keyed pre-open planning, shared by the asset-keyed
  backtest, with lifecycle parity vectors for renames, ticker reuse, splits,
  removals, mergers and late-event failures. Provider ingestion and paper
  planning remain separate gates.
- The coding-agent guide documents the plain-language idea → strategy files → `sqy` check/backtest
  workflow and distinguishes normal authoring from the optional agent-evaluation harness.
- Options core (toward 1.0): OCC identities, quote rules, one contract resolver for simulation
  and paper, the wheel state machine, the options authoring API (`signalquarry.sdk.options`) and
  `kind: options_single_leg` specs. Equity configuration hashes are unchanged.
- Low-evidence options simulator (Black-Scholes chains, open/close checkpoints, expiry and
  assignment, fees and spread haircut) behind `backtest`, `evaluate` (cost stress only, no G4,
  claims capped at `walk_forward`), `sweep` and `check` (options look-ahead mutation test).
- Options paper runner: per-minute polls, limit orders with cancel-and-confirm, assignment and
  expiration activities, position checks with a lifecycle grace, plan invariants before every
  order; Alpaca options venue (paper orders and activities, data-API quotes and chain
  snapshots); per-minute schedule templates. Journals refuse reserved field names.
- `sqy init --kind options` (reference wheel template), `examples/wheel_reference/`, and
  `sqy data probe options-coverage` (expired contracts and their bars; warns
  `OPTIONS_HISTORY_UNAVAILABLE`). 1.0 gate test: plan invariants hold for every simulated
  reference-wheel entry.
- `sqy init --upgrade-agents-md`: AGENTS.md templates now carry versioned
  `signalquarry:begin/end` blocks; the upgrade refreshes them and keeps your own notes
  (`AGENTS_MD_UNMANAGED` when there are no blocks).
- `perf publish` chains each snapshot to the file it replaces (`previous_snapshot_hash`;
  `PERF_FEED_INVALID` when that file is not a snapshot).
- Public API snapshot (`tools/public_api.txt`, checked in CI and tests).
- Private paper-activity observations can be replayed and compared offline by
  hash; command output exposes only aggregate counts, with no inferred event terms.
- `sqy check --parity`: backtest vs paper kernel on a fake venue for equity and options
  strategies (the project CI template runs it).
- Type checking with pyright in CI (standard everywhere, strict on `signalquarry.sdk` and
  `signalquarry.api`);
  `@strategy` and `@options_strategy` are generic over the params class, so a typed
  `decide(ctx, p: MyParams)` checks cleanly.
- REUSE 3.3 compliance (`LICENSES/`, default annotation in `REUSE.toml`), checked in CI.
- gitleaks (v8.28.0) scans the full git history in CI.
- Paper broker plugins (`broker: plugin:<name>`, entry point `signalquarry.brokers`): paper_only and
  the full broker interface are enforced (`BROKER_NOT_PAPER_ONLY`, `BROKER_PLUGIN_INVALID`).
- `signalquarry.plugins` (provisional): entry-point gates (add-only, cap the claim when
  they fail) and report sections; `sqy doctor` lists them (ADR 0009).
- NDJSON progress on stderr for `data fetch`, `check`, `sweep` and `evaluate` (JSON mode or
  `SIGNALQUARRY_PROGRESS=1`).
- `volume_cap` fill model (`execution.fill`): fills capped at a share of session volume,
  remainder retried; specs without it keep their configuration hash.
- Sell-side regulatory fees in `execution.costs` (`sell_bps`, `sell_per_share`,
  `sell_per_order_max`), applied in backtests and paper model costs; hash-neutral when unset.
- `TRIAL_BUDGET_NEARLY_USED` warning once a family has used 80% of its trial budget.
- `evaluation.holdout.training_cutoff`: the holdout starts after a model's training cutoff
  when that is earlier than the last `months` (hash-neutral when unset).
- `evaluate` reports `oos.sharpe_annual_90` (block-bootstrap interval) and `report.md` states it.
- `sqy perf capture`: private, chained performance snapshots under `paper/<alias>/captures/`.
- Performance snapshots (`perf capture|publish`) for options deployments, including the open
  leg and assigned shares.
- G5 for options (`paper drift`): 20 clean journal sessions plus at least one expiration or
  assignment, so options claims can reach `paper_forward`.
- Agent eval task `options-put` (replayed in CI); the scorer no longer counts an options
  spec's absent holdout as loosening.
- Sandboxed headless agent evaluations now support GitHub Copilot CLI and Grok Build alongside
  Claude Code and Codex; each subprocess receives only its selected provider credentials.
- AGENTS.md authoring block (v2) covers options strategies; `sqy init --upgrade-agents-md`
  brings existing projects up to date.
- `paper schedule --target systemd` also writes a post-close journal-head commitment timer
  and a weekly `ots upgrade` timer.
- `sqy doctor` reports whether the OpenTimestamps client is installed (`OTS_CLIENT_MISSING`).
- `sqy data record options`: record today's option chain (raw pages cached, hash-only record
  in `data/options/`), verified by `data verify` and listed by `data ls`.
- Weekly maintainer-only `live-data-smoke` workflow for read-only Alpaca bars, manifest
  verification, a sample backtest, and an options-history coverage report; without the
  dedicated data-only repository secrets, the run is explicitly skipped.
- Options backtests, evaluations and sweeps on real data use recorded chains where they exist
  (`OPTIONS_RECORDED_CHAINS_USED`), falling back to modelled prices elsewhere.

### Changed
- Factor trials are identified per family: recording the same configuration under a second family is that
  family's own trial in the default project evidence layout too (as it already was with `per_family`).
- Building each decision's per-symbol bar windows is about 26% faster on the 10y x 3,000-symbol
  diagnostic (184.7s to 135.9s on one arm64 laptop run; 10y x 500 symbols 28.0s to 20.2s). The four
  price fields are adjusted in one block and the per-decision session window is shared, with
  identical values, read-only arrays and ledger hashes.

### Fixed
- Alpaca data fetch now requests all corporate-action types and blocks unsupported events or splits with a new symbol before building a dataset, so those events cannot silently become ordinary price gaps.
- Evidence verification now reports non-object JSONL records as `EVIDENCE_LOG_CORRUPT`
  instead of an internal error.
- A paper snapshot requested with zero recent fills now returns an empty list instead
  of all historical fills.
- Agent eval scoring now checks that strategy and paper commands target the requested IDs,
  parses paper submission and journal state as structured data, and treats malformed strategy
  specs as failed runs rather than crashing the scorer.
- Agent-evaluation containers keep task references and scoring code away from
  the unprivileged agent process; command scores use evaluator-owned records.
- Agent-authored strategy code now runs only under a separate unprivileged verifier UID,
  against a root-owned read-only project copy; verification cannot inherit provider credentials.
- Claude Code evaluation passes allowed tools as separate command arguments, matching the documented
  `--allowedTools` syntax.
- Agent-evaluation runs reject zero or negative repetition counts instead of reporting an empty run
  set as passing.
- Exported study results were always labelled `historical`: synthetic runs are now
  `diagnostic` and options runs `option_proxy`, with options costs in the assumptions.
- Exported options profiles say `asset_class: options` (from the spec, not the publication
  file's equity default) and keep the `low_evidence_options` grade.
- Credential values no longer appear in `repr()` of the data client or credentials, and
  every command scrubs loaded credential values from its output and error traces.
- Options strategies no longer seal a holdout at freeze: there is no holdout gate for
  options, so the seal only discarded scarce data (templates declare `months: 0`).
- `duration_ms` measures the whole command (it was usually 0); a `GATE_FAILED` evaluation
  suggests `report` and `explain GATE_FAILED`; the quickstart says the demo fails its gates.
- A fresh `--lab` project's CI failed at `pytest` (no tests collected). The drill family and
  every `tools/new_family.py` family now ship a conformance test; lab CI runs `check --parity`.
- The NDA export now includes the paper journals it promised (broker order ids withheld)
  and the performance captures.
- The 64 KB cap on envelope `data` is enforced: larger data moves to an artifact file
  (`DATA_MOVED_TO_ARTIFACT`); `--detail full` keeps it inline.
- An unexpected exception now yields one `error` envelope (`INTERNAL_ERROR`, `data.trace_id`)
  with the traceback on stderr, instead of a bare traceback; `SIGNALQUARRY_DEBUG=1` re-raises.
- The project template no longer declares an unused `signalquarry.strategies` entry point;
  strategies are registered in `signalquarry.toml` (design doc updated).
- Options `paper arm` applies pending fills, expirations and assignments before checking
  positions, so arming the morning after an expiry no longer reports
  `PAPER_OPTIONS_LIFECYCLE_PENDING`.
- `sqy evidence export --family F [--tier public|nda]`: an EvidenceBundleV1 under the family's
  default-deny `publication/<family>.publication.yaml` (commercial families: category tier,
  lagged weekly or monthly paper returns only; positions, fills and exposure withheld).
  Successive exports chain with `supersedes`; `sqy evidence verify --bundle DIR`.
- Per-family evidence layout (`[evidence] layout = "per_family"`) with a chained
  `evidence/project_index.jsonl`; the DSR denominator stays project-wide.
- `sqy commit create|reveal|verify`: salted commitments to frozen strategies and paper
  journal heads, stamped with OpenTimestamps when `signalquarry[ots]` is installed.
- Paper `isolation: process` (default): decide runs in a child process without credentials,
  offline, with CPU and memory limits (not a sandbox).
- `sqy paper drift`: journal replay parity, fill slippage, shadow ledger (model costs and
  uncredited dividends) and gate G5, which unlocks `paper_forward`.
- `sqy paper run` (retrying loop over run-once), `sqy paper backup` and
  `sqy paper verify-continuity` (detects a rolled-back journal on the host).
- `sqy sweep`: parameter grids (each real-data point is a trial, budget-checked up front)
  with the probability of backtest overfitting (CSCV PBO).
- `sqy perf publish`: a sanitized live paper-account feed (`signalquarry-public-performance/v1`)
  for non-commercial families with `live_feed: allow`.
- `sqy init --lab`: a multi-family strategy lab template (per-family ledgers, transfer drill,
  commit-scope and cross-family import checks, bundle leak scan, data room, guarded deploy).
- `strategies.src_dirs` accepts glob patterns.
- Showcase document schemas (`sqy schema showcase/...`): strategy profile, study result,
  study summary, forward record.

## [0.1.0] — unreleased (branch `release/0.1`)

### Added
- `sqy` CLI with one JSON envelope per command (`signalquarry.cli/v1`), exit codes and a
  reason-code registry; `version`, `doctor`, `commands`, `schema`, `explain`, `docs --llms`.
- Strategy authoring API (`signalquarry.sdk`): `@strategy`, `Params`, `Ctx`, `Decision`, `ta`.
- `strategy.yaml` v1, `sqy init [--demo]` project templates with `AGENTS.md`.
- Session engine: next-open execution, costs, point-in-time splits, payable-date dividends,
  T+2/T+1 settlement, cash and margin accounts.
- `sqy check`: import policy, contract, determinism and look-ahead checks.
- Alpaca market data (raw bars, corporate actions), content-addressed cache and dataset
  manifests; `sqy data fetch|verify|ls`; deterministic synthetic data.
- Evidence: hash-chained trial ledger, `spec freeze`, sealed holdout, PSR/DSR/MinTRL,
  walk-forward and stress gates, claim levels; `sqy evaluate`, `trials`, `holdout status`.
- `sqy report` (Markdown + SVG).
- Alpaca **paper** forward tests: `sqy paper preflight|dry-run|arm|run-once|status|reconcile|halt|schedule`
  with a run lease, hash-chained journal, arm tokens and reconcile-before-decide.
- Documentation site and agent-eval harness.

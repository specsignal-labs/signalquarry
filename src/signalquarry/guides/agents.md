# Working with coding agents

Every project has an `AGENTS.md` (and a `CLAUDE.md` that imports it) written for
agents. It covers the golden path, authoring rules, the claim ladder, stop
conditions and what agents must never do.

## From a plain-language idea to a checked strategy

The coding agent interprets the user's description and edits the project;
SignalQuarry does not execute natural-language instructions. Have the agent
first restate the idea as explicit entry, exit, sizing, universe, and risk rules.
It should ask about any material ambiguity instead of silently choosing a
rule. Record the hypothesis and how it could be falsified in `strategy.yaml`,
and put deterministic strategy logic in `strategy.py` as a pure
`decide(ctx, params)` function.

Then have the agent follow that project's `AGENTS.md` golden path: read
`sqy docs --llms --full`, run `sqy check`, use or fetch the configured dataset
(`sqy data fetch --strategy <id>` when needed and authorized), and run
`sqy backtest`. Demo data stays explicitly synthetic. A backtest is exploratory;
the agent must report the returned evidence grade and claim level, not infer
market validity from a successful run. Freezing, evaluation, holdout access,
and any paper workflow must follow the project's gates and human-only steps.

When the request is a question rather than a single run ("does the filter help?",
"is it better than just holding?"), have the agent answer it as a study: declare the
benchmark in `strategy.yaml`, write the hypothesis, the baselines, a few bounded
variants and one comparison rule in `studies/<id>/study.yaml`, run
`sqy study check` and `sqy study run`, and report the verdict it returns
(`supported`, `not_supported` or `insufficient`) with the benchmark numbers. The
project's `AGENTS.md` has this under "Research method". A study is in-sample: it
does not replace freezing and evaluation.

The `evals/` harness serves a different purpose: it runs a selected coding-agent
CLI against fresh demo projects and scores the artifacts the agent leaves
behind. It is for repeatable agent-behavior evaluation, not required to author
or backtest an ordinary strategy. It does not make `sqy` call a model or grant
an agent paper-trading authority.

## The envelope

Each command prints one JSON object on stdout (`schema: signalquarry.cli/v1`):
`status`, `reason_codes`, `summary`, `data`, `metrics`, `evidence`
(`grade`, `claim_level`, `holdout`, `trial`), `artifacts` (path, sha256, kind),
`warnings` and `next_actions` (command + why). `sqy commands` lists every
command, reason code and exit code; `sqy docs --llms` prints this documentation
for an agent's context.

For factor research, an agent can add one `@factor` function in a project module,
declare its hypothesis and parameters in a neighboring `factor.yaml`, and list
that module under `[factors] modules` in `signalquarry.toml`. `sqy factor ls`
returns stable configuration identities, and `sqy check --factor-id ID` runs
synthetic conformance. To inspect a registered factor on cached data, run
`sqy factor evaluate --factor ID --dataset-id DATASET --universe-manifest PATH`
and repeat `--universe-manifest` for each dated membership decision. The
command verifies and replays those manifests, then returns descriptive metrics
with `data.scope: "unverified"`. It does not record a trial, assign an evidence
grade, or access a holdout. Source completeness, full action history, ticker
continuity and delisting outcomes remain unverified, so the output is not
point-in-time evidence or a profitability claim.

Agents with local MCP support can run the optional `sqy-mcp` stdio adapter.
Its tools return the same envelopes as the CLI. Use an explicit project path
when the host launches outside the project. The adapter has no tool for paper
arming or trial-budget extension; `sqy_paper_run_once` still requires a human
arm and must be used only when the user asks for a paper session.

`data` is capped at 64 KB so an envelope fits in a context window. Anything larger is
written to `$SIGNALQUARRY_CACHE_DIR/envelopes/<run_id>.data.json` (listed in `artifacts`
with kind `data`, warning `DATA_MOVED_TO_ARTIFACT`); `--detail full` keeps it inline
(for example `sqy --detail full docs --llms --full`).

Long commands report progress on **stderr** as one JSON object per line
(`{"type": "progress", "stage": "fetch" | "check" | "sweep" | "evaluate", ...}`) in JSON
mode or with `SIGNALQUARRY_PROGRESS=1`. Progress is advisory; only the stdout envelope
counts.

| Exit | Status | Agent should |
|---|---|---|
| 0 | ok | continue |
| 1 | error | `INTERNAL_ERROR`: report `data.trace_id` and the stderr traceback, then stop |
| 2 | blocked | stop research and report the reason codes; after blocked evaluation, `sqy report` may render existing evidence before stopping; never work around a gate |
| 64 | usage | fix the command line |
| 65 | invalid | fix the spec, code or data named in the summary |
| 69 | unavailable | a provider, broker or credential is missing; tell the human |
| 75 | busy | retry later |
| 78 | disabled | a human decision is required |

## Human-only actions

`sqy trials extend`, `sqy paper arm`, setting `submission: enabled`, and any
edit under `evidence/` or `paper/<alias>/`. The framework records these and
refuses the ones it can detect (for example, `paper arm` needs an interactive
terminal).

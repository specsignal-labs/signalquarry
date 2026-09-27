<!-- signalquarry:begin lab/intro v1 -->
# Working in this lab (people and coding agents)

This repository holds strategy families that may be licensed or sold separately.
Everything that makes that possible depends on the rules below. Read the framework's
guide first: `sqy docs --llms --full`.

Sections between `signalquarry:begin`/`end` markers are refreshed by
`sqy init --upgrade-agents-md`; keep your own notes outside them.
<!-- signalquarry:end lab/intro -->

<!-- signalquarry:begin lab/families-are-islands v1 -->
## Families are islands

- One family per directory under `families/<family>/`: its code (`src/`), strategy
  specs, tests, evidence (`evidence/`), paper configs and `publication.yaml`.
- **Every commit touches at most one family** (`tools/check_commit_scope.py`).
- A family never imports another family's package (`tools/check_family_imports.py`).
  Shared code goes upstream into SignalQuarry, not sideways.
- New family: `python tools/new_family.py NAME`.
<!-- signalquarry:end lab/families-are-islands -->

<!-- signalquarry:begin lab/evidence v1 -->
## Evidence

- Every configuration evaluated on real data is a trial in the family's ledger; the
  deflated Sharpe ratio counts every family's trials (`evidence/project_index.jsonl`).
- `ideas/` holds hypotheses as text only. Never run experiments outside the ledger.
- Never edit `families/*/evidence/`, `evidence/` or `paper/*/journal.jsonl`.
- Freeze before any claim; the holdout opens once per family.
<!-- signalquarry:end lab/evidence -->

<!-- signalquarry:begin lab/publication v1 -->
## Publication

- `families/<family>/publication.yaml` is default-deny: category tier, no backtest
  results, weekly paper returns with a lag of at least 14 days, positions and fills
  withheld. Only a human changes it.
- Weekly routine (human): `sqy evidence export --family F`, then
  `python tools/bundle_leak_scan.py BUNDLE --family F`, then ingest into the showcase.
<!-- signalquarry:end lab/publication -->

<!-- signalquarry:begin lab/never v1 -->
## Never

- Arm paper trading, set `submission: enabled`, or touch any live endpoint.
- Commit data, credentials, salts, or anything under `deals/`.
- Use the paper account or keys of another deployment (one paper account per deployment).
<!-- signalquarry:end lab/never -->

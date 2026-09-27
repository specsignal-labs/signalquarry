# Strategy lab: design

Status: accepted. Decisions are recorded in `docs/adr/`.

## Purpose

The only home of strategies that may be licensed or sold. Every family must be
transferable with clean, verifiable provenance, while the deflated Sharpe ratio still
counts every trial made anywhere in the lab.

## Layout

```
signalquarry.toml   pyproject.toml   AGENTS.md
families/<family>/{src/<prefix>_<family>/<strategy>/{strategy.py,strategy.yaml}, tests/,
                   evidence/{trials,freezes,holdouts}.jsonl, publication.yaml, README.md}
evidence/project_index.jsonl     (chains every family ledger head)
evidence/commitments/<family>/   (commit-reveal records and OpenTimestamps proofs)
paper/<alias>.paper.yaml, paper/<alias>/journal.jsonl
tools/{check_commit_scope, check_family_imports, new_family, extract_family.sh,
       bundle_leak_scan, dataroom}.py
deploy/{deploy.py, config.toml, host-bootstrap.sh}
ideas/   deals/ (git-ignored except README)
```

## Why one repository with isolated families

One agent context, one lock file and one DSR denominator — while each family stays
sellable: no cross-family imports; at most one family per commit; per-family ledgers
chained into the project index; and a family's directory tree hash (the `c_code`
commitment) survives extraction unchanged.

## Evidence and publication

- Trials, freezes and holdout openings live in the family; the index proves the total
  trial count to a buyer without revealing other families' trials.
- Commitments at freeze (`sqy commit create`) and on journal heads; salts outside the
  repository (`$SIGNALQUARRY_CONFIG_DIR/salts`), backed up in two places.
- `publication.yaml` is default-deny: category tier, no backtest results, weekly paper
  returns with a lag of at least 14 days, positions/fills/exposure withheld.
- Weekly (≈15 min, human): sync journals and proofs from the host; `sqy evidence export`;
  `tools/bundle_leak_scan.py`; verify with the showcase's verifier; open the showcase PR.

## Paper runners

- A dedicated small VM by default; one unix user and one Alpaca paper account per
  deployment; keys via systemd credentials; `isolation: process` for decide.
- `deploy/deploy.py` refuses during market hours ±30 minutes and while configured marker
  paths exist on the host (read-only checks); releases flip atomically.
- Never shares hosts, users, keys or paper accounts with other trading systems.

## Data room and sale

1. `tools/dataroom.py --family F --recipient R` — NDA bundle + recipient watermark;
   shared through a private repository per deal, read-only, with a revocation date.
2. Sale: `tools/extract_family.sh F DEST` (tree hash checked), verify ledgers and the
   project-index inclusion, transfer the repository, hand over journals and proofs
   (paper accounts are not transferable), mark the family `sold` in the showcase registry.
3. A transfer drill on the `drill` family every quarter.

## CI (no secrets)

Offline `sqy check`, tests, `sqy evidence verify` (and append-only against the base),
one family per commit, no cross-family imports, publication policies validate, and
no file mentions a name in `lab.forbidden_references` (for example the showcase repository).

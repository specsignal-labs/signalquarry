# 0005. Validation gates and trial ledger

Status: accepted (2026-09-25)

## Context

Agents can generate and test strategies faster than humans can audit them. Selection bias from many trials is the dominant failure mode.

## Decision

The engine appends every distinct run to a hash-chained trial ledger. A per-family holdout is sealed at `spec freeze` and opened once. Default gates G0–G5 unlock claim levels (`in_sample`, `walk_forward`, `holdout_passed`, `paper_forward`); DSR uses the project-wide trial count. Statistics (PSR, DSR, MinTRL, bootstrap, PBO) are implemented in-house with numpy and `statistics.NormalDist`.

## Consequences

Every CLI envelope reports the claim level and trial counts. Gate failure exits with status 2 so agents stop. Options are always graded low-evidence and have no holdout.

## Amendment (2026-09-26): per-family layout

`[evidence] layout = "per_family"` stores each family's ledgers under its own
directory (transferable on sale) and chains every family append into
`evidence/project_index.jsonl`. The DSR denominator stays project-wide; a family log
that disagrees with the index is treated as corrupt.

## Reporting a blocked evaluation

A blocked evaluation stops research, retries and holdout access. Rendering the
existing evidence with `sqy report` is allowed before stopping: the report retains
the returned grade, claim level and failure reasons. This reporting step does not
change gates, configuration, trial budgets or evidence records.

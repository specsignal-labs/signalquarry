# 0003. Look-ahead is prevented by structure

Status: accepted (2026-09-25)

## Context

Documentation-only rules against look-ahead fail with agents, which also carry memorized market history.

## Decision

The context builder exposes arrays truncated at the information cutoff; prices are point-in-time adjusted from raw bars plus corporate actions; holdout rows are clipped before the engine starts; `check` enforces an import policy and runs a mutation test that perturbs every post-cutoff bar.

## Consequences

A strategy cannot see the future through the framework. The mutation test is the regression guard.

## Amendment (2026-09-26)

The mutation check compares decisions **through the cutoff session inclusive**
(the decision for session `cut` may only use bars before `cut`), at three cutoffs
and with bars scaled both down and up. The earlier version compared only decisions
before the cutoff and so could not see a one-bar leak; a meta-test now injects
that off-by-one into the context builder and requires the check to fail.

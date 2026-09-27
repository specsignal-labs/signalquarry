# 0001. Layering and public surface

Status: accepted (2026-09-25)

## Context

Agents copy whatever they can import. Internals exposed as public API become frozen by accident, and pure code that can reach the network cannot be trusted to be deterministic.

## Decision

Public surface is `sdk`, `api`, `plugins` and `testing`; everything else lives in `_internal`. import-linter enforces the layer order cli → api → {paper | evidence | publish} → validation → engine → {data | project} → {sdk | plugins} → contracts → canonical. Pure layers may not import network, subprocess or paper modules. `canonical` and `evidence.verify` are stdlib-only.

## Consequences

Internals can be refactored freely. Agents see immediately when they reach into `_internal`. A griffe snapshot in CI catches unlabeled public-surface changes.

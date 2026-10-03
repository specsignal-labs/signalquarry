# 0007. Packaging and release

Status: accepted (2026-09-25)

## Context

A solo maintainer needs a small, verifiable release surface.

## Decision

One pure wheel `signalquarry` with CLI entry points `signalquarry` and `sqy`; Python 3.12–3.14; runtime dependencies pydantic, numpy, pyarrow, pyyaml only. Releases use signed tags, PyPI Trusted Publishing, PEP 740 attestations, build provenance and a CycloneDX SBOM. Apache-2.0 with NOTICE and TRADEMARKS; DCO, no CLA; no telemetry. `init` templates are CC0.

The optional `mcp` extra adds the official MCP Python SDK for a local stdio
adapter launched as `sqy-mcp`. The base wheel retains the four mandatory
runtime dependencies. The adapter calls `signalquarry.api` and returns the
same envelopes. It does not expose human-only paper arming or trial-budget
extension; paper submission still uses the kernel's existing gates.

## Consequences

New runtime dependencies require an ADR.

# 0007. Packaging and release

Status: accepted (2026-09-25)

## Context

A solo maintainer needs a small, verifiable release surface.

## Decision

One pure wheel `signalquarry` with CLI entry points `signalquarry` and `sqy`; Python 3.12–3.14; runtime dependencies pydantic, numpy, pyarrow, pyyaml only. Releases use signed tags, PyPI Trusted Publishing, PEP 740 attestations, build provenance and a CycloneDX SBOM. Apache-2.0 with NOTICE and TRADEMARKS; DCO, no CLA; no telemetry. `init` templates are CC0.

## Consequences

New runtime dependencies require an ADR.

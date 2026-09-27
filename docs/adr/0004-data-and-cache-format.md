# 0004. Data layer and cache format

Status: accepted (2026-09-25)

## Context

Results must be reproducible from recorded inputs, without redistributing provider data.

## Decision

Providers implement `MarketDataProvider`; registering third-party providers through entry points is deferred until the protocol settles (ADR 0009). Raw provider pages are stored gzip-compressed and content-addressed by SHA-256; normalized per-symbol Parquet is derived. The cache lives outside the project (`$SIGNALQUARRY_CACHE_DIR`). Committed `DatasetManifestV1` files contain hashes only and are marked `redistributable: false`. The Alpaca adapter defaults to the SIP feed with `adjustment=raw`.

## Consequences

pyarrow becomes a core dependency. Users bring their own data keys; no market data is ever committed or shared.

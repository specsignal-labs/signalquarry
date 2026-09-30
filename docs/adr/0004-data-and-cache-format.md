# 0004. Data layer and cache format

Status: accepted (2026-09-25)

## Context

Results must be reproducible from recorded inputs, without redistributing provider data.

## Decision

Providers implement `MarketDataProvider`; registering third-party providers through entry points is deferred until the protocol settles (ADR 0009). Raw provider pages are stored gzip-compressed and content-addressed by SHA-256. The cache lives outside the project (`$SIGNALQUARRY_CACHE_DIR`). Committed `DatasetManifestV1` files contain hashes only and are marked `redistributable: false`. The Alpaca adapter defaults to the SIP feed with `adjustment=raw`.

The initial per-symbol Parquet layout was not implemented. For cross-sectional research, derived Parquet is instead stored per field as session-by-symbol columns, keyed by the canonical dataset identity (ADR 0010). The raw pages and dataset manifest remain the source of truth; the panel can be deleted and rebuilt. Raw prices are stored in int64 micro-units with a separate presence mask. The panel does not itself establish a point-in-time security universe or corporate-action knowledge cutoff.

## Consequences

pyarrow becomes a core dependency. Users bring their own data keys; no market data is ever committed or shared.

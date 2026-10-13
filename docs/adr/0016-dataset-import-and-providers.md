# 0016. Dataset import and data providers

Status: accepted (2026-10-13) as amended under "Decisions". Implementation follows the rollout
given there; the acceptance tests below are its exit criteria.

## Context

The only way to get daily bars into a project is `sqy data fetch`, which talks to Alpaca
(ADR 0004). A developer who holds data from another vendor, or a file they are licensed to
use, cannot run a strategy on it. The plugin groups are `gates`, `report_sections` and
`brokers` (ADR 0009); there is none for data.

What must not be lost in opening this up:

- **Replay.** A manifest records the SHA-256 of every raw page, the pages are kept
  byte-for-byte in the local cache, and `sqy data verify` rebuilds the dataset from them. The
  dataset identity is derived from the rebuilt arrays.
- **Raw prices with dated actions.** Bars are requested unadjusted and splits and dividends
  are applied point-in-time by the engine, so no future corporate action leaks into a past
  price. A vendor's back-adjusted series does not have that property: every historical price
  in it depends on actions that happened later.
- **Nothing published.** Manifests hold hashes and coverage, never prices, and are marked
  non-redistributable.

## Proposed decision

### 1. One contract: raw pages in, a normalized dataset out

A provider is anything that can produce *raw pages* (bytes with an endpoint label and
parameters) for daily bars and for corporate actions, and parse its own pages into the
normalized rows the library already builds datasets from. Everything after that is the
existing path: content-addressed storage, `make_manifest`, `dataset_from_manifest`,
`data verify`. Alpaca becomes the first implementation of the contract rather than a special
case in `api/data.py`.

### 2. A plugin group `data_providers`

Entry point `signalquarry.data_providers`, with a protocol carrying `name`, `api_version`,
the feeds it serves, `daily_bars(symbols, start, end, feed)`,
`corporate_actions(symbols, start, end)`, the two parsers, and two declarations:
`adjustment` (`raw` or `adjusted`) and `redistributable` (default false). Discovery, version
checks and failure handling follow ADR 0009: a broken provider is an error at the command that
needs it, never a silent fallback to another source. Network access stays inside the provider;
`SIGNALQUARRY_OFFLINE=1` forbids it for every provider.

### 3. A built-in `local` provider and `sqy data import`

`sqy data import --path bars.{csv,parquet} [--actions actions.csv] --source TEXT
--adjustment raw|adjusted [--license TEXT]`. The input schema is fixed: `session`, `symbol`,
`open`, `high`, `low`, `close`, `volume`; actions as `symbol`, `type` (`split` or
`dividend`), `ex_date`, `pay_date`, `ratio` or `amount`. The command validates types,
duplicates and ordering, stores the *file bytes* as cached pages, and writes a manifest with
`provider: local`, the declared source, licence text, adjustment and import time. Replay then
works exactly as for fetched data. `strategy.yaml` may name `feed: local` and, optionally, pin
`data.dataset` to one dataset id (a hash-neutral field: absent means "the latest matching
manifest", as today).

### 4. What imported data may claim

The evidence grade says how far a result on that data can be taken:

| Data | Grade | Ceiling |
|---|---|---|
| fetched through a provider, raw with dated actions | `historical` | as today |
| imported file, declared `raw`, with an actions file | `historical`, with the statement "provenance supplied by the user" in every report | as today |
| imported file, declared `raw`, no actions file | `historical`, same statement, plus a warning that splits and dividends are not modelled | as today |
| declared `adjusted`, from any source | `historical_adjusted` | `in_sample` |

An adjusted series cannot support a walk-forward claim because the test folds would be read
through prices that already contain the future. This is the one place where the proposal adds
a grade.

### 5. Quality before use

`sqy data import` runs the data-quality assessment and prints its findings. Findings do not
block the import. A study may declare thresholds that do.

### 6. Dated membership

`sqy universe import --path membership.csv` (`symbol`, `start`, `end`, optional `asset_id` and
`observed_at`) records dated universe membership, including delisted names, for factor
evaluation. Without `observed_at` the membership is labelled `reconstructed`: it describes
what the source says today about the past, not what was knowable then, and factor results on
it stay `unverified` (ADR 0010, ADR 0013).

## Acceptance tests required before this ADR is accepted

1. The same input file gives the same dataset identity and the same backtest ledger on two
   machines; `sqy data verify` detects a changed cached page.
2. A strategy runs unchanged on fetched and on imported data for the same bars and produces
   the same ledger hash.
3. An `adjusted` dataset never yields a claim above `in_sample`, whatever the gates return.
4. With `SIGNALQUARRY_OFFLINE=1` no provider makes a network call.
5. A provider that fails to load, or declares another API version, stops the command with a
   coded error; no other provider is tried.
6. No manifest, report or exported bundle contains a price from imported data.
7. Existing manifests and existing configuration hashes are unchanged.

## Decisions

Settled on 2026-10-13. The owner delegated these decisions to the maintainer of this change; the reasoning is written down so that each one can be revisited.

1. **Two grades, decided by what the file can support, not by who supplied it.**

   | Data | Grade | Ceiling |
   |---|---|---|
   | fetched through a provider, raw with dated actions | `historical` | as today |
   | imported, declared `raw`, with an actions file | `historical`, and every report states "provenance supplied by the user" | as today, including `holdout_passed` |
   | imported, declared `adjusted`; or declared `raw` with no actions file | `historical_adjusted` | `in_sample` |

   The gates exist to protect the researcher's own decision, and a file they are entitled to
   use is as good for that as a vendor API. What an import cannot do is make a back-adjusted
   series point-in-time, and a raw series without its splits and dividends is wrong on every
   ex-date. Those two cases share one grade and one ceiling; section 4's separate "raw, no
   actions" row is withdrawn. An import with no actions must say so (`--no-actions`), so the
   lower ceiling is chosen rather than discovered.
2. **Source is required free text; licence is optional free text.** Both are recorded as
   given. The framework cannot verify either, and a short list of choices would only invite a
   wrong answer.
3. **Results on imported data are never published.** Export stays default-deny for them
   whatever the declared licence says, until 1.0 reconsiders publication as a whole.
4. **Rollout: the local import first, the plugin group later.** `sqy data import` and the
   `local` feed are built on the existing library, manifest and replay path. The
   `data_providers` plugin group of section 2 is deferred until a second network provider is
   actually wanted: an extension point with one implementation is a guess about the second.
   Section 1 still holds as the shape the import must fit (raw pages in, a normalized dataset
   out), so the later step is a refactor and not a redesign.
5. **`sqy universe import` follows the bar import** and keeps the labels of section 6.

## Consequences

Any daily data a user is entitled to use becomes usable, under the same replay and
point-in-time rules as fetched data, and the difference between a raw and an adjusted series is
visible in what a result is allowed to claim. The cost is a second code path for data and one
more grade to explain.

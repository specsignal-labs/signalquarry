# Market data

## Importing daily bars from a local CSV

Use a file you are entitled to use. Import runs offline, validates the entire input before
writing, and assesses data quality; quality findings are returned as warnings and do not
block a valid import.
The import envelope returns finding codes and counts. Use `sqy data quality --dataset-id ID`
for the full dates and symbol details. This keeps the import response bounded so a repeated
import needs no new output artifact, even when a sparse file has years of missing sessions.

```bash
sqy data import --bars bars.csv --actions actions.csv --adjustment raw \
  --source "Research vendor export" --license "Licence text as supplied" --project my-project
sqy data import --bars adjusted.csv --no-actions --adjustment adjusted \
  --source "Adjusted vendor export" --project my-project
sqy data import --bars raw.csv --no-actions --adjustment raw \
  --source "Export without actions" --project my-project
sqy data verify --project my-project
sqy data ls --project my-project
```

`--source` is required free text and `--license` is optional free text. Both are recorded
exactly as supplied; neither is verified. Choose exactly one of `--actions` and
`--no-actions`. Adjusted bars cannot be combined with an actions file.

The UTF-8 bars CSV has this exact header, in this order:

```csv
session,symbol,open,high,low,close,volume
2024-01-02,SPY,100.123456,101,99,100.5,1000000
```

Sessions use `YYYY-MM-DD` and must be NYSE trading sessions within the verified calendar
range, 2016–2035. Symbols use the strategy format: 1–10 uppercase letters, digits or dots,
starting with a letter. Prices are positive decimal text with at most six decimal places
and must fit integer micro-dollars. High and low enclose open and close. Volume is finite
and nonnegative, and can be fractional. Rows can be in any order; a repeated
`(session, symbol)` is refused with its line number. A missing symbol row stays an absent
bar, as with fetched data.

The actions CSV has this exact header:

```csv
symbol,type,ex_date,pay_date,ratio,amount
SPY,split,2024-02-01,,2,
SPY,dividend,2024-03-01,2024-03-15,,0.50
```

A split needs a positive ratio of new shares per old share, with empty `pay_date` and
`amount`. A dividend needs a nonnegative amount per share, a pay date on or after its ex-date,
and an empty `ratio`. Action dates use `YYYY-MM-DD`; action symbols must occur in the bars.
A repeated action type, symbol and ex-date is refused. An actions file containing only the
header explicitly declares that there are no events; a completely empty file is refused.

| Declaration | Evidence grade | Claim ceiling |
|---|---|---|
| `raw` with an actions file | `historical` | Existing gates, including the sealed holdout |
| `adjusted` with `--no-actions` | `historical_adjusted` | `in_sample` |
| `raw` with `--no-actions` | `historical_adjusted` | `in_sample` |

Adjusted prices can incorporate future actions; raw bars without actions omit split and
dividend economics. These cases cannot open a holdout or pass `paper preflight`, even when
the evaluation gates pass. Every imported-data evidence block and report states
“provenance supplied by the user”. Backtests, sweeps and studies record real-data trials.
Results produced on imported data cannot be exported under any licence or publication tier.

Set the strategy's `data.feed` to `local`. Resolution picks the most recently imported
manifest covering all its symbols. Imports store the original file bytes as content-addressed
pages in the existing private cache; committed manifests contain hashes, declarations,
symbols and coverage, never prices. Verification rebuilds from those bytes. Re-importing
identical bytes with the same source, licence and declarations returns the original manifest
and leaves its timestamp unchanged. Different declarations get separate manifests while
the normalized dataset identity remains determined by the rows, actions and `local:local`
source. No provider plugin, universe import or dataset pin is required.

The equivalent Python entry point is `signalquarry.api.data_import(bars=Path(...),
source="...", adjustment="raw", actions=Path(...), project=Path(...))`; use
`no_actions=True` when omitting the actions file.

## Credentials

Set `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`, or create
`~/.config/signalquarry/credentials.toml` (permissions 0600):

```toml
[data]
key_id = "..."
secret_key = "..."

[paper.demo]          # one table per paper deployment (see Paper trading)
key_id = "..."
secret_key = "..."
```

Values are never printed. `sqy doctor` reports which source was found (environment or file).
As a last line of defense, every command scrubs the credential values it has seen
(environment or file) from its output and error traces, replacing them with `[REDACTED]`.

## Fetching

```bash
sqy data fetch --strategy sma-trend          # symbols and benchmark of the strategy
sqy data fetch --symbols SPY,QQQ --start 2016-01-01 --feed sip
sqy data verify                              # re-hash cached pages against every manifest
sqy data ls
```

Bars are requested **raw** (`adjustment=raw`); splits and dividends come from
the corporate-actions endpoint and are applied point-in-time by the engine, so
no future dividend leaks into past prices. Every response page is stored
byte-for-byte in `$SIGNALQUARRY_CACHE_DIR` (default `~/.cache/signalquarry`),
content-addressed by SHA-256. The project commits only
`data/manifests/*.json`: page hashes, coverage and the dataset identity — no
prices (`redistributable: false`).

The default feed is SIP. Decisions use bars through the previous session, which
are older than 15 minutes when you decide, so free plans can use SIP too.

Set `SIGNALQUARRY_OFFLINE=1` to forbid every network call (the project CI
template does this).

## Data quality

```bash
sqy data quality --strategy sma-trend
sqy data quality --dataset-id alpaca-sip-1day-example
```

The command checks session calendars, missing bars, OHLC consistency, nonpositive
prices, zero volume, stale closes and large moves unexplained by recorded splits.
Synthetic datasets skip the holiday and missing-trading-day checks. Findings are
information and do not block a run; the command succeeds even when it finds issues.

The full report is written to `data/quality/DATASET_ID.json` in the project, or
`data/quality/synthetic.json` for synthetic data. A later assessment of the same
dataset overwrites that report. It holds dates and counts only, with no price,
volume or return values. The command envelope contains a compact summary and
the report's SHA-256.

With `--strategy`, the sufficiency block describes the common history of the
strategy's symbols. It reports years before the family's sealed holdout, or the
configured holdout when unsealed, and whole walk-forward test folds after the
training burn-in. The flags compare that history with G1's five-year requirement
and G2's six-fold requirement. They do not establish that either gate passes:
evaluation also checks trading activity and performance. Warm-up sessions are
the strategy's declared lookback. Options strategies have no holdout, so none is
subtracted.

## Prospective asset snapshots

```bash
sqy universe snapshot
sqy universe verify
sqy universe as-of --known-at 2026-09-28T16:00:00+00:00
```

`snapshot` makes one read-only request to Alpaca's **paper** assets endpoint
using the data credentials. It omits the status filter to capture active and
inactive US-equity assets together. The raw response stays in the local cache;
`data/universe/snapshots/` holds a hash-only manifest with the UTC capture time,
counts, and no asset names or symbols. `verify` re-hashes and re-parses it.
`as-of` finds the latest verified capture at or before the given cutoff, only
if it is no more than 31 days old. Earlier dates are unavailable. Alpaca's
current asset list is not a historical membership archive, and `us_equity`
does not identify common stocks alone. These snapshots therefore do not yet
make a factor backtest point-in-time.

## Building a dated common-stock universe

`sqy universe build` constructs membership for one decision session. It needs
an asset snapshot, an Alpaca dataset with 20 prior sessions, and a complete
classification snapshot that was observed by the cutoff. The classification
file uses this shape:

```json
{
  "schema": "signalquarry.instrument-classification-snapshot/v1",
  "source": "your dated security-master source",
  "observed_at": "2026-09-28T19:00:00Z",
  "records": [
    {"symbol": "AAA", "security_type": "common_stock", "listed_at": "2020-01-02"},
    {"symbol": "ETF", "security_type": "other", "listed_at": null}
  ]
}
```

Include exactly one record for every asset in the captured asset snapshot,
including inactive assets and non-common instruments. Records may use stable
`asset_id` UUIDs instead of symbols. Symbol keys require unique symbols across
the full asset snapshot. Common stocks need a listing date. The caller supplies
the classification source; SignalQuarry hashes and keeps the snapshot in its
private local cache so `universe verify` can replay the build.

Choose explicit thresholds for each build. This example uses illustrative
values; it does not prescribe a trading universe:

```bash
sqy universe build \
  --session 2026-09-29 \
  --known-at 2026-09-28T21:00:00+00:00 \
  --dataset-id alpaca-sip-1day-example \
  --classification-file ./security-master.json \
  --minimum-price 5 \
  --minimum-listing-age-days 90 \
  --minimum-dollar-volume-percentile 20
sqy universe verify
```

The cutoff must precede the decision session. Asset and classification
snapshots must be no more than 31 days old, and the dataset must have been
fetched by the cutoff and end no later than its UTC date. The build uses the
last 20 complete sessions before the decision session. If fetched on the date
of its latest session, the dataset must be fetched at least 16:15
America/New_York so the regular session's daily bar has finished. It filters on the latest
prior raw close, listing age, then the cross-sectional percentile rank of each
eligible stock's 20-session median close-times-volume. A missing bar in the
lookback excludes that name from the ranking. No provider metadata or bars are
downloaded by `universe build`.

The manifest records the filter policy, source hashes, counts, and member
symbols. Command output exposes only the summary and hashes. Verification
replays membership from the cached classification snapshot, asset pages, and
dataset pages. Earlier periods require dated snapshots that actually existed by
each cutoff; a current source file cannot reconstruct them. Raw close and
volume filters do not establish corporate-action-adjusted factor returns.
The command reports classification provenance as unverified because it cannot
independently establish when or how the caller's classification was assembled.

## Corporate-action audit captures

```bash
sqy data capture-actions --symbols SPY,QQQ --start 2026-09-01 --end 2026-09-30
sqy data verify
sqy data ls
```

The capture requests every corporate-action category and keeps the raw pages
and an immutable metadata-and-hashes observation record in the private cache under
`corporate-actions/`. The record contains provider IDs, category, dates, row
hashes, and page hashes, but no merger consideration, CUSIP, or other raw terms.
It is marked nonredistributable and is not written into the public project.
`sqy data verify` replays the cached pages and checks every observation; `data ls`
lists capture hashes without asserting that they verified. Capture
does not make an unsupported action executable or establish when a provider
first knew it; the current backtest and paper guards remain in force.

## Options coverage

Alpaca's options history starts in February 2024. Check whether expired contracts and
their daily bars are available for a month before relying on them:

```
sqy data probe options-coverage --underlying QQQ --month 2024-03
```

If no bars come back the envelope warns `OPTIONS_HISTORY_UNAVAILABLE`; options backtests
keep using modelled prices, and a paper forward test is the primary evidence.

To build real quote history from now on, record the chain each day (for example from
the paper host's post-close timer or by hand):

```
sqy data record options --underlying QQQ --max-dte 60 --width 0.2
```

It stores the provider's raw snapshot pages (puts and calls within the strike window and
expiration range) in the cache and writes a hash-only record under
`data/options/QQQ/` to commit; `sqy data verify` re-hashes recorded chains and
`sqy data ls` lists them. Backtests, evaluations and sweeps of options strategies on real
data use a recorded session's bid and ask instead of modelled prices (the snapshot stands
in for every checkpoint that session; contracts it lacks stay modelled) and warn
`OPTIONS_RECORDED_CHAINS_USED:<sessions>`. Synthetic data never uses recordings. Results
stay graded `low_evidence_options`.

## Synthetic data

Demo projects use deterministic synthetic symbols (`SYNA`, `SYNB`, `SYNC`,
`SYNX`) with splits and dividends. Their grade is `synthetic` and the claim level
stays `none`.

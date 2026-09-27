# Market data

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

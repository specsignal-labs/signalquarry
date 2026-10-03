# Contributing

Thank you for helping. SignalQuarry is maintained by one person, so small,
focused changes with tests are the easiest to review.

## Before you start

- Open an issue first for anything larger than a bug fix. Decisions live in
  `docs/adr/`; changing one means updating its ADR in the same pull request.
- The authoring API (`signalquarry.sdk`), `strategy.yaml` v1, CLI names and flags,
  the envelope, exit codes, reason codes and the ledger, seal, manifest and bundle
  formats change additively only in 0.x. `tools/public_api.txt` records the public
  Python surface; a pull request that changes it needs the `api-change` label.
- New runtime dependencies need an ADR. There is no live-trading path and there
  will not be one.

## Development

```bash
uv sync --locked            # Python 3.12+; on Apple Silicon use a native arm64 Python
uv run ruff check src tests && uv run ruff format --check src tests
uv run lint-imports         # layer contracts
uv run pyright              # standard everywhere, strict on sdk and api
uv run python -m pytest
uv run python tools/gen_docs.py --check
uv run python tools/check_headers.py
uvx --from 'reuse[charset-normalizer]' reuse lint
gitleaks git --no-banner --redact .   # secrets across history (go install …/gitleaks/v8@v8.28.0)
uv run python tools/api_snapshot.py --check   # public API unchanged (or rerun without --check)
```

Every source file starts with an SPDX license header. Engine changes must keep
the golden tests and the backtest↔paper parity test green; benchmark with
`uv run python tools/bench.py`.

## Private Alpaca response cassettes

`tools/alpaca_cassette.py` can capture an explicit, ordered session of
allowlisted Alpaca `GET` responses and replay it through the existing data and
paper broker transports. The recorder excludes credentials and request headers,
redacts known credential values, refuses request URLs containing credentials,
rejects account/order endpoints and every mutating request, and refuses cassette
destinations inside any Git worktree. A cassette checksum covers request and
response metadata; each response body has a separately verified SHA-256 hash.
Generated `private-provider` cassette files can contain market data; keep them
outside the repository and never attach them to a pull request. Test examples
must use `recording_kind="synthetic"` and are not real provider evidence.

The allowed paper-origin reads are the clock, calendar, asset list and option
contract catalog. Account, position, order and activity responses are not
recordable. Replay is strict and offline: it fails on a request mismatch or an
unconsumed entry, and has no network fallback. This is transport plumbing only;
the A2 real-provider and paper-lifecycle verification gates still require
separate owner-approved evidence.

For a deliberate local data capture, load credentials from the configured
credential store and choose a destination under a private cache directory:

```python
from datetime import date
from pathlib import Path

from signalquarry._internal.data.alpaca import AlpacaDataClient, UrllibTransport
from signalquarry._internal.data.credentials import load_data_credentials
from tools.alpaca_cassette import RecordingAlpacaTransport

credentials = load_data_credentials()
if credentials is None:
    raise RuntimeError("configure data credentials first")
cassette_path = Path.home() / ".cache/signalquarry/cassettes/spy-bars.json"
with RecordingAlpacaTransport(
    UrllibTransport(),
    cassette_path,
    recording_kind="private-provider",
    credentials=(credentials.key_id, credentials.secret_key),
) as recording:
    client = AlpacaDataClient(credentials.key_id, credentials.secret_key, transport=recording)
    client.daily_bars(("SPY",), date(2025, 1, 2), date(2025, 1, 3), "iex")
```

Replay the same request with `ReplayTransport.read(cassette_path)` as the
client's `transport`, then call `assert_complete()` to ensure the request
sequence matched the cassette exactly.

## Design docs

`docs/design/ARCHITECTURE.md` must match the code. Its "Generated from the code" section
(layers, direct imports, component inventory, command tree, plugin groups) is rebuilt by
`uv run python tools/gen_docs.py`. Adding, removing or renaming a framework module, or
changing the import contracts, must come with an edit to that page or an ADR in
`docs/adr/`. Enable the hook that checks both before each commit:

```bash
git config core.hooksPath .githooks
```

CI runs the same check over every pull request. When a change needs no design update, add
the trailer `Architecture: unchanged` to a commit message (or `SIGNALQUARRY_ARCH_OK=1` for
the local hook).

## Live-data smoke (maintainers)

The weekly `live-data-smoke` workflow runs a read-only check against Alpaca after it is
enabled on the default branch. Configure repository secrets
`SQY_ALPACA_DATA_KEY_ID` and `SQY_ALPACA_DATA_SECRET_KEY` with a dedicated data-only key.
Without both secrets, scheduled or manually dispatched runs report a skip. The workflow
fetches about 400 calendar days of SPY/QQQ bars, verifies the temporary manifest, runs the
sample SMA backtest, and records whether sampled QQQ option contracts have historical bars.
Project files and cached provider pages stay under the ephemeral runner temp directory and
are not uploaded or committed. The workflow has no order-submission step.

## Sign-off (DCO)

Contributions are accepted under the [Developer Certificate of Origin](https://developercertificate.org/).
Sign every commit: `git commit -s`. There is no CLA.

## AI assistance

AI-assisted contributions are welcome. Say so in the pull request and review
the result as if you wrote it: you are responsible for every line.

The smoke sets the disposable starter strategy to the same IEX feed it fetches;
the default SIP configuration of ordinary projects is unchanged.

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

## Sign-off (DCO)

Contributions are accepted under the [Developer Certificate of Origin](https://developercertificate.org/).
Sign every commit: `git commit -s`. There is no CLA.

## AI assistance

AI-assisted contributions are welcome. Say so in the pull request and review
the result as if you wrote it: you are responsible for every line.

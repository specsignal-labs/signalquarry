# Agent instructions

- This repository is public. Never add secrets, account identifiers,
  market data files, or strategy logic other than the documented examples.
- Follow the ADRs in `docs/adr/`. Changing a decision means updating its ADR.
- Layers are enforced by import-linter (`uv run lint-imports`): cli → api →
  paper/evidence → validation → engine → data/project → sdk → contracts → canonical.
  `canonical` and `evidence.verify` stay stdlib-only.
- The backtest and the paper runner share `engine.plan_orders`/`plan_pre_open`;
  change them only with the parity test (`tests/paper/test_parity.py`) green.
- There is no live-trading path. Only the Alpaca paper origin may appear in
  broker code; brokers declare `paper_only = True`. Never weaken the kernel's
  lease, journal, arm-token or reconcile checks to make a test pass.
- Reason codes are append-only; register every emitted code in
  `_internal/contracts/reason_codes.py`.
- Before committing: `uv run ruff check src tests`, `uv run ruff format --check src tests`,
  `uv run lint-imports`, `uv run python -m pytest`, and `python tools/leak_scan.py .`
  (CI runs it with the private rules and blocks on any finding).
- Use Python 3.12+. On Apple Silicon, verify `platform.machine() == "arm64"`.

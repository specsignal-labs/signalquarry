# {{project}}

A strategy lab built on SignalQuarry: independently transferable strategy families,
per-family evidence ledgers chained into one project index, commit-reveal commitments,
default-deny publication and isolated paper deployments. Design: `docs/design/LAB.md`.

```bash
uv sync
uv run sqy check                 # offline conformance for every family
uv run python tools/new_family.py momentum
uv run sqy evidence verify
```

The `drill` family exists only for the quarterly transfer drill
(`tools/extract_family.sh drill /tmp/drill-transfer`). It never trades.

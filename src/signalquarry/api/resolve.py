# SPDX-License-Identifier: Apache-2.0
"""Shared loading for commands: project strategies and the dataset a strategy runs on."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from signalquarry._internal.data.dataset import Dataset
from signalquarry._internal.data.library import Library, LibraryError, dataset_from_manifest, select_manifest
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.project.project import (
    LoadedStrategy,
    ProjectError,
    find_root,
    load_config,
    load_strategies,
)
from signalquarry.api.envelope import Envelope

SYNTHETIC_START, SYNTHETIC_END = date(2014, 1, 2), date(2025, 12, 31)


def library_for(root: Path) -> Library:
    cache = Path(
        os.environ.get("SIGNALQUARRY_CACHE_DIR") or Path.home() / ".cache" / "signalquarry"
    ).expanduser()
    return Library(cache_dir=cache, manifest_dir=root / "data" / "manifests")


@dataclass(frozen=True)
class Resolved:
    root: Path
    strategy: LoadedStrategy
    dataset: Dataset
    grade: str
    dataset_id: str

    def recorded_chains(self) -> dict[tuple[str, date], dict[str, tuple[Decimal, Decimal]]] | None:
        """Real recorded option quotes for options strategies on real data (never with synthetic bars)."""
        if self.grade == "synthetic" or self.strategy.spec.kind != "options_single_leg":
            return None
        from signalquarry.api.data import load_recorded_chains

        return load_recorded_chains(self.root) or None

    @property
    def evidence_grade(self) -> str:
        """What results on this data may be called: options prices are modelled, so never better
        than ``low_evidence_options`` (``synthetic`` stays ``synthetic``)."""
        if self.grade != "synthetic" and self.strategy.spec.kind == "options_single_leg":
            return "low_evidence_options"
        return self.grade


def resolve(command: str, strategy_id: str, project: Path | None) -> Resolved | Envelope:
    try:
        root = find_root(project)
        config = load_config(root)
        strategies = load_strategies(config)
    except ProjectError as exc:
        return Envelope(
            command=command, status="invalid", reason_codes=[exc.code], summary=exc.detail or exc.code
        )
    strategy = strategies.get(strategy_id)
    if strategy is None:
        return Envelope(
            command=command,
            status="invalid",
            reason_codes=["STRATEGY_NOT_FOUND"],
            summary=strategy_id,
            data={"strategies": sorted(strategies)},
        )
    if config.provider == "synthetic":
        return Resolved(
            root,
            strategy,
            synthetic_dataset(SYNTHETIC_START, SYNTHETIC_END, symbols=strategy.spec.data.symbols),
            "synthetic",
            "synthetic",
        )
    library = library_for(root)
    manifest = select_manifest(library, strategy.spec.data.symbols, strategy.spec.data.feed)
    if manifest is None:
        return Envelope(
            command=command,
            status="unavailable",
            reason_codes=["PROVIDER_UNAVAILABLE"],
            summary=f"no fetched {strategy.spec.data.feed} dataset covers {', '.join(strategy.spec.data.symbols)}",
            next_actions=[
                {
                    "command": f"sqy data fetch --strategy {strategy_id}",
                    "why": "Fetch and record the data first.",
                }
            ],
        )
    try:
        dataset = dataset_from_manifest(library, manifest)
    except LibraryError as exc:
        return Envelope(
            command=command,
            status="blocked",
            reason_codes=[exc.code],
            summary=str(exc),
            next_actions=[
                {"command": "sqy data verify", "why": "The cached data no longer matches its manifest."}
            ],
        )
    return Resolved(root, strategy, dataset, "historical", manifest["dataset_id"])

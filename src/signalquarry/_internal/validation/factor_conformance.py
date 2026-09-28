# SPDX-License-Identifier: Apache-2.0
"""Synthetic contract, determinism and future-bar checks for declared factors."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from types import MappingProxyType

import numpy as np

from signalquarry._internal.data.dataset import FIELDS, Dataset
from signalquarry._internal.data.panel import LoadedPanel
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.factors import run_factor
from signalquarry._internal.validation.conformance import CheckResult, perturb_after
from signalquarry.sdk.factors import FactorDef
from signalquarry.sdk.strategy import Params


def _panel(dataset: Dataset) -> LoadedPanel:
    symbols = dataset.symbols
    micro = MappingProxyType(
        {
            field: np.column_stack([dataset.series[symbol].micro[field] for symbol in symbols])
            for field in FIELDS
        }
    )
    return LoadedPanel(
        dataset_identity=dataset.identity(),
        sessions=dataset.sessions,
        symbols=symbols,
        micro=micro,
        volume=np.column_stack([dataset.series[symbol].volume for symbol in symbols]),
        present=np.column_stack([dataset.series[symbol].present for symbol in symbols]),
    )


def _score(
    definition: FactorDef, params: Params, panel: LoadedPanel, universe: tuple[str, ...], session: date
) -> Mapping[str, float]:
    return run_factor(definition, params, panel.window(decision_session=session), universe)


def run_factor_checks(
    definition: FactorDef, params: Params, *, universe: tuple[str, ...] = ("SYNA", "SYNB", "SYNC", "SYNX")
) -> list[CheckResult]:
    """Check a factor on synthetic data only; passing does not attest real-data provenance."""
    if not universe or len(set(universe)) != len(universe):
        return [CheckResult("contract", False, "FACTOR_UNIVERSE_INVALID")]
    data = synthetic_dataset(date(2018, 1, 2), date(2023, 12, 29), symbols=universe)
    panel = _panel(data)
    cuts = tuple(int(len(data.sessions) * fraction) for fraction in (0.45, 0.6, 0.75))
    baseline: dict[int, Mapping[str, float]] = {}
    for cut in cuts:
        try:
            baseline[cut] = _score(definition, params, panel, universe, data.sessions[cut])
        except Exception as exc:
            return [CheckResult("contract", False, f"{type(exc).__name__}: {exc}")]
    results = [CheckResult("contract", True, "finite declared-symbol scores on synthetic panels")]
    try:
        repeated = {cut: _score(definition, params, panel, universe, data.sessions[cut]) for cut in cuts}
    except Exception as exc:
        results.append(CheckResult("determinism", False, f"repeat failed: {type(exc).__name__}: {exc}"))
        return results
    results.append(CheckResult("determinism", baseline == repeated, "3 repeated decision sessions"))
    for cut in cuts:
        for low, high in ((0.5, 0.8), (2.0, 1.3)):
            try:
                changed = _panel(perturb_after(data, cut, low, high))
                observed = _score(definition, params, changed, universe, data.sessions[cut])
            except Exception as exc:
                results.append(CheckResult("lookahead", False, f"mutated run failed: {exc}"))
                return results
            if observed != baseline[cut]:
                results.append(
                    CheckResult("lookahead", False, f"score for {data.sessions[cut]} changed with later bars")
                )
                return results
    results.append(CheckResult("lookahead", True, "3 cuts, 2 perturbations per cut"))
    return results

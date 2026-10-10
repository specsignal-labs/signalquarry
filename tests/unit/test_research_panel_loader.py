# SPDX-License-Identifier: Apache-2.0
"""``signalquarry.research.load_panel``: bars for exploration that stop before a sealed holdout."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from signalquarry import research
from signalquarry._internal.data.alpaca import AlpacaDataClient, RateLimiter
from signalquarry._internal.data.dataset import MICRO
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry.api import data_fetch, holdout_seal, init, spec_freeze
from tests.fakes import FakeAlpaca
from tests.helpers import Split, dataset, weekdays

END = date(2024, 12, 31)
GAP = 400  # a session index inside the fetched history (fetching starts in 2016)


def _demo(tmp_path: Path, name: str, symbol: str = "SYNA") -> Path:
    project = tmp_path / name
    package = name.replace("-", "_")
    assert init(project, demo=True, package=package).status == "ok"
    if symbol != "SYNA":
        spec = project / f"src/{package}/sma_trend/strategy.yaml"
        spec.write_text(spec.read_text().replace("SYNA", symbol))
    return project


def _files(project: Path) -> set[str]:
    return {str(path.relative_to(project)) for path in project.rglob("*") if path.is_file()}


def test_synthetic_panel_matches_the_dataset_and_is_read_only(tmp_path: Path) -> None:
    project = _demo(tmp_path, "panel-demo")
    before = _files(project)
    panel = research.load_panel("sma-trend", project=project)
    source = synthetic_dataset(symbols=("SYNA",))
    assert (panel.strategy_id, panel.family, panel.dataset_id, panel.grade) == (
        "sma-trend",
        "sma-trend",
        "synthetic",
        "synthetic",
    )
    assert panel.dataset_identity == source.identity()
    assert panel.sealed_from is None and panel.split_adjusted is True
    assert panel.sessions == source.sessions and panel.symbols == ("SYNA",)
    assert panel.fields == research.PANEL_FIELDS == ("open", "high", "low", "close", "volume")
    np.testing.assert_allclose(panel.field("close")[:, 0], source.series["SYNA"].micro["close"] / MICRO)
    np.testing.assert_allclose(panel.field("volume")[:, 0], source.series["SYNA"].volume)
    assert panel.present.all() and panel.present.shape == (len(source.sessions), 1)
    for values in (*panel.values.values(), panel.present):
        assert values.flags.writeable is False
    with pytest.raises(ValueError):
        panel.field("close")[0, 0] = 1.0
    with pytest.raises(TypeError):
        panel.values["close"] = panel.field("open")  # type: ignore[index]
    assert _files(project) == before  # nothing recorded, nothing written

    table = panel.to_arrow()
    assert table.column_names == ["session", "symbol", "open", "high", "low", "close", "volume", "present"]
    assert table.num_rows == len(source.sessions)
    assert table.column("session")[0].as_py() == source.sessions[0]
    assert table.column("close")[-1].as_py() == pytest.approx(float(panel.field("close")[-1, 0]))

    only = research.load_panel("sma-trend", fields=("close",), project=project)
    assert only.fields == ("close",)
    with pytest.raises(research.ResearchError) as missing:
        only.field("open")
    assert missing.value.code == "PANEL_FIELD_UNKNOWN"


def test_a_sealed_family_clips_the_panel_before_the_holdout(tmp_path: Path) -> None:
    project = _demo(tmp_path, "panel-seal")
    full = research.load_panel("sma-trend", project=project)
    frozen = spec_freeze("sma-trend", project=project)
    seal = date.fromisoformat(frozen.data["holdout_start"])
    panel = research.load_panel("sma-trend", project=project)
    assert panel.sealed_from == seal
    assert panel.sessions[-1] < seal <= full.sessions[len(panel.sessions)]
    assert len(panel.sessions) == sum(session < seal for session in full.sessions)
    np.testing.assert_array_equal(panel.field("close"), full.field("close")[: len(panel.sessions)])
    assert panel.dataset_identity == full.dataset_identity  # the identity is the dataset's, not the clip's


def test_prices_are_on_the_last_session_share_basis_using_only_earlier_splits(tmp_path: Path) -> None:
    # SYNB has a 2-for-1 split half-way through the synthetic sample.
    project = _demo(tmp_path, "panel-split", symbol="SYNB")
    source = synthetic_dataset(symbols=("SYNB",))
    (split,) = source.splits
    at = source.sessions.index(split.ex_date)
    raw = research.load_panel("sma-trend", split_adjusted=False, project=project)
    adjusted = research.load_panel("sma-trend", project=project)
    assert raw.split_adjusted is False
    # Raw prices halve across the ex-date; adjusted ones are continuous and equal raw afterwards.
    assert raw.field("close")[at, 0] / raw.field("close")[at - 1, 0] == pytest.approx(0.5, abs=0.1)
    assert adjusted.field("close")[at, 0] / adjusted.field("close")[at - 1, 0] == pytest.approx(1.0, abs=0.1)
    np.testing.assert_allclose(adjusted.field("close")[:at, 0], raw.field("close")[:at, 0] / 2)
    np.testing.assert_allclose(adjusted.field("close")[at:, 0], raw.field("close")[at:, 0])
    np.testing.assert_allclose(adjusted.field("volume")[:at, 0], raw.field("volume")[:at, 0] * 2)
    np.testing.assert_allclose(adjusted.field("volume")[at:, 0], raw.field("volume")[at:, 0])

    # A panel that ends before the split knows nothing about it: adjusted equals raw.
    spec = next(project.glob("src/*/sma_trend/strategy.yaml"))
    spec.write_text(
        spec.read_text().replace(
            "holdout: {months: 12}", "holdout: {months: 12, training_cutoff: 2018-12-31}"
        )
    )
    sys.modules.pop("panel_split.sma_trend.strategy", None)
    assert holdout_seal("sma-trend", project=project).data["holdout_start"] == "2019-01-01"
    early_raw = research.load_panel("sma-trend", split_adjusted=False, project=project)
    early = research.load_panel("sma-trend", project=project)
    assert early.sessions[-1] < split.ex_date
    np.testing.assert_array_equal(early.field("close"), early_raw.field("close"))
    np.testing.assert_array_equal(early.field("volume"), early_raw.field("volume"))


def _market() -> object:
    sessions = weekdays(date(2015, 1, 5), 2600)
    rng = np.random.Generator(np.random.PCG64(3))
    prices = {}
    for symbol, drift in (("SPY", 0.0012), ("QQQ", 0.0006)):
        closes = 100 * np.cumprod(1 + drift + rng.normal(0, 0.004, len(sessions)))
        opens = np.concatenate(([100.0], closes[:-1]))
        present = [True] * len(sessions)
        if symbol == "QQQ":
            present[GAP] = False
        prices[symbol] = {
            "open": list(np.round(opens, 2)),
            "close": list(np.round(closes, 2)),
            "present": present,
        }
    return dataset(sessions, prices, splits=[Split("QQQ", sessions[1300], 2)])


def test_recorded_data_needs_a_seal_and_then_stops_before_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    project = tmp_path / "panel-real"
    assert init(project, package="panel_real").status == "ok"
    with pytest.raises(research.ResearchError) as unfetched:
        research.load_panel("sma-trend", project=project)
    assert unfetched.value.code == "PROVIDER_UNAVAILABLE"

    spec = project / "src/panel_real/sma_trend/strategy.yaml"
    spec.write_text(spec.read_text().replace("benchmark: SPY", "benchmark: QQQ"))
    client = AlpacaDataClient(
        "k", "s", transport=FakeAlpaca(_market(), page_size=5000), limiter=RateLimiter(sleep=lambda s: None)
    )
    fetched = data_fetch(strategy_id="sma-trend", project=project, client=client, end=END)
    assert fetched.status == "ok"

    with pytest.raises(research.ResearchError) as unsealed:
        research.load_panel("sma-trend", project=project)
    assert unsealed.value.code == "PANEL_HOLDOUT_UNSEALED"
    assert "sqy holdout seal --strategy sma-trend" in str(unsealed.value)

    sealed = holdout_seal("sma-trend", project=project)
    seal = date.fromisoformat(sealed.data["holdout_start"])
    before = _files(project)
    panel = research.load_panel("sma-trend", project=project)
    assert _files(project) == before
    assert (panel.grade, panel.dataset_id) == ("historical", fetched.data["dataset_id"])
    assert panel.dataset_identity == fetched.data["dataset_identity"]
    assert panel.symbols == ("SPY", "QQQ")  # the strategy's symbols, then its benchmark
    assert panel.sealed_from == seal and panel.sessions[-1] < seal
    # An absent bar is NaN in every field and false in `present`.
    row = panel.sessions.index(weekdays(date(2015, 1, 5), 2600)[GAP])
    assert np.argwhere(~panel.present).tolist() == [[row, 1]]
    assert all(np.isnan(panel.field(name)[row, 1]) for name in panel.fields)
    assert not np.isnan(panel.field("close")[:, 0]).any()
    # The benchmark's split is inside the panel: its earlier prices are halved.
    raw = research.load_panel("sma-trend", split_adjusted=False, project=project)
    np.testing.assert_allclose(panel.field("close")[:10, 1], raw.field("close")[:10, 1] / 2)
    np.testing.assert_array_equal(panel.field("close")[:, 0], raw.field("close")[:, 0])

    (project / "data" / "manifests").joinpath(
        next(iter((project / "data" / "manifests").glob("*.json"))).name
    ).write_text("{}")
    with pytest.raises(research.ResearchError) as broken:
        research.load_panel("sma-trend", project=project)
    assert broken.value.code in ("PROVIDER_UNAVAILABLE", "DATA_MANIFEST_INVALID")


def test_refusals_and_the_optional_pandas_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _demo(tmp_path, "panel-errors")
    for fields in ((), ("close", "close"), ("vwap",)):
        with pytest.raises(research.ResearchError) as bad:
            research.load_panel("sma-trend", fields=fields, project=project)
        assert bad.value.code == "PANEL_FIELD_UNKNOWN"
    with pytest.raises(research.ResearchError) as unknown:
        research.load_panel("nope", project=project)
    assert unknown.value.code == "STRATEGY_NOT_FOUND"
    with pytest.raises(research.ResearchError) as outside:
        research.load_panel("sma-trend", project=tmp_path)
    assert outside.value.code == "PROJECT_NOT_FOUND"

    panel = research.load_panel("sma-trend", fields=("close",), project=project)
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(research.ResearchError) as no_pandas:
        panel.to_pandas()
    assert no_pandas.value.code == "PANDAS_NOT_INSTALLED"

    class _Frame:
        def __init__(self, data: dict) -> None:
            self.data, self.index = data, None

        def set_index(self, keys: list[str]) -> _Frame:
            self.index = keys
            return self

    fake = type(sys)("pandas")
    fake.DataFrame = _Frame  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pandas", fake)
    frame = panel.to_pandas()
    assert frame.index == ["session", "symbol"] and set(frame.data) == {
        "session",
        "symbol",
        "close",
        "present",
    }

    # A seal that leaves no session before it is reported, not returned as an empty panel.
    spec = next(project.glob("src/*/sma_trend/strategy.yaml"))
    spec.write_text(
        spec.read_text().replace(
            "holdout: {months: 12}", "holdout: {months: 12, training_cutoff: 2000-01-01}"
        )
    )
    sys.modules.pop("panel_errors.sma_trend.strategy", None)
    assert holdout_seal("sma-trend", project=project).status == "ok"
    with pytest.raises(research.ResearchError) as empty:
        research.load_panel("sma-trend", project=project)
    assert empty.value.code == "PANEL_EMPTY"

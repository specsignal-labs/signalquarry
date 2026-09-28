# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest

from signalquarry._internal.data.alpaca import AlpacaDataClient, RateLimiter
from signalquarry._internal.data.library import Library, LibraryError, build_dataset, make_manifest
from signalquarry._internal.data.panel import PanelStore
from signalquarry._internal.data.synthetic import synthetic_dataset
from tests.fakes import FakeAlpaca


def test_panel_window_has_strict_cutoff_mask_and_immutable_values(tmp_path: Path) -> None:
    dataset = synthetic_dataset(date(2024, 1, 2), date(2024, 1, 8), symbols=("SYNA", "SYNB"))
    dataset.series["SYNA"].present[1] = False
    store = PanelStore(tmp_path / "cache")
    path = store.build(dataset)
    assert store.build(dataset) == path

    window = store.window(
        dataset.identity(),
        decision_session=date(2024, 1, 5),
        symbols=("SYNB", "SYNA"),
        lookback=2,
    )
    loaded = store.load(dataset.identity(), symbols=("SYNB", "SYNA"))
    assert loaded.window(decision_session=date(2024, 1, 5), lookback=2).sessions == window.sessions
    np.testing.assert_array_equal(
        loaded.window(decision_session=date(2024, 1, 5), lookback=2).micro["close"],
        window.micro["close"],
    )
    assert window.sessions == (date(2024, 1, 3), date(2024, 1, 4))
    assert window.symbols == ("SYNB", "SYNA")
    np.testing.assert_array_equal(window.micro["close"][:, 0], dataset.series["SYNB"].micro["close"][1:3])
    np.testing.assert_array_equal(window.volume[:, 0], dataset.series["SYNB"].volume[1:3])
    assert not window.present[0, 1]
    assert window.present[1, 1]
    assert not window.micro["close"].flags.writeable
    with pytest.raises(ValueError):
        window.micro["close"].flags.writeable = True
    with pytest.raises(TypeError):
        window.micro["close"] = np.empty((0, 0))

    empty = store.window(dataset.identity(), decision_session=date(2024, 1, 2), symbols=())
    assert empty.sessions == () and empty.present.shape == (0, 0)


def test_panel_rejects_corruption_and_bad_requests(tmp_path: Path) -> None:
    dataset = synthetic_dataset(date(2024, 1, 2), date(2024, 1, 5), symbols=("SYNA",))
    store = PanelStore(tmp_path / "cache")
    path = store.build(dataset)
    identity = dataset.identity()
    with pytest.raises(LibraryError, match="DATA_PANEL_SYMBOL_INVALID"):
        store.window(identity, decision_session=date(2024, 1, 5), symbols=("SYNA", "SYNA"))
    with pytest.raises(ValueError, match="DATA_PANEL_LOOKBACK_INVALID"):
        store.window(identity, decision_session=date(2024, 1, 5), lookback=0)
    with pytest.raises(LibraryError, match="DATA_PANEL_INVALID"):
        store.path("../../outside")

    (path / "close.parquet").write_bytes(b"corrupt")
    with pytest.raises(LibraryError, match="DATA_PANEL_INVALID"):
        store.window(identity, decision_session=date(2024, 1, 5))
    with pytest.raises(LibraryError, match="DATA_PANEL_INVALID"):
        store.build(dataset)


def test_panel_can_be_derived_from_verified_manifest(tmp_path: Path) -> None:
    start, end = date(2024, 1, 2), date(2024, 1, 8)
    source = synthetic_dataset(start, end, symbols=("SYNA",))
    fake = FakeAlpaca(source)
    client = AlpacaDataClient(
        "key", "secret", transport=fake, limiter=RateLimiter(sleep=lambda _: None), sleep=lambda _: None
    )
    bar_pages = client.daily_bars(("SYNA",), start, end, "sip")
    action_pages = client.corporate_actions(("SYNA",), start, end)
    dataset = build_dataset(bar_pages, action_pages, source="alpaca:sip")
    library = Library(tmp_path / "cache", tmp_path / "manifests")
    for page in (*bar_pages, *action_pages):
        library.store_page(page)
    manifest = make_manifest(
        provider="alpaca",
        feed="sip",
        symbols=("SYNA",),
        start=start,
        end=end,
        bar_pages=bar_pages,
        action_pages=action_pages,
        dataset=dataset,
        fetched_at=datetime(2024, 1, 9, tzinfo=UTC),
    )
    store = PanelStore(library.cache_dir)
    assert store.build_from_manifest(library, manifest).is_dir()
    window = store.window(manifest["dataset_identity"], decision_session=end)
    assert window.sessions[-1] < end
    assert window.micro["close"][-1, 0] == dataset.series["SYNA"].micro["close"][-2]

    altered = dict(manifest, dataset_identity="sha256:" + "0" * 64)
    with pytest.raises(LibraryError, match="DATA_MANIFEST_INVALID"):
        store.build_from_manifest(library, altered)

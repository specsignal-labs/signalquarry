# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import signalquarry.api.factor as factor_api
from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.data.dataset import MICRO, Dataset, SymbolSeries
from signalquarry._internal.data.panel import LoadedPanel
from signalquarry._internal.data.universe_build import UNIVERSE_BUILD_SCHEMA
from signalquarry.cli.main import main


def _project(tmp_path: Path) -> Path:
    package = "factor_eval_" + uuid.uuid4().hex[:8]
    root = tmp_path / "project"
    directory = root / "src" / package
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text("", encoding="utf-8")
    (directory / "factor.py").write_text(
        "from signalquarry.sdk import FactorCtx, Params, factor\n"
        "class P(Params):\n    period: int = 2\n"
        "@factor(params=P, lookback=lambda p: p.period)\n"
        "def score(ctx: FactorCtx, p: P) -> dict[str, float]:\n"
        "    del p\n"
        "    close = ctx.panel('close')\n"
        "    return {symbol: float(close[-1, i]) for i, symbol in enumerate(ctx.universe)}\n",
        encoding="utf-8",
    )
    (directory / "factor.yaml").write_text(
        "schema: signalquarry.factor/v1\n"
        "id: price-momentum\nfamily: momentum\nversion: '1'\n"
        "hypothesis:\n  statement: Past prices predict next returns.\n"
        "  falsification: Rank IC is nonpositive out of sample.\n"
        "params:\n  period: 2\n"
        "evaluation:\n  horizons: [1, 5]\n  chronological_blocks: 2\n"
        "  cost_bps: 15\n  capital: 1000000\n",
        encoding="utf-8",
    )
    (root / "signalquarry.toml").write_text(
        '[project]\nname = "demo"\n[factors]\nmodules = ["' + package + '.factor"]\n',
        encoding="utf-8",
    )
    return root


def _dataset() -> Dataset:
    sessions = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(80))
    index = np.arange(len(sessions), dtype=np.float64)
    series: dict[str, SymbolSeries] = {}
    for number in range(10):
        symbol = f"S{number:02}"
        close = np.rint((100 + number * 2) * (1 + (number + 1) * 0.001 * index) * MICRO).astype(np.int64)
        series[symbol] = SymbolSeries(
            micro={name: close.copy() for name in ("open", "high", "low", "close")},
            volume=np.full(len(sessions), 1_000.0),
            present=np.ones(len(sessions), dtype=bool),
        )
    return Dataset(sessions, series, source="synthetic:factor-evaluation")


def _panel(dataset: Dataset, symbols: tuple[str, ...] | None = None) -> LoadedPanel:
    selected = dataset.symbols if symbols is None else symbols
    return LoadedPanel(
        dataset.identity(),
        dataset.sessions,
        selected,
        MappingProxyType(
            {
                name: np.column_stack([dataset.series[symbol].micro[name] for symbol in selected])
                for name in ("open", "high", "low", "close")
            }
        ),
        np.column_stack([dataset.series[symbol].volume for symbol in selected]),
        np.column_stack([dataset.series[symbol].present for symbol in selected]),
    )


def _universe_manifests(root: Path, dataset_manifest: dict[str, str], dataset: Dataset) -> list[Path]:
    build_dir = root / "data" / "universe" / "builds"
    build_dir.mkdir(parents=True)
    result = []
    for number in (30, 50):
        session = dataset.sessions[number]
        known_at = datetime.combine(session - timedelta(days=1), datetime.min.time(), UTC)
        members = list(dataset.symbols)
        body = {
            "schema": UNIVERSE_BUILD_SCHEMA,
            "decision_session": session.isoformat(),
            "known_at": known_at.isoformat(),
            "dataset_id": dataset_manifest["dataset_id"],
            "dataset_identity": dataset_manifest["dataset_identity"],
            "dataset_manifest_hash": dataset_manifest["manifest_hash"],
            "member_symbols": members,
            "members_hash": canonical_hash(members),
            "redistributable": False,
        }
        manifest = {**body, "manifest_hash": canonical_hash(body)}
        path = build_dir / f"{number}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        result.append(path.relative_to(root))
    return result


def _stub_data(monkeypatch: pytest.MonkeyPatch, dataset: Dataset, dataset_manifest: dict) -> list[str]:
    class Library:
        cache_dir = Path("/tmp/signalquarry-factor-evaluate-cache")

        @staticmethod
        def manifests() -> list[dict]:
            return [dataset_manifest]

    class PanelStoreStub:
        def __init__(self, cache_dir: Path) -> None:
            assert cache_dir == Library.cache_dir

        @staticmethod
        def build(_: Dataset) -> None:
            return None

        @staticmethod
        def load(_: str, *, symbols: tuple[str, ...] | None = None) -> LoadedPanel:
            return _panel(dataset, symbols)

    monkeypatch.setattr(factor_api, "library_for", lambda _: Library())
    monkeypatch.setattr(factor_api, "dataset_from_manifest", lambda *_: dataset)
    monkeypatch.setattr(factor_api, "PanelStore", PanelStoreStub)
    replayed: list[str] = []
    monkeypatch.setattr(
        factor_api,
        "replay_universe_build",
        lambda _root, manifest: replayed.append(manifest["manifest_hash"]),
    )
    return replayed


def test_factor_evaluate_cli_returns_unverified_metrics_without_trial_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path)
    dataset = _dataset()
    dataset_manifest = {
        "dataset_id": "research-data",
        "dataset_identity": dataset.identity(),
        "manifest_hash": "sha256:" + "a" * 64,
    }
    replayed = _stub_data(monkeypatch, dataset, dataset_manifest)
    paths = _universe_manifests(root, dataset_manifest, dataset)

    assert (
        main(
            [
                "--json",
                "factor",
                "evaluate",
                "--factor",
                "price-momentum",
                "--dataset-id",
                "research-data",
                "--universe-manifest",
                str(paths[0]),
                "--universe-manifest",
                str(paths[1]),
                "--project",
                str(root),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "factor evaluate"
    assert payload["data"]["scope"] == "unverified"
    assert payload["data"]["authority"] == {
        "provenance_verified": False,
        "trial_recorded": False,
        "evidence_grade": None,
        "holdout_accessed": False,
    }
    assert [row["horizon"] for row in payload["data"]["horizons"]] == [1, 5]
    assert len(payload["data"]["decision_sessions"]) == 2
    assert len(replayed) == 2
    assert not (root / "evidence" / "trials.jsonl").exists()
    assert not (root / "evidence" / "factor_trials.jsonl").exists()


def test_factor_evaluate_rejects_a_universe_manifest_for_another_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    dataset = _dataset()
    dataset_manifest = {
        "dataset_id": "research-data",
        "dataset_identity": dataset.identity(),
        "manifest_hash": "sha256:" + "a" * 64,
    }
    _stub_data(monkeypatch, dataset, dataset_manifest)
    paths = _universe_manifests(root, dataset_manifest, dataset)
    manifest_path = root / paths[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_id"] = "different-dataset"
    manifest["manifest_hash"] = canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = factor_api.factor_evaluate(
        "price-momentum", "research-data", universe_manifests=[paths[0]], project=root
    )

    assert result.status == "invalid"
    assert result.reason_codes == ["DATA_MANIFEST_INVALID"]


def test_factor_evaluate_rejects_a_tampered_universe_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    dataset = _dataset()
    dataset_manifest = {
        "dataset_id": "research-data",
        "dataset_identity": dataset.identity(),
        "manifest_hash": "sha256:" + "a" * 64,
    }
    _stub_data(monkeypatch, dataset, dataset_manifest)
    paths = _universe_manifests(root, dataset_manifest, dataset)
    manifest_path = root / paths[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["known_at"] = "2024-01-01T00:00:00+00:00"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = factor_api.factor_evaluate(
        "price-momentum", "research-data", universe_manifests=[paths[0]], project=root
    )

    assert result.status == "invalid"
    assert result.reason_codes == ["DATA_MANIFEST_INVALID"]


def test_factor_evaluate_requires_explicit_universe_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    dataset = _dataset()
    dataset_manifest = {
        "dataset_id": "research-data",
        "dataset_identity": dataset.identity(),
        "manifest_hash": "sha256:" + "a" * 64,
    }
    _stub_data(monkeypatch, dataset, dataset_manifest)

    result = factor_api.factor_evaluate(
        "price-momentum", "research-data", universe_manifests=[], project=root
    )

    assert result.status == "unavailable"
    assert result.reason_codes == ["UNIVERSE_INPUT_UNAVAILABLE"]

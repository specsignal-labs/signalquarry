# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import tomllib
import uuid
from datetime import UTC, date, time, timedelta
from pathlib import Path

import pytest

from signalquarry import api
from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.data.panel import LoadedPanel, PanelStore
from signalquarry._internal.factors.evaluate import ScorePanel, UniverseAt
from signalquarry.cli.main import main
from signalquarry.sdk import FactorDef, Params


def _sqy(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, dict]:
    code = main(["--json", *args])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    package = "offline_factor_" + uuid.uuid4().hex
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(cache))
    for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    result = api.init(root, demo=True, kind="factor", package=package)
    assert result.status == "ok"
    yield root
    assert not (root / ".signalquarry").exists()
    assert not list((root / "evidence").rglob("*.jsonl"))
    assert list(cache.iterdir()) == []


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_factor_demo_cli_workflow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "demo"
    package = "offline_cli_" + uuid.uuid4().hex
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(cache))
    code, created = _sqy(capsys, "init", str(root), "--demo", "--kind", "factor", "--package", package)
    assert code == 0
    assert [action["command"] for action in created["next_actions"]][1:] == [
        "sqy factor ls",
        "sqy check --factor-id volume-shock",
        "sqy factor evaluate --factor volume-shock --synthetic",
    ]
    config = tomllib.loads((root / "signalquarry.toml").read_text(encoding="utf-8"))
    assert config["factors"]["modules"] == [f"{package}.volume_shock.factor"]
    assert config["strategies"]["modules"] == [f"{package}.sma_trend.strategy"]
    factor_dir = root / "src" / package / "volume_shock"
    assert {
        f"src/{package}/volume_shock/{name}" for name in ("__init__.py", "factor.py", "factor.yaml")
    } <= set(created["data"]["files"])
    hypothesis = (factor_dir / "factor.yaml").read_text(encoding="utf-8")
    assert "demonstration on synthetic data" in hypothesis
    assert "on purpose" in hypothesis

    code, listed = _sqy(capsys, "factor", "ls", "--project", str(root))
    assert code == 0
    assert [item["id"] for item in listed["data"]["factors"]] == ["volume-shock"]
    assert listed["data"]["factors"][0]["family"] == "volume-shock"
    code, checked = _sqy(capsys, "check", "--factor-id", "volume-shock", "--project", str(root))
    assert code == 0
    assert all(check["ok"] for check in checked["data"]["checks"])
    before = _files(root)
    code, evaluated = _sqy(
        capsys, "factor", "evaluate", "--factor", "volume-shock", "--synthetic", "--project", str(root)
    )
    assert code == 0
    assert evaluated["data"]["synthetic"]["symbols"] == 200
    assert evaluated["data"]["horizons"][0]["mean_ic"] > 0
    assert evaluated["evidence"] == {"grade": "synthetic", "claim_level": "none"}
    assert not evaluated["artifacts"]
    assert _files(root) == before
    assert not (root / ".signalquarry").exists()
    assert not list((root / "evidence").rglob("*.jsonl"))
    assert list(cache.iterdir()) == []


def test_synthetic_diagnostics_shape_determinism_and_signal(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = api.factor_evaluate("volume-shock", synthetic=True, synthetic_symbols=60, project=project)
    assert result.status == "ok"
    data = result.data
    assert set(data) == {
        "factor",
        "family",
        "configuration_hash",
        "code_tree_hash",
        "dataset_id",
        "dataset_identity",
        "universe_identity",
        "decision_sessions",
        "label_identity",
        "scope",
        "authority",
        "evaluation",
        "conformance_checks",
        "horizons",
        "synthetic",
    }
    assert data["factor"] == data["family"] == "volume-shock"
    assert data["scope"] == "synthetic"
    assert data["dataset_id"] == "synthetic-panel"
    assert data["synthetic"] == {
        "symbols": 60,
        "seed": 7,
        "planted_ic": 0.05,
        "decision_sessions": len(data["decision_sessions"]),
    }
    assert data["synthetic"]["decision_sessions"] == 124
    assert data["authority"] == {
        "provenance_verified": False,
        "trial_recorded": False,
        "evidence_grade": None,
        "holdout_accessed": False,
    }
    assert result.evidence == {"grade": "synthetic", "claim_level": "none"}
    assert "synthetic diagnostics" in result.summary
    assert "say nothing about real markets" in result.summary
    assert "no trial recorded" in result.summary
    assert "no holdout touched" in result.summary
    assert [check["name"] for check in data["conformance_checks"]] == [
        "import_policy",
        "contract",
        "determinism",
        "lookahead",
    ]
    assert all(check["ok"] for check in data["conformance_checks"])
    assert data["evaluation"] == {
        "horizons": [1, 5, 21],
        "chronological_blocks": 6,
        "cost_bps": "10",
        "capital": "1000000",
    }
    assert [row["horizon"] for row in data["horizons"]] == [1, 5, 21]
    assert data["horizons"][0]["mean_ic"] > 0
    assert data["horizons"][0]["scored_pairs"] > 0

    before = _files(project)
    code, repeated = _sqy(
        capsys,
        "factor",
        "evaluate",
        "--factor",
        "volume-shock",
        "--synthetic",
        "--symbols",
        "60",
        "--seed",
        "7",
        "--planted-ic",
        "0.05",
        "--project",
        str(project),
    )
    assert code == 0
    assert repeated["data"]["horizons"] == data["horizons"]
    changed = api.factor_evaluate(
        "volume-shock", synthetic=True, synthetic_symbols=60, synthetic_seed=8, project=project
    )
    assert changed.status == "ok"
    assert changed.data["horizons"] != data["horizons"]
    zero = api.factor_evaluate(
        "volume-shock", synthetic=True, synthetic_symbols=60, synthetic_planted_ic=0.0, project=project
    )
    assert zero.status == "ok"
    assert abs(zero.data["horizons"][0]["mean_ic"]) < 0.02
    assert _files(project) == before


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"universe_manifests": [Path("universe.json")]},
        {"synthetic": True, "dataset_id": "data"},
        {"synthetic": True, "universe_manifests": [Path("universe.json")]},
        {"synthetic": True, "dataset_id": "data", "universe_manifests": [Path("universe.json")]},
    ],
)
def test_evaluation_mode_usage_errors(project: Path, options: dict) -> None:
    result = api.factor_evaluate("volume-shock", project=project, **options)
    assert result.status == "usage"
    assert result.reason_codes == ["USAGE_INVALID"]


@pytest.mark.parametrize(
    ("options", "name"),
    [
        ({"synthetic_symbols": 1}, "n_symbols"),
        ({"synthetic_symbols": 3001}, "n_symbols"),
        ({"synthetic_seed": True}, "seed"),
        ({"synthetic_planted_ic": -0.01}, "planted_ic"),
        ({"synthetic_planted_ic": 0.51}, "planted_ic"),
        ({"synthetic_planted_ic": float("nan")}, "planted_ic"),
    ],
)
def test_synthetic_argument_usage_errors(project: Path, options: dict, name: str) -> None:
    result = api.factor_evaluate("volume-shock", synthetic=True, project=project, **options)
    assert result.status == "usage"
    assert result.reason_codes == ["USAGE_INVALID"]
    assert name in result.summary


@pytest.mark.parametrize(
    "flags",
    [
        (),
        ("--synthetic", "--dataset-id", "data"),
        ("--synthetic", "--universe-manifest", "universe.json"),
        ("--synthetic", "--symbols", "1"),
    ],
)
def test_cli_evaluation_usage_errors(
    project: Path, capsys: pytest.CaptureFixture[str], flags: tuple[str, ...]
) -> None:
    code, result = _sqy(
        capsys, "factor", "evaluate", "--factor", "volume-shock", "--project", str(project), *flags
    )
    assert code == 64
    assert result["status"] == "usage"
    assert result["reason_codes"] == ["USAGE_INVALID"]


@pytest.mark.parametrize("flags", [(), ("--lab",), ("--demo", "--lab")])
def test_factor_kind_requires_demo_without_lab(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flags: tuple[str, ...]
) -> None:
    root = tmp_path / "invalid"
    code, result = _sqy(capsys, "init", str(root), "--kind", "factor", *flags)
    assert code == 64
    assert result["status"] == "usage"
    assert result["reason_codes"] == ["USAGE_INVALID"]
    assert "requires --demo" in result["summary"]
    assert "--lab" in result["summary"]
    assert not root.exists()


def test_unknown_factor_and_missing_real_dataset(project: Path) -> None:
    unknown = api.factor_evaluate("unknown", synthetic=True, project=project)
    assert unknown.status == "invalid"
    assert unknown.reason_codes == ["FACTOR_NOT_FOUND"]
    assert unknown.summary == "unknown"
    assert unknown.data == {"factors": ["volume-shock"]}
    missing = api.factor_evaluate(
        "volume-shock", "missing", universe_manifests=[Path("universe.json")], project=project
    )
    assert missing.status == "invalid"
    assert missing.reason_codes == ["DATA_MANIFEST_INVALID"]
    assert missing.summary == "DATA_MANIFEST_INVALID:expected one manifest for dataset missing"
    assert missing.evidence is None


def test_synthetic_membership_timing_and_temporary_cleanup(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import signalquarry.api.factor as factor_api

    directories: list[Path] = []
    original_score = factor_api.score_factor

    def store(cache_dir: Path) -> PanelStore:
        directories.append(cache_dir)
        assert not cache_dir.is_relative_to(project)
        return PanelStore(cache_dir)

    def score(
        definition: FactorDef, params: Params, panel: LoadedPanel, memberships: tuple[UniverseAt, ...]
    ) -> ScorePanel:
        for member in memberships:
            assert member.observed_at == member.decision_cutoff
            assert member.observed_at.tzinfo == UTC
            assert member.observed_at.time() == time(21)
            assert member.observed_at.date() == member.session - timedelta(days=1)
            assert member.identity == canonical_hash({"session": member.session, "symbols": member.symbols})
        assert memberships[0].session == date(2016, 2, 1)
        assert all(
            panel.sessions.index(right.session) - panel.sessions.index(left.session) == 21
            for left, right in zip(memberships, memberships[1:], strict=False)
        )
        return original_score(definition, params, panel, memberships)

    monkeypatch.setattr(factor_api, "PanelStore", store)
    monkeypatch.setattr(factor_api, "score_factor", score)
    result = api.factor_evaluate("volume-shock", synthetic=True, synthetic_symbols=60, project=project)
    assert result.status == "ok"
    assert directories and all(not path.exists() for path in directories)

    def fail(*_: object) -> ScorePanel:
        assert directories[-1].exists()
        raise ValueError("FACTOR_LABEL_ALIGNMENT_INVALID")

    monkeypatch.setattr(factor_api, "score_factor", fail)
    result = api.factor_evaluate("volume-shock", synthetic=True, synthetic_symbols=60, project=project)
    assert result.status == "invalid"
    assert result.reason_codes == ["FACTOR_LABEL_ALIGNMENT_INVALID"]
    assert all(not path.exists() for path in directories)


def test_a_dataset_without_universe_manifests_keeps_its_earlier_error(project: Path) -> None:
    # Not a usage error: the real-data path reports the missing dated universe, as before.
    result = api.factor_evaluate("volume-shock", "data", project=project)
    assert result.status != "usage" and result.reason_codes != ["USAGE_INVALID"]

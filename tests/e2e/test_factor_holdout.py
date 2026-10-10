# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import fcntl
import importlib
import json
import tomllib
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from signalquarry import api
from signalquarry._internal.data.alpaca import AlpacaDataClient, RateLimiter
from signalquarry._internal.data.library import build_dataset, make_manifest
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.project.factors import load_factors
from signalquarry._internal.project.project import load_config
from signalquarry._internal.validation import ledger
from signalquarry.cli.main import command_catalog, main
from tests.fakes import FakeAlpaca


def _bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and "evidence" in path.relative_to(root).parts
    }


def _add_factor(
    root: Path,
    identifier: str,
    *,
    family: str = "alpha",
    budget: int = 5,
    months: int = 3,
    cutoff: str | None = None,
) -> None:
    package = root.name.replace("-", "_")
    directory = root / "src" / package / identifier
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text("")
    (directory / "factor.py").write_text(
        "from signalquarry.sdk import Params, factor\n"
        "@factor(params=Params, lookback=lambda p: 1)\n"
        "def score(ctx, p):\n    return {}\n"
    )
    (directory / "factor.yaml").write_text(
        f"id: {identifier}\nfamily: {family}\nversion: '1'\n"
        "hypothesis:\n  statement: Past prices predict returns.\n  falsification: IC is nonpositive.\n"
        f"evaluation:\n  trial_budget: {budget}\n  holdout:\n    months: {months}\n"
        + (f"    training_cutoff: '{cutoff}'\n" if cutoff else "")
    )
    config_path = root / "signalquarry.toml"
    text = config_path.read_text()
    previous = tomllib.loads(text)["factors"]["modules"]
    modules = [*previous, f"{package}.{identifier}.factor"]
    config_path.write_text(
        text.replace("modules = " + json.dumps(previous), "modules = " + json.dumps(modules))
    )
    importlib.invalidate_caches()


@pytest.fixture(params=["project", "per_family"])
def project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> tuple[Path, str]:
    root = tmp_path / ("holdout_" + uuid.uuid4().hex[:8])
    package = root / "src" / root.name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (root / "signalquarry.toml").write_text(
        f'[factors]\nmodules = []\n[evidence]\nlayout = "{request.param}"\n'
    )
    _add_factor(root, "momentum")
    monkeypatch.setenv("SIGNALQUARRY_CACHE_DIR", str(tmp_path / "cache"))
    start, end = date(2023, 1, 3), date(2024, 3, 28)
    source = synthetic_dataset(start, end, symbols=("SYNA",))
    client = AlpacaDataClient(
        "key",
        "secret",
        transport=FakeAlpaca(source),
        limiter=RateLimiter(sleep=lambda _: None),
        sleep=lambda _: None,
    )
    bars = client.daily_bars(("SYNA",), start, end, "sip")
    actions = client.corporate_actions(("SYNA",), start, end)
    dataset = build_dataset(bars, actions, source="alpaca:sip")
    library = api.factor.library_for(root)
    for page in (*bars, *actions):
        library.store_page(page)
    manifest = make_manifest(
        provider="alpaca",
        feed="sip",
        symbols=("SYNA",),
        start=start,
        end=end,
        bar_pages=bars,
        action_pages=actions,
        dataset=dataset,
        fetched_at=datetime(2024, 4, 1, tzinfo=UTC),
    )
    library.write_manifest(manifest)
    ledger.append(root, "trials", "other", {"kind": "budget_extension", "family": "other", "added": 10})
    return root, manifest["dataset_id"]


def _sqy(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, dict[str, Any]]:
    code = main(["--json", *arguments])
    return code, json.loads(capsys.readouterr().out)


def test_factor_holdout_seal_records_effective_declaration_dataset_and_contributors(
    project: tuple[Path, str],
) -> None:
    root, dataset_id = project
    _add_factor(root, "value", budget=10, months=12, cutoff="2023-01-31")
    _add_factor(root, "quality", budget=2, months=6, cutoff="2023-02-28")
    expected_factors = load_factors(load_config(root))
    result = api.factor_holdout_seal("alpha", dataset_id, project=root)
    assert result.status == "ok" and result.data["newly_sealed"] is True
    seal = result.data["seal"]
    assert seal["schema"] == "signalquarry.factor-holdout/v1"
    assert seal["trial_budget"] == 2
    assert seal["holdout"] == {"months": 12, "training_cutoff": "2023-01-31"}
    assert seal["holdout_start"] == "2023-02-01"
    assert seal["dataset_id"] == dataset_id
    assert seal["dataset_last_session"] == "2024-03-28"
    assert seal["dataset_identity"] == api.factor.library_for(root).manifests()[0]["dataset_identity"]
    assert seal["factors"] == [
        {
            "id": identifier,
            "configuration_hash": item.configuration_hash,
            "trial_budget": item.spec.evaluation.trial_budget,
            "holdout": item.spec.evaluation.holdout.model_dump(mode="json"),
        }
        for identifier, item in sorted(expected_factors.items())
    ]
    assert result.warnings == ["HUMAN_ACTION_RECORDED"]


def test_factor_holdout_seal_repeat_writes_nothing(project: tuple[Path, str]) -> None:
    root, dataset_id = project
    first = api.factor_holdout_seal("alpha", dataset_id, project=root)
    before = _bytes(root)
    repeated = api.factor_holdout_seal("alpha", dataset_id, project=root)
    assert repeated.data["newly_sealed"] is False
    assert repeated.data["seal"] == first.data["seal"]
    assert repeated.warnings == []
    assert repeated.data["differences"] == {}
    assert _bytes(root) == before


@pytest.mark.parametrize(("budget", "months", "cutoff"), [(10, 1, None), (1, 24, "2022-12-30")])
def test_factor_holdout_later_lenient_or_stricter_declaration_warns_without_resealing(
    project: tuple[Path, str], budget: int, months: int, cutoff: str | None
) -> None:
    root, dataset_id = project
    first = api.factor_holdout_seal("alpha", dataset_id, project=root)
    before = _bytes(root)
    _add_factor(root, "later", budget=budget, months=months, cutoff=cutoff)
    repeated = api.factor_holdout_seal("alpha", dataset_id, project=root)
    assert repeated.data["seal"] == first.data["seal"] and repeated.data["newly_sealed"] is False
    assert repeated.warnings == ["FACTOR_HOLDOUT_DECLARATION_CHANGED"]
    differences = repeated.data["differences"]
    assert len(differences["factors"]["sealed"]) == 1
    assert len(differences["factors"]["current"]) == 2
    if budget < 5:
        assert differences["trial_budget"] == {"sealed": 5, "current": 1}
        assert differences["holdout"]["current"] == {"months": 24, "training_cutoff": "2022-12-30"}
    else:
        assert "trial_budget" not in differences and "holdout" not in differences
    assert _bytes(root) == before


def test_factor_holdout_existing_seal_survives_removed_registration_and_dataset(
    project: tuple[Path, str],
) -> None:
    root, dataset_id = project
    first = api.factor_holdout_seal("alpha", dataset_id, project=root)
    before = _bytes(root)
    config = root / "signalquarry.toml"
    modules = tomllib.loads(config.read_text())["factors"]["modules"]
    config.write_text(config.read_text().replace(json.dumps(modules), "[]"))
    for path in (root / "data/manifests").glob("*.json"):
        path.unlink()
    repeated = api.factor_holdout_seal("alpha", "missing-dataset", project=root)
    assert repeated.data["seal"] == first.data["seal"]
    assert repeated.warnings == ["FACTOR_HOLDOUT_DECLARATION_CHANGED"]
    assert _bytes(root) == before


@pytest.mark.parametrize("family", ["missing", "other"])
def test_factor_holdout_unknown_or_strategy_only_family_refuses_without_writing(
    project: tuple[Path, str], family: str
) -> None:
    root, dataset_id = project
    before = _bytes(root)
    result = api.factor_holdout_seal(family, dataset_id, project=root)
    assert (result.status, result.reason_codes) == ("invalid", ["FACTOR_FAMILY_EMPTY"])
    assert _bytes(root) == before


def test_factor_holdout_no_declaration_refuses_without_writing(project: tuple[Path, str]) -> None:
    root, dataset_id = project
    spec = root / "src" / root.name / "momentum/factor.yaml"
    spec.write_text(spec.read_text().replace("months: 3", "months: 0"))
    before = _bytes(root)
    result = api.factor_holdout_seal("alpha", dataset_id, project=root)
    assert (result.status, result.reason_codes) == ("invalid", ["FACTOR_HOLDOUT_UNDECLARED"])
    assert _bytes(root) == before


def test_factor_holdout_unknown_dataset_refuses_without_writing(project: tuple[Path, str]) -> None:
    root, _ = project
    before = _bytes(root)
    result = api.factor_holdout_seal("alpha", "missing", project=root)
    assert result.reason_codes == ["DATA_MANIFEST_INVALID"]
    assert _bytes(root) == before


@pytest.mark.parametrize("damage", ["json", "object", "hash", "missing-page", "corrupt-page", "duplicate"])
def test_factor_holdout_unreadable_dataset_refuses_without_writing(
    project: tuple[Path, str], damage: str
) -> None:
    root, dataset_id = project
    library = api.factor.library_for(root)
    path = root / "data/manifests" / f"{dataset_id}.json"
    manifest = library.manifests()[0]
    expected = "DATA_MANIFEST_INVALID"
    if damage == "json":
        path.write_text("{")
    elif damage == "object":
        path.write_text("[]")
    elif damage == "hash":
        manifest["dataset_identity"] = "forged"
        path.write_text(json.dumps(manifest))
    elif damage == "missing-page":
        library.page_path(manifest["bar_pages"][0]["sha256"]).unlink()
        expected = "DATA_PAGE_MISSING"
    elif damage == "corrupt-page":
        import gzip

        library.page_path(manifest["bar_pages"][0]["sha256"]).write_bytes(gzip.compress(b"{}"))
        expected = "DATA_PAGE_CORRUPT"
    else:
        (path.parent / "duplicate.json").write_bytes(path.read_bytes())
    before = _bytes(root)
    result = api.factor_holdout_seal("alpha", dataset_id, project=root)
    assert (result.status, result.reason_codes) == ("invalid", [expected])
    assert _bytes(root) == before


def test_factor_holdout_bad_family_path_refuses_without_writing(project: tuple[Path, str]) -> None:
    root, dataset_id = project
    before = _bytes(root)
    assert api.factor_holdout_seal("../outside", dataset_id, project=root).reason_codes == ["USAGE_INVALID"]
    assert api.factor_holdout_status(family="../outside", project=root).reason_codes == ["USAGE_INVALID"]
    assert _bytes(root) == before


def test_factor_holdout_busy_seal_refuses_without_writing(project: tuple[Path, str]) -> None:
    root, dataset_id = project
    before = _bytes(root)
    with (root / "signalquarry.toml").open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        result = api.factor_holdout_seal("alpha", dataset_id, project=root)
    assert (result.status, result.exit_code, result.reason_codes) == ("busy", 75, ["FACTOR_EVIDENCE_BUSY"])
    assert _bytes(root) == before


def test_factor_holdout_status_is_read_only_and_reports_sealed_and_unsealed_families(
    project: tuple[Path, str],
) -> None:
    root, dataset_id = project
    api.factor_holdout_seal("alpha", dataset_id, project=root)
    _add_factor(root, "unsealed", family="beta")
    ledger.append(
        root,
        "factor_trials",
        "alpha",
        {"kind": "factor_trial", "family": "alpha", "metrics": {"p_value": 0.1}},
    )
    before = _bytes(root)
    result = api.factor_holdout_status(project=root)
    assert result.status == "ok"
    alpha, beta = result.data["families"]
    assert (
        alpha["family"],
        alpha["holdout_start"],
        alpha["trial_budget"],
        alpha["trials_used"],
        alpha["remaining_budget"],
        alpha["dataset_id"],
        alpha["opened"],
    ) == ("alpha", "2023-12-29", 5, 1, 4, dataset_id, False)
    assert alpha["holdout"] == {"months": 3, "training_cutoff": None}
    assert (beta["family"], beta["state"], beta["seal"], beta["trial_budget"], beta["opened"]) == (
        "beta",
        "not_sealed",
        None,
        None,
        False,
    )
    assert api.factor_holdout_status(family="alpha", project=root).data["families"] == [alpha]
    assert _bytes(root) == before


def test_factor_holdout_status_unknown_family_refuses_without_writing(project: tuple[Path, str]) -> None:
    root, _ = project
    before = _bytes(root)
    assert api.factor_holdout_status(family="missing", project=root).reason_codes == ["FACTOR_FAMILY_EMPTY"]
    assert _bytes(root) == before


@pytest.mark.parametrize("project", ["per_family"], indirect=True)
def test_factor_holdout_per_family_seal_is_indexed_and_verifies(
    project: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    root, dataset_id = project
    seal = api.factor_holdout_seal("alpha", dataset_id, project=root).data["seal"]
    path = root / "families/alpha/evidence/factor_holdouts.jsonl"
    assert path.is_file() and not (root / "evidence/factor_holdouts.jsonl").exists()
    index = ledger.project_index(root).entries()[-1]
    assert (index["family"], index["log"], index["head"], index["entries"]) == (
        "alpha",
        "factor_holdouts",
        seal["hash"],
        1,
    )
    code, payload = _sqy(capsys, "evidence", "verify", "--project", str(root))
    assert code == 0 and payload["status"] == "ok"
    assert "families/alpha/evidence/factor_holdouts.jsonl" in {row["path"] for row in payload["data"]["logs"]}


def test_factor_holdout_edited_seal_is_detected(
    project: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    root, dataset_id = project
    api.factor_holdout_seal("alpha", dataset_id, project=root)
    path = ledger.factor_holdouts(root, "alpha").path
    path.write_text(path.read_text().replace('"trial_budget":5', '"trial_budget":500'))
    before = _bytes(root)
    code, payload = _sqy(capsys, "evidence", "verify", "--project", str(root))
    assert (code, payload["reason_codes"]) == (2, ["EVIDENCE_LOG_CORRUPT"])
    assert api.factor_holdout_seal("alpha", dataset_id, project=root).reason_codes == ["EVIDENCE_LOG_CORRUPT"]
    assert api.factor_holdout_status(project=root).reason_codes == ["EVIDENCE_LOG_CORRUPT"]
    assert _bytes(root) == before


def test_factor_holdout_cli_envelopes_and_exit_codes(
    project: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    root, dataset_id = project
    code, sealed = _sqy(
        capsys,
        "factor",
        "holdout",
        "seal",
        "--family",
        "alpha",
        "--dataset-id",
        dataset_id,
        "--project",
        str(root),
    )
    assert (code, sealed["command"], sealed["schema"], sealed["status"], sealed["data"]["newly_sealed"]) == (
        0,
        "factor holdout seal",
        "signalquarry.cli/v1",
        "ok",
        True,
    )
    code, status = _sqy(capsys, "factor", "holdout", "status", "--family", "alpha", "--project", str(root))
    assert (code, status["command"], status["data"]["families"][0]["opened"]) == (
        0,
        "factor holdout status",
        False,
    )
    code, invalid = _sqy(
        capsys,
        "factor",
        "holdout",
        "seal",
        "--family",
        "missing",
        "--dataset-id",
        dataset_id,
        "--project",
        str(root),
    )
    assert (code, invalid["reason_codes"]) == (65, ["FACTOR_FAMILY_EMPTY"])
    assert _sqy(capsys, "factor", "holdout", "seal", "--family", "alpha")[0] == 64
    assert _sqy(capsys, "factor", "holdout", "open")[0] == 64
    assert _sqy(capsys, "factor", "search")[0] == 64
    assert _sqy(capsys, "factor", "emit")[0] == 64


def test_factor_holdout_cli_help_marks_sealing_as_human_only(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["factor", "holdout", "seal", "--help"]) == 0
    help_text = capsys.readouterr().out
    assert "Explicit human command" in help_text
    assert "not an MCP tool" in help_text
    assert "--family" in help_text and "--dataset-id" in help_text


def test_factor_holdout_command_catalog_includes_both_nested_commands() -> None:
    factor = next(command for command in command_catalog() if command["name"] == "factor")
    subcommands = {command["name"]: command for command in factor["subcommands"]}
    assert set(subcommands) == {"ls", "evaluate", "search", "holdout seal", "holdout status"}
    assert "Explicit human command" in subcommands["holdout seal"]["help"]
    assert {tuple(option["flags"]) for option in subcommands["holdout seal"]["options"]} >= {
        ("--family",),
        ("--dataset-id",),
    }


def test_factor_holdout_api_missing_project_is_an_envelope(tmp_path: Path) -> None:
    assert api.factor_holdout_seal("alpha", "missing", project=tmp_path).reason_codes == ["PROJECT_NOT_FOUND"]
    assert api.factor_holdout_status(project=tmp_path).reason_codes == ["PROJECT_NOT_FOUND"]


def test_factor_holdout_mcp_discovery_has_no_sealing_opening_or_search_tools() -> None:
    pytest.importorskip("mcp")
    from mcp import Client

    from signalquarry.mcp.server import mcp

    async def exercise() -> None:
        async with Client(mcp) as client:
            tools = {item.name for item in (await client.list_tools()).tools}
            assert "sqy_factor_ls" in tools
            assert not any(
                "seal" in name or "holdout_open" in name or "factor_search" in name or "factor_emit" in name
                for name in tools
            )

    asyncio.run(exercise())

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import fcntl
import json
import os
import shutil
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from signalquarry import api
from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.data.library import LibraryError
from signalquarry._internal.data.synthetic import synthetic_panel
from signalquarry._internal.factors.search import search_expressions
from signalquarry._internal.validation import ledger
from signalquarry._internal.validation.factor_trials import factor_trial_accounting
from signalquarry.cli.main import command_catalog, main

from .test_factor_evaluate_api import _project, _stub_data, _universe_manifests


def _bytes(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for base in (root / "evidence", root / "families", root / ".signalquarry" / "factor_searches")
        if base.exists()
        for path in (base, *base.rglob("*"))
    }


@pytest.fixture
def search_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any]]:
    root = _project(tmp_path)
    dataset = synthetic_panel(
        date(2023, 1, 2),
        date(2023, 10, 31),
        n_symbols=24,
        planted_ic=0.5,
        late_listing_fraction=0,
        delisting_fraction=0,
        missing_rate=0,
        split_fraction=0,
        dividend_fraction=0,
    ).dataset
    spec = next(root.glob("src/*/factor.yaml"))
    spec.write_text(
        spec.read_text()
        + f"  trial_budget: 50\n  holdout:\n    months: 0\n    training_cutoff: '{dataset.sessions[-10]}'\n"
    )
    manifest = {
        "dataset_id": "research-data",
        "dataset_identity": dataset.identity(),
        "manifest_hash": "sha256:" + "a" * 64,
    }
    _stub_data(monkeypatch, dataset, manifest)
    paths = _universe_manifests(root, manifest, dataset)
    assert api.factor_holdout_seal("momentum", "research-data", project=root).status == "ok"
    return root, {
        "family": "momentum",
        "dataset_id": "research-data",
        "universe_manifests": paths,
        "training_cutoff": dataset.sessions[-20],
        "horizon": 1,
        "seed": 17,
        "budget": 18,
    }


def _search(root: Path, args: dict[str, Any], **changes: Any):
    return api.factor_search(project=root, **{**args, **changes})


def test_search_proposal_records_every_candidate_is_small_and_idempotent(search_project) -> None:
    root, args = search_project
    code_before = {
        str(path): path.read_bytes()
        for path in root.glob("src/**/*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    result = _search(root, args)
    assert result.status == "ok", result.as_dict()
    assert result.data["scope"] == "exploratory" and result.data["evidence_grade"] == "none"
    assert result.data["holdout_accessed"] is False
    assert result.data["trials_evaluated"] == 18
    assert len(result.data["candidates"]) == 10
    assert len(json.dumps(result.data).encode()) < 16_384
    assert result.artifacts[0]["path"].startswith(".signalquarry/factor_searches/")
    document = json.loads((root / result.artifacts[0]["path"]).read_text())
    rows = ledger.factor_trials(root).entries()
    assert len(rows) == len(document["candidates"]) == 18
    assert document["report"]["scope"] == "exploratory"
    planted = next(
        row for row in document["candidates"] if row["expression"] == "rank((volume / ts_mean(volume, 5)))"
    )
    assert planted["discovery"] and planted["mean_ic"] > 0.2
    assert all(row["metrics"]["search_identity"] == result.data["search_identity"] for row in rows)
    before = _bytes(root)
    second = _search(root, args)
    assert second.status == "ok", second.as_dict()
    assert second.data == result.data and second.artifacts == result.artifacts
    assert _bytes(root) == before
    assert code_before == {
        str(path): path.read_bytes()
        for path in root.glob("src/**/*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def test_search_crash_resume_matches_uninterrupted_ledger_and_hash(
    search_project, monkeypatch, tmp_path
) -> None:
    root, args = search_project
    fixed_at = datetime.now(UTC)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_at

    monkeypatch.setattr(api.factor, "datetime", FixedDatetime)
    twin = tmp_path / "twin"
    shutil.copytree(root, twin)
    # Keep registered code rooted in its original project; only evidence/artifacts are copied below.
    normal = _search(root, args)
    normal_bytes = _bytes(root)
    shutil.rmtree(root / "evidence")
    shutil.copytree(twin / "evidence", root / "evidence")
    shutil.rmtree(root / ".signalquarry")
    original = api.factor.record_formula_trial
    calls = 0

    def interrupted(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected crash")
        return original(*a, **kw)

    monkeypatch.setattr(api.factor, "record_formula_trial", interrupted)
    failed = _search(root, args)
    assert failed.status == "error"
    assert factor_trial_accounting(root, "momentum").family_trials_used == 3
    assert not (root / ".signalquarry/factor_searches").exists()
    monkeypatch.setattr(api.factor, "record_formula_trial", original)
    resumed = _search(root, args)
    assert resumed.status == "ok", resumed.as_dict()
    assert resumed.data["report_hash"] == normal.data["report_hash"]
    resumed_bytes = _bytes(root)
    for name, content in normal_bytes.items():
        if name.endswith("search.json"):
            first, second = json.loads(content), json.loads(resumed_bytes[name])
            first.pop("created_at")
            second.pop("created_at")
            assert first == second
        else:
            assert resumed_bytes[name] == content
    before = _bytes(root)
    assert _search(root, args).data["report_hash"] == resumed.data["report_hash"]
    assert before == _bytes(root)


@pytest.mark.parametrize(
    "change,code",
    [
        ({"family": "unsealed"}, "FACTOR_HOLDOUT_UNSEALED"),
        ({"dataset_id": "missing"}, "DATA_MANIFEST_INVALID"),
        ({"universe_manifests": []}, "UNIVERSE_INPUT_UNAVAILABLE"),
        ({"training_cutoff": date(2023, 10, 30)}, "FACTOR_TRAINING_WINDOW_INVALID"),
        ({"horizon": 30}, "FACTOR_TRAINING_WINDOW_INVALID"),
        ({"seed": -1}, "FACTOR_SEARCH_CONFIG_INVALID"),
        ({"budget": 51}, "FACTOR_SEARCH_BUDGET_EXHAUSTED"),
        ({"budget": -1}, "FACTOR_SEARCH_CONFIG_INVALID"),
        ({"accepted": ["missing"]}, "FACTOR_NOT_FOUND"),
    ],
)
def test_search_refusals_write_nothing(search_project, change, code) -> None:
    root, args = search_project
    before = _bytes(root)
    result = _search(root, args, **change)
    assert result.reason_codes == [code], result.as_dict()
    assert _bytes(root) == before


@pytest.mark.parametrize(
    "damage",
    [
        "dataset",
        "universe",
        "replay",
        "p-value",
        "trial",
        "artifact",
        "config-lock",
        "search-lock",
        "formula",
    ],
)
def test_search_tamper_and_contention_refusals_write_nothing(search_project, monkeypatch, damage) -> None:
    root, args = search_project
    expected = "FACTOR_SEARCH_INPUT_UNVERIFIED"
    handle = None
    if damage == "dataset":
        monkeypatch.setattr(
            api.factor,
            "dataset_from_manifest",
            lambda *_: (_ for _ in ()).throw(LibraryError("DATA_PAGE_MISSING")),
        )
        expected = "DATA_PAGE_MISSING"
    elif damage == "universe":
        path = root / args["universe_manifests"][0]
        body = json.loads(path.read_text())
        body["member_symbols"] = ["forged"]
        path.write_text(json.dumps(body))
        expected = "DATA_MANIFEST_INVALID"
    elif damage == "replay":
        monkeypatch.setattr(
            api.factor, "replay_universe_build", lambda *_: (_ for _ in ()).throw(ValueError("unverified"))
        )
    elif damage in {"p-value", "trial", "artifact"}:
        result = _search(root, args)
        if damage == "artifact":
            (root / result.artifacts[0]["path"]).write_text("[]")
            expected = "FACTOR_SEARCH_NONDETERMINISTIC"
        else:
            log = ledger.factor_trials(root)
            entries = log.entries()
            entries[0]["metrics"]["p_value" if damage == "p-value" else "scope"] = "bad"
            previous = None
            for entry in entries:
                entry["prev"] = previous
                entry["hash"] = canonical_hash({k: v for k, v in entry.items() if k != "hash"})
                previous = entry["hash"]
            log.path.write_text("".join(json.dumps(row) + "\n" for row in entries))
            expected = (
                "FACTOR_SEARCH_INPUT_UNVERIFIED" if damage == "p-value" else "FACTOR_SEARCH_NONDETERMINISTIC"
            )
    elif damage in {"config-lock", "search-lock"}:
        handle = os.open(root if damage == "search-lock" else root / "signalquarry.toml", os.O_RDONLY)
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected = "FACTOR_EVIDENCE_BUSY"
    elif damage == "formula":

        def bad(*a, **kw):
            report = search_expressions(*a, **kw)
            return replace(
                report,
                candidates=(
                    replace(report.candidates[0], identity="sha256:" + "f" * 64),
                    *report.candidates[1:],
                ),
            )

        monkeypatch.setattr(api.factor, "search_expressions", bad)
        expected = "FACTOR_SEARCH_NONDETERMINISTIC"
    before = _bytes(root)
    try:
        result = _search(root, args)
        assert result.reason_codes == [expected], result.as_dict()
        assert _bytes(root) == before
    finally:
        if handle is not None:
            os.close(handle)


def test_search_separate_families_and_different_searches_pay_again(search_project) -> None:
    root, args = search_project
    first = _search(root, args, budget=7)
    second = _search(root, args, seed=18, budget=7)
    assert second.status == first.status == "ok"
    assert second.data["accounting_before"]["family_trials_used"] == 7
    seal = ledger.factor_holdouts(root).entries()[0]
    ledger.append(
        root,
        "factor_holdouts",
        "other",
        {k: v for k, v in {**seal, "family": "other"}.items() if k not in {"schema", "seq", "prev", "hash"}},
    )
    third = _search(root, args, family="other", budget=7)
    assert third.status == "ok", third.as_dict()
    assert third.data["accounting_before"]["family_trials_used"] == 0
    assert third.data["accounting_before"]["project_trials_used"] == 14
    assert factor_trial_accounting(root, "momentum").family_trials_used == 14
    assert factor_trial_accounting(root, "other").family_trials_used == 7
    before = _bytes(root)
    assert _search(root, args, budget=40).reason_codes == ["FACTOR_SEARCH_BUDGET_EXHAUSTED"]
    assert _bytes(root) == before
    with pytest.raises(TypeError):
        _search(root, args, family_budget=10000)
    with pytest.raises(TypeError):
        _search(root, args, project_trials_used=0)


def test_search_cli_and_help(search_project, capsys) -> None:
    root, args = search_project
    argv = ["--json", "factor", "search", "--project", str(root)]
    for key in ("family", "dataset_id", "training_cutoff", "horizon", "seed", "budget"):
        argv.extend(["--" + key.replace("_", "-"), str(args[key])])
    for path in args["universe_manifests"]:
        argv.extend(["--universe-manifest", str(path)])
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["data"]["scope"] == "exploratory"
    assert main(["factor", "search", "--help"]) == 0
    help_text = capsys.readouterr().out
    assert "exploratory" in help_text and "--accepted" in help_text
    factor_command = next(item for item in command_catalog() if item["name"] == "factor")
    row = next(item for item in factor_command["subcommands"] if item["name"] == "search")
    assert not any(
        name in {"--family-budget", "--family-trials-used", "--project-trials-used", "--p-value"}
        for option in row["options"]
        for name in option["flags"]
    )


def test_search_evaluator_isolation_and_cutoff_mutation(search_project, monkeypatch) -> None:
    import signalquarry._internal.factors.search as core
    import signalquarry._internal.factors.training as training

    root, args = search_project
    seal_start = date.fromisoformat(ledger.factor_holdouts(root).entries()[0]["holdout_start"])
    original = core.evaluate_expression
    labels_original = training.derive_forward_return_labels
    observed = []

    def isolated(expression, context, **kwargs):
        observed.append(context.sessions)
        assert max(context.sessions) < seal_start
        assert max(context.sessions) < args["training_cutoff"]
        return original(expression, context, **kwargs)

    def isolated_labels(dataset, **kwargs):
        assert max(dataset.sessions) < seal_start
        assert max(dataset.sessions) < args["training_cutoff"]
        return labels_original(dataset, **kwargs)

    monkeypatch.setattr(core, "evaluate_expression", isolated)
    monkeypatch.setattr(training, "derive_forward_return_labels", isolated_labels)
    first = _search(root, args, accepted=["price-momentum"], budget=7)
    assert first.status == "ok", first.as_dict()
    before = _bytes(root)
    dataset = api.factor.dataset_from_manifest(None, None)
    stop = dataset.sessions.index(args["training_cutoff"])
    for series in dataset.series.values():
        for values in (*series.micro.values(), series.volume, series.present):
            values[stop:] = 0
    second = _search(root, args, accepted=["price-momentum"], budget=7)
    assert second.status == "ok" and second.data == first.data
    assert observed and _bytes(root) == before


@pytest.mark.parametrize(
    "damage",
    [
        "conformance",
        "short-search",
        "duplicate-build",
        "input-object",
        "artifact-collision",
        "artifact-missing",
    ],
)
def test_search_additional_refusals_are_read_only(search_project, monkeypatch, damage) -> None:
    root, args = search_project
    expected = "FACTOR_SEARCH_INPUT_UNVERIFIED"
    changes = {}
    if damage == "conformance":
        monkeypatch.setattr(api.factor, "import_policy", lambda *a, **kw: SimpleNamespace(ok=False))
        changes["accepted"] = ["price-momentum"]
        expected = "CONFORMANCE_FAILED"
    elif damage == "short-search":

        def short(*a, **kw):
            report = search_expressions(*a, **kw)
            return replace(report, trials_evaluated=1, candidates=report.candidates[:1])

        monkeypatch.setattr(api.factor, "search_expressions", short)
    elif damage == "duplicate-build":
        changes["universe_manifests"] = [args["universe_manifests"][0]] * 2
        expected = "FACTOR_EVALUATION_SESSIONS_INVALID"
    elif damage == "input-object":
        path = root / args["universe_manifests"][0]
        path.write_text("[]")
        expected = "DATA_MANIFEST_INVALID"
    elif damage in {"artifact-collision", "artifact-missing"}:
        result = _search(root, args)
        path = root / result.artifacts[0]["path"]
        if damage == "artifact-missing":
            path.unlink()
        else:
            body = json.loads(path.read_text())
            body["scope"] = "claimed"
            path.write_text(json.dumps(body))
        expected = "FACTOR_SEARCH_NONDETERMINISTIC"
    before = _bytes(root)
    result = _search(root, args, **changes)
    assert result.reason_codes == [expected], result.as_dict()
    assert _bytes(root) == before


def test_search_per_family_records_indexed_trials_and_replays_without_writes(search_project) -> None:
    root, args = search_project
    shutil.rmtree(root / "evidence")
    config = root / "signalquarry.toml"
    config.write_text(config.read_text() + '\n[evidence]\nlayout = "per_family"\n')
    assert api.factor_holdout_seal("momentum", "research-data", project=root).status == "ok"
    first = _search(root, args, budget=7)
    assert first.status == "ok", first.as_dict()
    assert len(ledger.factor_trials(root, "momentum").entries()) == 7
    assert len(ledger.project_index(root).entries()) == 8
    assert ledger.verify_index(root) == []
    before = _bytes(root)
    second = _search(root, args, budget=7)
    assert second.data == first.data and before == _bytes(root)

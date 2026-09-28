# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from signalquarry.api.factor import check_registered_factor, factor_ls
from signalquarry.cli.main import main


def _project(tmp_path: Path, *, extra_module: bool = False) -> tuple[Path, Path, str]:
    package = "factor_pkg_" + uuid.uuid4().hex[:8]
    root = tmp_path / "project"
    directory = root / "src" / package / "momentum"
    directory.mkdir(parents=True)
    (root / "src" / package / "__init__.py").write_text("", encoding="utf-8")
    (directory / "__init__.py").write_text("", encoding="utf-8")
    (directory / "factor.py").write_text(
        "from signalquarry.sdk import FactorCtx, Params, factor\n"
        "class P(Params):\n    period: int = 2\n"
        "@factor(params=P, lookback=lambda p: p.period)\n"
        "def score(ctx: FactorCtx, p: P) -> dict[str, float]:\n"
        "    return {symbol: float(ctx.panel('close')[-1, i]) for i, symbol in enumerate(ctx.universe)}\n",
        encoding="utf-8",
    )
    spec = directory / "factor.yaml"
    spec.write_text(
        "schema: signalquarry.factor/v1\n"
        "id: price-momentum\nfamily: momentum\nversion: '1'\n"
        "hypothesis:\n  statement: Past prices predict next returns.\n"
        "  falsification: Rank IC is nonpositive out of sample.\n"
        "params:\n  period: 2\n",
        encoding="utf-8",
    )
    modules = [f"{package}.momentum.factor"]
    if extra_module:
        duplicate = root / "src" / package / "copy"
        duplicate.mkdir()
        (duplicate / "__init__.py").write_text("", encoding="utf-8")
        (duplicate / "factor.py").write_text((directory / "factor.py").read_text(), encoding="utf-8")
        (duplicate / "factor.yaml").write_text(spec.read_text(), encoding="utf-8")
        modules.append(f"{package}.copy.factor")
    (root / "signalquarry.toml").write_text(
        '[project]\nname = "demo"\n[factors]\nmodules = ' + json.dumps(modules) + "\n",
        encoding="utf-8",
    )
    return root, spec, package


def test_registered_factor_identity_and_cli_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, spec, _ = _project(tmp_path)
    listed = factor_ls(project=root)
    assert listed.status == "ok" and listed.summary == "1 registered factors"
    item = listed.data["factors"][0]
    assert item["id"] == "price-momentum" and item["family"] == "momentum"
    assert item["configuration_hash"].startswith("sha256:")
    checked = check_registered_factor("price-momentum", project=root)
    assert checked.status == "ok"
    assert checked.data["configuration_hash"] == item["configuration_hash"]
    assert [result["name"] for result in checked.data["checks"]] == [
        "import_policy",
        "contract",
        "determinism",
        "lookahead",
    ]
    assert not (root / "evidence" / "trials.jsonl").exists()

    assert main(["--json", "factor", "ls", "--project", str(root)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["factors"][0]["configuration_hash"] == item["configuration_hash"]
    assert main(["--json", "check", "--factor-id", "price-momentum", "--project", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["configuration_hash"] == item["configuration_hash"]

    spec.write_text(spec.read_text().replace("period: 2", "period: 3"), encoding="utf-8")
    updated = factor_ls(project=root).data["factors"][0]
    assert updated["configuration_hash"] != item["configuration_hash"]


def test_factor_registration_fails_closed(tmp_path: Path) -> None:
    root, spec, _ = _project(tmp_path)
    assert check_registered_factor("missing", project=root).reason_codes == ["FACTOR_NOT_FOUND"]
    spec.unlink()
    assert factor_ls(project=root).reason_codes == ["FACTOR_SPEC_MISSING"]
    spec.write_text("schema: bad\nid: price-momentum\n", encoding="utf-8")
    assert factor_ls(project=root).reason_codes == ["FACTOR_SPEC_INVALID"]
    spec.write_text("schema: [unterminated\n", encoding="utf-8")
    assert factor_ls(project=root).reason_codes == ["FACTOR_SPEC_INVALID"]
    spec.write_text("- not a mapping\n", encoding="utf-8")
    assert factor_ls(project=root).reason_codes == ["FACTOR_SPEC_NOT_A_MAPPING"]


def test_invalid_registered_params_are_rejected(tmp_path: Path) -> None:
    root, spec, _ = _project(tmp_path)
    spec.write_text(spec.read_text().replace("period: 2", "period: no-number"), encoding="utf-8")
    assert factor_ls(project=root).reason_codes == ["FACTOR_PARAMS_INVALID"]


def test_duplicate_factor_ids_and_config_shape(tmp_path: Path) -> None:
    root, _, _ = _project(tmp_path, extra_module=True)
    assert factor_ls(project=root).reason_codes == ["FACTOR_ID_DUPLICATE"]
    (root / "signalquarry.toml").write_text('[factors]\nmodules = "not a list"\n', encoding="utf-8")
    assert factor_ls(project=root).reason_codes == ["PROJECT_CONFIG_INVALID"]

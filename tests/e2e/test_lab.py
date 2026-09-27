# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from signalquarry.cli.main import main


def _sqy(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(["--json", *argv])
    return code, json.loads(capsys.readouterr().out)


def _tool(lab: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *argv], cwd=lab, capture_output=True, text=True, check=False)


def _git(lab: Path, *argv: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *argv],
        cwd=lab,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def lab(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    root = tmp_path / "lab"
    code, payload = _sqy(capsys, "init", str(root), "--lab", "--package", "lab" + uuid.uuid4().hex[:8])
    assert code == 0, payload
    assert {"families/drill/publication.yaml", ".github/workflows/lab-ci.yml", "docs/design/LAB.md"} <= set(
        payload["data"]["files"]
    )
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "lab")
    return root


def test_lab_check_new_family_and_publication(lab: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _sqy(capsys, "check", "--project", str(lab))[0] == 0
    created = _tool(lab, "tools/new_family.py", "momentum")
    assert created.returncode == 0, created.stderr
    code, payload = _sqy(capsys, "check", "--project", str(lab))
    assert code == 0 and {s["strategy"] for s in payload["data"]["strategies"]} == {"drill", "momentum-a"}
    # The lab's own test suite (what lab-ci.yml runs) collects a conformance test per family.
    suite = _tool(lab, "-m", "pytest", "-q", "-p", "no:cacheprovider")
    assert suite.returncode == 0, suite.stdout + suite.stderr
    assert "2 passed" in suite.stdout
    code, payload = _sqy(capsys, "evidence", "export", "--family", "drill", "--project", str(lab))
    assert code == 0, payload
    bundle = Path(payload["data"]["path"])
    profile = json.loads((bundle / "strategies/drill/profile.json").read_text())
    assert profile["commercial"] is True and profile["disclosure_tier"] == "category"
    scan = _tool(lab, "tools/bundle_leak_scan.py", str(bundle), "--family", "drill")
    assert scan.returncode == 0, scan.stdout
    leaky = bundle / "strategies/drill/profile.json"
    leaky.write_text(leaky.read_text().replace("Transfer drill (never traded)", "Uses ABOVE_AVERAGE on SPY"))
    assert _tool(lab, "tools/bundle_leak_scan.py", str(bundle), "--family", "drill").returncode == 1


def test_lab_guards(lab: Path) -> None:
    assert _tool(lab, "tools/new_family.py", "momentum").returncode == 0
    _git(lab, "add", "-A")
    _git(lab, "commit", "-qm", "momentum")
    assert _tool(lab, "tools/check_family_imports.py").returncode == 0
    assert _tool(lab, "tools/check_commit_scope.py", "HEAD~1..HEAD").returncode == 0

    momentum = next(lab.glob("families/momentum/src/*/momentum_a/strategy.py"))
    drill_package = next(lab.glob("families/drill/src/*")).name
    momentum.write_text(f"import {drill_package}\n" + momentum.read_text())
    readme = lab / "families/drill/README.md"
    readme.write_text(readme.read_text() + "\nedited alongside another family\n")
    _git(lab, "add", "-A")
    _git(lab, "commit", "-qm", "two families at once")
    assert _tool(lab, "tools/check_family_imports.py").returncode == 1
    scope = _tool(lab, "tools/check_commit_scope.py", "HEAD~1..HEAD")
    assert scope.returncode == 1 and "COMMIT_TOUCHES_SEVERAL_FAMILIES" in scope.stdout

    config = lab / "signalquarry.toml"
    config.write_text(
        config.read_text().replace("forbidden_references = []", 'forbidden_references = ["my-showcase"]')
    )
    assert _tool(lab, "tools/check_references.py").returncode == 0
    (lab / "families/drill/notes.md").write_text("see my-showcase for the page\n")
    _git(lab, "add", "-A")
    assert _tool(lab, "tools/check_references.py").returncode == 1


@pytest.mark.skipif(shutil.which("git-filter-repo") is None, reason="git-filter-repo not installed")
def test_transfer_drill_keeps_the_tree_hash(lab: Path, tmp_path: Path) -> None:
    completed = subprocess.run(
        ["sh", "tools/extract_family.sh", "drill", str(tmp_path / "drill")],
        cwd=lab,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (tmp_path / "drill" / "publication.yaml").is_file()


def test_lab_rejects_demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _sqy(capsys, "init", str(tmp_path / "x"), "--lab", "--demo")
    assert code == 64

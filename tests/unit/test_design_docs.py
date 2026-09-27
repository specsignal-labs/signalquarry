# SPDX-License-Identifier: Apache-2.0
"""The architecture page follows the code: generated graphs, and design docs required for changes."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import check_design_docs as check  # noqa: E402
import gen_architecture as arch  # noqa: E402

PYPROJECT = '[tool.importlinter]\nroot_package = "x"\n'


def test_architecture_changes_are_classified() -> None:
    assert check.architecture_changes("M\tsrc/signalquarry/api/data.py", PYPROJECT, PYPROJECT) == []
    assert check.architecture_changes("A\tsrc/signalquarry/templates/x/y.py", PYPROJECT, PYPROJECT) == []
    added = check.architecture_changes("A\tsrc/signalquarry/api/new.py", PYPROJECT, PYPROJECT)
    assert added == ["module added: src/signalquarry/api/new.py"]
    renamed = check.architecture_changes(
        "R100\tsrc/signalquarry/a.py\tsrc/signalquarry/b.py", PYPROJECT, PYPROJECT
    )
    assert renamed == ["module renamed: src/signalquarry/a.py"]
    contracts = check.architecture_changes(
        "M\tpyproject.toml", PYPROJECT, PYPROJECT + "include_external_packages = true\n"
    )
    assert contracts == ["import-linter contracts changed in pyproject.toml"]
    assert check.documented("M\tdocs/design/ARCHITECTURE.md") and check.documented("A\tdocs/adr/0010-x.md")
    assert not check.documented("M\tdocs/adr/README.md")


def _git(repo: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    base = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, env={**base, **(env or {})}
    )


def _run(repo: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "tools/check_design_docs.py", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


def test_hook_and_range_modes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    shutil.copy(ROOT / "tools" / "check_design_docs.py", repo / "tools")
    (repo / "src" / "signalquarry").mkdir(parents=True)
    (repo / "src" / "signalquarry" / "core.py").write_text("")
    (repo / "docs" / "design").mkdir(parents=True)
    (repo / "docs" / "design" / "ARCHITECTURE.md").write_text("# A\n")
    (repo / "pyproject.toml").write_text(PYPROJECT)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    (repo / "src" / "signalquarry" / "extra.py").write_text("")
    _git(repo, "add", "-A")
    blocked = _run(repo, "--staged")
    assert blocked.returncode == 1 and "module added: src/signalquarry/extra.py" in blocked.stdout
    assert _run(repo, "--staged", env={"SIGNALQUARRY_ARCH_OK": "1"}).returncode == 0
    (repo / "docs" / "design" / "ARCHITECTURE.md").write_text("# A\n\nextra component\n")
    _git(repo, "add", "-A")
    assert _run(repo, "--staged").returncode == 0
    _git(repo, "commit", "-qm", "extra, documented")

    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "src" / "signalquarry" / "more.py").write_text("")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add more")
    assert _run(repo, "--range", "main..feature").returncode == 1
    _git(repo, "commit", "--allow-empty", "-qm", "note\n\nArchitecture: unchanged")
    acknowledged = _run(repo, "--range", "main..feature")
    assert acknowledged.returncode == 0 and "ACKNOWLEDGED" in acknowledged.stdout


def test_generated_block_covers_every_component_and_is_idempotent() -> None:
    text = (ROOT / "docs" / "design" / "ARCHITECTURE.md").read_text()
    block = arch.block()
    for component in (
        "cli",
        "api",
        "paper",
        "engine",
        "options",
        "sdk",
        "plugins",
        "contracts",
        "canonical",
        "testing",
    ):
        assert f"`{component}`" in block
    assert arch.apply(arch.apply(text)) == arch.apply(text)
    assert arch.START in text and arch.END in text

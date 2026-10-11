# SPDX-License-Identifier: Apache-2.0
"""The research-method instructions reach new projects and upgraded ones."""

from __future__ import annotations

import re
from pathlib import Path

from signalquarry.api import init, upgrade_agents_md

BLOCK = re.compile(
    r"<!-- signalquarry:begin project/research-method v1 -->\n.*?<!-- signalquarry:end project/research-method -->\n\n",
    re.DOTALL,
)


def test_new_projects_tell_agents_how_to_compare(tmp_path: Path) -> None:
    project = tmp_path / "method-lab"
    assert init(project, demo=True, package="method_lab").status == "ok"
    text = (project / "AGENTS.md").read_text()
    assert "## Research method: compared with what?" in text
    for phrase in (
        "Declare `benchmark:`",
        "sqy backtest --strategy <id> --param name=value --label <name>",
        "sqy runs compare <reference> <run>",
        "RUNS_NOT_COMPARABLE",
        "sqy study init --strategy <id> --id <study>",
        "`supported`, `not_supported` or\n  `insufficient`",
        "an edited file is a different study",
        "They never raise the claim level",
        "sqy diagnose --strategy <id>",
    ):
        assert phrase in text, phrase
    assert "<!-- signalquarry:begin project/files v3 -->" in text
    assert "`studies/<id>/study.yaml`" in text and "`.signalquarry/`" in text


def test_upgrading_an_older_project_adds_the_block_and_keeps_local_notes(tmp_path: Path) -> None:
    project = tmp_path / "method-old"
    assert init(project, demo=True, package="method_old").status == "ok"
    path = project / "AGENTS.md"
    current = path.read_text()
    older, removed = BLOCK.subn("", current)
    assert removed == 1
    older = older.replace("project/files v3 -->", "project/files v2 -->").replace(
        "- `studies/<id>/study.yaml` — a declared comparison (you write it before running it)\n", ""
    )
    path.write_text(older + "\n## My notes\n\nKeep this.\n")

    envelope = upgrade_agents_md(project=project)
    assert envelope.status == "ok", envelope.summary
    assert "project/research-method" in envelope.data["added"]
    assert "project/files" in envelope.data["updated"]
    upgraded = path.read_text()
    assert "## Research method: compared with what?" in upgraded
    assert "`studies/<id>/study.yaml`" in upgraded and "## My notes\n\nKeep this.\n" in upgraded
    assert upgrade_agents_md(project=project).data["updated"] == []

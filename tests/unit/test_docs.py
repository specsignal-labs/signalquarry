# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import runpy
from pathlib import Path

from signalquarry._internal.contracts.reason_codes import REASON_CODES
from signalquarry.api.docs import GUIDES
from signalquarry.cli.main import command_catalog, main

ROOT = Path(__file__).parents[2]


def test_generated_docs_are_fresh() -> None:
    module = runpy.run_path(str(ROOT / "tools" / "gen_docs.py"))
    stale = [
        path.relative_to(ROOT).as_posix()
        for path, text in module["generated"]().items()
        if not path.is_file() or path.read_text(encoding="utf-8") != text
    ]
    assert stale == [], "run `uv run python tools/gen_docs.py`"


def test_llms_full_covers_every_guide_command_and_code(capsys) -> None:
    import json

    assert main(["--json", "--detail", "full", "docs", "--llms", "--full"]) == 0
    text = json.loads(capsys.readouterr().out)["data"]["text"]
    assert all(f"({name}.md)" in text for name, _ in GUIDES)
    assert all(f"`sqy {c['name']}`" in text for c in command_catalog())
    assert all(f"`{code}`" in text for code in REASON_CODES)


def test_every_option_is_documented_without_markup() -> None:
    for command in command_catalog():
        for item in [command, *command["subcommands"]]:
            assert item["help"], item["name"]
            for option in item["options"]:
                assert option["help"] and "<" not in option["help"], (item["name"], option)


def test_public_api_snapshot_is_current() -> None:
    module = runpy.run_path(str(ROOT / "tools" / "api_snapshot.py"))
    recorded = (ROOT / "tools" / "public_api.txt").read_text(encoding="utf-8")
    assert module["snapshot"]() == recorded, "run `python tools/api_snapshot.py` and label the PR api-change"

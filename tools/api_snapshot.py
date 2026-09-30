# SPDX-License-Identifier: Apache-2.0
"""Snapshot of the public Python surface (sdk, api, testing) for review in every PR.

``python tools/api_snapshot.py`` rewrites ``tools/public_api.txt``; ``--check`` fails with
a diff when the surface changed without the snapshot being updated. A changed snapshot
needs the ``api-change`` label and, for removals or signature changes in ``sdk``, a
deprecation (see CONTRIBUTING.md).
"""

from __future__ import annotations

import dataclasses
import difflib
import importlib
import inspect
import re
import sys
from pathlib import Path

SNAPSHOT = Path(__file__).with_name("public_api.txt")
MODULES = (
    "signalquarry",
    "signalquarry.sdk",
    "signalquarry.sdk.options",
    "signalquarry.api",
    "signalquarry.plugins",
    "signalquarry.testing",
)
ALL_PUBLIC = ("signalquarry.sdk.ta", "signalquarry.sdk.xs")  # modules without __all__: public functions


def _clean(text: str) -> str:
    return re.sub(r" at 0x[0-9a-f]+", "", text)


def _signature(obj: object) -> str:
    try:
        return _clean(str(inspect.signature(obj)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "(...)"


def _describe(qualified: str, obj: object) -> list[str]:
    module = getattr(obj, "__module__", "") or ""
    if inspect.ismodule(obj):
        return [f"{qualified}: module"]
    if not inspect.isclass(obj) and not callable(obj):
        return [f"{qualified}: {type(obj).__name__}"]
    if not module.startswith("signalquarry"):
        return [f"{qualified}: re-export of {module}.{getattr(obj, '__qualname__', type(obj).__name__)}"]
    if inspect.isclass(obj):
        bases = ", ".join(b.__name__ for b in obj.__bases__)
        lines = [f"{qualified}: class({bases})"]
        if dataclasses.is_dataclass(obj):
            lines += [
                f"{qualified}.{f.name}: field {f.type}"
                for f in dataclasses.fields(obj)
                if not f.name.startswith("_")
            ]
        fields = getattr(obj, "model_fields", None)
        if isinstance(fields, dict):
            lines += [
                f"{qualified}.{name}: field {info.annotation!r}" for name, info in sorted(fields.items())
            ]
        for name, member in sorted(vars(obj).items()):
            if name.startswith("_") or name.startswith("model_"):
                continue
            target = member.__func__ if isinstance(member, (classmethod, staticmethod)) else member
            if callable(target):
                lines.append(f"{qualified}.{name}{_signature(target)}")
            elif isinstance(member, property):
                lines.append(f"{qualified}.{name}: property")
        return [_clean(line) for line in lines]
    return [f"{qualified}{_signature(obj)}"]


def snapshot() -> str:
    lines: list[str] = []
    for name in (*MODULES, *ALL_PUBLIC):
        module = importlib.import_module(name)
        names = getattr(module, "__all__", None) if name not in ALL_PUBLIC else None
        if names is None:
            names = [
                n
                for n, v in vars(module).items()
                if not n.startswith("_") and inspect.isfunction(v) and v.__module__ == name
            ]
        for attribute in sorted(names):
            lines += _describe(f"{name}.{attribute}", getattr(module, attribute))
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    current = snapshot()
    if "--check" not in argv:
        SNAPSHOT.write_text(current, encoding="utf-8")
        print(f"wrote {SNAPSHOT}")
        return 0
    recorded = SNAPSHOT.read_text(encoding="utf-8") if SNAPSHOT.is_file() else ""
    if recorded == current:
        print("public API unchanged")
        return 0
    sys.stdout.writelines(
        difflib.unified_diff(recorded.splitlines(True), current.splitlines(True), "recorded", "current")
    )
    print("\npublic API changed: run `python tools/api_snapshot.py`, commit, and label the PR api-change")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

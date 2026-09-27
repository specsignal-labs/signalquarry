# SPDX-License-Identifier: Apache-2.0
"""Generate the code-derived part of ``docs/design/ARCHITECTURE.md``.

The block between ``<!-- generated:architecture:start -->`` and
``<!-- generated:architecture:end -->`` is rebuilt from the code itself:

- the import graph between components, parsed from the source with ``ast`` and laid out
  by the layer contract in ``pyproject.toml`` (import-linter);
- the component inventory (every module of every package);
- the CLI command tree (from the command registry);
- the plugin groups (from ``signalquarry.plugins.GROUPS``).

``tools/gen_docs.py`` calls :func:`apply`, so ``gen_docs.py --check`` (CI, tests and the
pre-commit hook) fails whenever the code's architecture no longer matches the page.
"""

from __future__ import annotations

import ast
import re
import tomllib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "signalquarry"
START, END = "<!-- generated:architecture:start -->", "<!-- generated:architecture:end -->"
SKIP_DIRS = {"templates", "schemas", "guides", "__pycache__"}


def component_of(module: str) -> str | None:
    """``signalquarry._internal.paper.brokers.fake`` -> ``paper``; ``signalquarry.api.data`` -> ``api``."""
    parts = module.split(".")
    if parts[0] != "signalquarry" or len(parts) < 2:
        return None
    if parts[1] == "_internal":
        return parts[2] if len(parts) > 2 else None
    return parts[1] if parts[1] not in ("__main__",) else "cli"


def modules() -> dict[str, Path]:
    """Every Python module of the package (templates, schemas and guides excluded)."""
    found: dict[str, Path] = {}
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC.parent)
        if SKIP_DIRS & set(relative.parts):
            continue
        name = ".".join(relative.with_suffix("").parts)
        found[name.removesuffix(".__init__")] = path
    return found


def _imports(name: str, path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - node.level + 1]
                prefix = ".".join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            targets.add(prefix)
            targets.update(f"{prefix}.{alias.name}" for alias in node.names)
    return {t for t in targets if t.startswith("signalquarry")}


def edges() -> dict[str, set[str]]:
    """Component -> components it imports (runtime and type-checking imports alike)."""
    known = modules()
    graph: dict[str, set[str]] = defaultdict(set)
    for name, path in known.items():
        source = component_of(name)
        if source is None:
            continue
        graph.setdefault(source, set())
        for target in _imports(name, path):
            while target not in known and "." in target:
                target = target.rsplit(".", 1)[0]
            other = component_of(target)
            if other is not None and other != source:
                graph[source].add(other)
    return graph


def layers() -> list[list[str]]:
    """The layer contract, top (may import everything below) to bottom."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    contract = next(c for c in config["tool"]["importlinter"]["contracts"] if c["type"] == "layers")
    out = []
    for layer in contract["layers"]:
        names = [part.strip().strip("()") for part in layer.split("|")]
        out.append([component_of(name) or name for name in names])
    return out


def _node(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def import_graph() -> str:
    """The layer stack: each layer may import only from layers below it."""
    graph = edges()
    placed = {name for layer in layers() for name in layer}
    lines = ["```mermaid", "flowchart TB"]
    previous = None
    for number, layer in enumerate(layers()):
        present = [name for name in layer if name in graph]
        if not present:
            continue
        node = f"L{number}"
        lines.append(f'  {node}["<b>{" · ".join(present)}</b>"]')
        if previous is not None:
            lines.append(f"  {previous} --> {node}")
        previous = node
    for name in sorted(set(graph) - placed):
        targets = sorted(graph[name])
        lines.append(f'  {_node(name)}(["{name}<br/>outside the layers"])')
        for number, layer in enumerate(layers()):
            if set(layer) & set(targets):
                lines.append(f"  {_node(name)} -.-> L{number}")
    lines.append("```")
    return "\n".join(lines)


def import_table() -> str:
    graph = edges()
    order = [name for layer in layers() for name in layer]
    rows = sorted(graph, key=lambda c: (order.index(c) if c in order else len(order), c))
    lines = ["| Component | Imports (direct) |", "|---|---|"]
    for name in rows:
        targets = sorted(graph[name], key=lambda c: (order.index(c) if c in order else len(order), c))
        lines.append(f"| `{name}` | {', '.join(f'`{t}`' for t in targets) or '—'} |")
    return "\n".join(lines)


def inventory() -> str:
    by_component: dict[str, list[str]] = defaultdict(list)
    for name in modules():
        component = component_of(name)
        if component is None:
            continue
        leaf = name.split(f".{component}", 1)[-1].lstrip(".") or "(package)"
        by_component[component].append(leaf)
    order = [name for layer in layers() for name in layer]
    rows = sorted(by_component, key=lambda c: (order.index(c) if c in order else len(order), c))
    lines = ["| Component | Modules |", "|---|---|"]
    lines += [f"| `{c}` | {', '.join(f'`{m}`' for m in sorted(by_component[c]))} |" for c in rows]
    return "\n".join(lines)


def command_tree() -> str:
    from signalquarry.cli.main import command_catalog

    lines = ["```mermaid", "flowchart LR", '  sqy(["sqy"])']
    for command in command_catalog():
        node = _node(f"c_{command['name']}")
        lines.append(f'  sqy --> {node}["{command["name"]}"]')
        for sub in command["subcommands"]:
            lines.append(f'  {node} --> {_node(f"c_{command['name']}_{sub['name']}")}["{sub["name"]}"]')
    lines.append("```")
    return "\n".join(lines)


def plugin_groups() -> str:
    from signalquarry.plugins import GROUPS

    return ", ".join(f"`signalquarry.{group}`" for group in GROUPS)


def block() -> str:
    return "\n".join(
        [
            START,
            "",
            "*Generated by `tools/gen_architecture.py` from the source; do not edit by hand.*",
            "",
            "### Layers",
            "",
            "Layers are the import-linter contract (`pyproject.toml`), top to bottom: a component",
            "may import only from layers below it. Components on one layer cannot import each other.",
            "",
            import_graph(),
            "",
            "### Direct imports",
            "",
            "Parsed from the source, so this is what the code does, not what it should do.",
            "",
            import_table(),
            "",
            "### Component inventory",
            "",
            inventory(),
            "",
            "### Command tree",
            "",
            command_tree(),
            "",
            f"**Plugin entry-point groups:** {plugin_groups()}.",
            "",
            END,
        ]
    )


def apply(text: str) -> str:
    """Replace the generated block in ``text`` (append it under a heading if absent)."""
    if START in text and END in text:
        head, rest = text.split(START, 1)
        tail = rest.split(END, 1)[1]
        return head + block() + tail
    return text.rstrip("\n") + "\n\n## Generated from the code\n\n" + block() + "\n"

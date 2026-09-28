# SPDX-License-Identifier: Apache-2.0
"""Find a project (``signalquarry.toml``), load its settings and its strategies.

Strategies are found in two ways, both explicit:

* ``[strategies] modules = ["pkg.strategy_module", ...]`` in ``signalquarry.toml``
  (``src_dirs`` are put on ``sys.path`` first), and
* installed entry points in the ``signalquarry.strategies`` group.

Each strategy module defines exactly one ``@strategy`` function and has a
``strategy.yaml`` next to it.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import tomllib
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from signalquarry import __version__
from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.contracts.spec import StrategySpecV1, load_spec
from signalquarry.sdk.strategy import ATTRIBUTE, Params, StrategyDef

CONFIG_NAME = "signalquarry.toml"
ENTRY_POINT_GROUP = "signalquarry.strategies"


class ProjectError(RuntimeError):
    """A project or strategy could not be loaded. ``code`` is a reason code."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    name: str
    provider: Literal["synthetic", "alpaca"]
    src_dirs: tuple[Path, ...]
    modules: tuple[str, ...]
    factor_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadedStrategy:
    spec: StrategySpecV1
    definition: StrategyDef
    params: Params
    module: ModuleType
    package_dir: Path

    @property
    def code_tree_hash(self) -> str:
        return code_tree_hash(self.package_dir)

    @property
    def configuration_hash(self) -> str:
        major_minor = ".".join(__version__.split(".")[:2])
        return canonical_hash(
            {
                "spec": self.spec.outcome_document(),
                "params": self.params.model_dump(mode="json"),
                "code_tree_hash": self.code_tree_hash,
                "framework": major_minor,
            }
        )


def find_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / CONFIG_NAME).is_file():
            return candidate
    raise ProjectError("PROJECT_NOT_FOUND", str(current))


def load_config(root: Path) -> ProjectConfig:
    try:
        document: dict[str, Any] = tomllib.loads((root / CONFIG_NAME).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ProjectError("PROJECT_CONFIG_INVALID", str(exc)) from exc
    project = document.get("project", {})
    data = document.get("data", {})
    strategies = document.get("strategies", {})
    factors = document.get("factors", {})
    if (
        not isinstance(factors, dict)
        or not isinstance(factors.get("modules", []), list)
        or any(not isinstance(name, str) or not name for name in factors.get("modules", []))
    ):
        raise ProjectError("PROJECT_CONFIG_INVALID", "factors.modules must be a list of module names")
    provider = data.get("provider", "alpaca")
    if provider not in ("synthetic", "alpaca"):
        raise ProjectError("PROJECT_CONFIG_INVALID", f"data.provider={provider}")
    return ProjectConfig(
        root=root,
        name=str(project.get("name", root.name)),
        provider=provider,
        src_dirs=tuple(
            path
            for item in strategies.get("src_dirs", ["src"])
            for path in (sorted(root.glob(item)) if any(c in item for c in "*?[") else [root / item])
        ),
        modules=tuple(strategies.get("modules", [])),
        factor_modules=tuple(factors.get("modules", [])),
    )


def code_tree_hash(directory: Path) -> str:
    """SHA-256 over the strategy package's source files (path + content), ignoring caches."""
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if (
            path.is_file()
            and "__pycache__" not in relative.parts
            and path.suffix in {".py", ".yaml", ".yml", ".json"}
        ):
            digest.update(relative.as_posix().encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + digest.hexdigest()


def _definition(module: ModuleType) -> StrategyDef:
    found = [
        value for value in vars(module).values() if isinstance(getattr(value, ATTRIBUTE, None), StrategyDef)
    ]
    unique = {id(getattr(item, ATTRIBUTE)): getattr(item, ATTRIBUTE) for item in found}
    if len(unique) != 1:
        raise ProjectError(
            "STRATEGY_MODULE_INVALID", f"{module.__name__} defines {len(unique)} @strategy functions"
        )
    return next(iter(unique.values()))


def load_module_strategy(module_name: str) -> LoadedStrategy:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ProjectError("STRATEGY_IMPORT_FAILED", f"{module_name}: {exc}") from exc
    definition = _definition(module)
    module_file = Path(module.__file__ or "")
    package_dir = module_file.parent
    spec_path = package_dir / "strategy.yaml"
    if not spec_path.is_file():
        raise ProjectError("STRATEGY_SPEC_MISSING", str(spec_path))
    try:
        spec = load_spec(spec_path)
    except ValueError as exc:
        raise ProjectError("STRATEGY_SPEC_INVALID", str(exc)[:500]) from exc
    if (spec.kind == "options_single_leg") != (definition.kind == "options"):
        raise ProjectError(
            "STRATEGY_KIND_MISMATCH",
            f"{spec.id}: strategy.yaml kind {spec.kind} needs "
            + ("@options_strategy" if spec.kind == "options_single_leg" else "@strategy"),
        )
    try:
        params = definition.params.model_validate(spec.params)
    except ValueError as exc:
        raise ProjectError("STRATEGY_PARAMS_INVALID", str(exc)[:500]) from exc
    return LoadedStrategy(spec, definition, params, module, package_dir)


def load_strategies(config: ProjectConfig) -> dict[str, LoadedStrategy]:
    for directory in reversed(config.src_dirs):
        text = str(directory)
        if directory.is_dir() and text not in sys.path:
            sys.path.insert(0, text)
    names = list(config.modules) + [item.value for item in entry_points(group=ENTRY_POINT_GROUP)]
    loaded: dict[str, LoadedStrategy] = {}
    for name in dict.fromkeys(names):
        strategy = load_module_strategy(name)
        if strategy.spec.id in loaded:
            raise ProjectError("STRATEGY_ID_DUPLICATE", strategy.spec.id)
        loaded[strategy.spec.id] = strategy
    return loaded

# SPDX-License-Identifier: Apache-2.0
"""Extension points (provisional): installed packages add gates and report sections.

A plugin is an object (or a class with a no-argument constructor) exposed through an
entry point::

    [project.entry-points."signalquarry.gates"]
    max-turnover = "my_pkg.gates:MaxTurnover"

Every plugin declares ``name`` and ``api_version`` (currently :data:`API_VERSION`).

- **Gates are add-only.** A plugin gate can only lower a claim: when it fails, or
  raises, the claim level is capped at ``in_sample``. It never unlocks anything and
  cannot replace a built-in gate.
- **Report sections** append Markdown to ``report.md`` under their own heading.
- **Brokers** (``broker: plugin:<name>`` in a paper config) build a paper broker; it must
  declare ``paper_only = True`` and implement the paper broker interface plus ``calendar()``.

Deliberately not pluggable: the context builder, clocks, ledgers and seals, canonical
hashing, the claim ladder and the minimum gates, the paper kernel's safety checks and
any live-trading hook.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Protocol, runtime_checkable

API_VERSION = 1
GROUPS = ("gates", "report_sections", "brokers")
_NAME = re.compile(r"[a-z][a-z0-9-]{0,39}")


@dataclass(frozen=True)
class GateContext:
    """What a gate may look at: evaluation results, never the data or the strategy code."""

    strategy_id: str
    family: str
    grade: str
    spec: Mapping[str, Any]
    gates: Mapping[str, Mapping[str, Any]]
    folds: tuple[Mapping[str, Any], ...]
    oos: Mapping[str, Any]
    oos_returns: tuple[float, ...]


@dataclass(frozen=True)
class GateOutcome:
    ok: bool
    detail: Mapping[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class ReportContext:
    spec: Mapping[str, Any]
    result: Mapping[str, Any] | None
    evaluation: Mapping[str, Any] | None


@runtime_checkable
class GatePlugin(Protocol):
    name: str
    api_version: int

    def check(self, context: GateContext) -> GateOutcome: ...


@runtime_checkable
class ReportSectionPlugin(Protocol):
    name: str
    api_version: int
    title: str

    def render(self, context: ReportContext) -> str: ...


@runtime_checkable
class BrokerPlugin(Protocol):
    """Builds a paper broker for one deployment (``broker: plugin:<name>`` in its paper config).

    The broker must set ``paper_only = True`` and implement the paper broker protocol plus
    ``calendar(start, end) -> list[date]``; the kernel refuses anything else. Credentials
    come from the deployment's ``[paper.<alias>]`` profile (None when absent).
    """

    name: str
    api_version: int

    def create(self, alias: str, key_id: str | None, secret_key: str | None) -> Any: ...


@dataclass(frozen=True)
class Loaded:
    group: str
    name: str
    source: str
    plugin: Any


@dataclass(frozen=True)
class Discovery:
    plugins: tuple[Loaded, ...] = ()
    errors: tuple[str, ...] = ()


_PROTOCOLS: dict[str, type] = {
    "gates": GatePlugin,
    "report_sections": ReportSectionPlugin,
    "brokers": BrokerPlugin,
}
EntryPoints = Callable[[str], Iterable[metadata.EntryPoint]]


def _installed(group: str) -> Iterable[metadata.EntryPoint]:
    return metadata.entry_points(group=group)


def discover(group: str, *, entry_points: EntryPoints = _installed) -> Discovery:
    """Load the ``signalquarry.<group>`` entry points; problems become errors, not exceptions."""
    if group not in _PROTOCOLS:
        raise ValueError(f"unknown plugin group {group!r}")
    plugins: list[Loaded] = []
    errors: list[str] = []
    seen: set[str] = set()
    for point in sorted(entry_points(f"signalquarry.{group}"), key=lambda p: (p.name, p.value)):
        source = f"{point.name} = {point.value}"
        try:
            loaded = point.load()
            plugin = loaded() if isinstance(loaded, type) else loaded
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not break the command
            errors.append(f"PLUGIN_LOAD_FAILED {group}:{source}: {type(exc).__name__}: {exc}")
            continue
        name = getattr(plugin, "name", None)
        version = getattr(plugin, "api_version", None)
        if version != API_VERSION:
            errors.append(f"PLUGIN_API_VERSION {group}:{source}: api_version {version!r} != {API_VERSION}")
            continue
        if (
            not isinstance(name, str)
            or not _NAME.fullmatch(name)
            or not isinstance(plugin, _PROTOCOLS[group])
        ):
            errors.append(
                f"PLUGIN_INVALID {group}:{source}: needs a kebab-case name and the {group} protocol"
            )
            continue
        if name in seen:
            errors.append(f"PLUGIN_DUPLICATE {group}:{name}")
            continue
        seen.add(name)
        plugins.append(Loaded(group, name, source, plugin))
    return Discovery(tuple(plugins), tuple(errors))


__all__ = [
    "API_VERSION",
    "BrokerPlugin",
    "GROUPS",
    "Discovery",
    "GateContext",
    "GateOutcome",
    "GatePlugin",
    "Loaded",
    "ReportContext",
    "ReportSectionPlugin",
    "discover",
]

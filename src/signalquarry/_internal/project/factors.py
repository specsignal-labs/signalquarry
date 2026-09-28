# SPDX-License-Identifier: Apache-2.0
"""Load explicitly registered project factors and derive their configuration identity."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from signalquarry import __version__
from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.contracts.factor_spec import FactorSpecV1, load_factor_spec
from signalquarry._internal.project.project import ProjectConfig, ProjectError, code_tree_hash
from signalquarry.sdk.factors import ATTRIBUTE, FactorDef
from signalquarry.sdk.strategy import Params


@dataclass(frozen=True)
class LoadedFactor:
    spec: FactorSpecV1
    definition: FactorDef
    params: Params
    module: ModuleType
    package_dir: Path

    @property
    def code_tree_hash(self) -> str:
        return code_tree_hash(self.package_dir)

    @property
    def configuration_hash(self) -> str:
        """Factor formula/config identity; evaluation choices get a separate trial key."""
        major_minor = ".".join(__version__.split(".")[:2])
        return canonical_hash(
            {
                "schema": "signalquarry.factor-configuration/v1",
                "spec": self.spec.model_dump(mode="json", by_alias=True),
                "params": self.params.model_dump(mode="json"),
                "definition": {"module": self.definition.module, "name": self.definition.name},
                "code_tree_hash": self.code_tree_hash,
                "framework": major_minor,
            }
        )


def load_module_factor(module_name: str, root: Path) -> LoadedFactor:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ProjectError("FACTOR_IMPORT_FAILED", f"{module_name}: {str(exc)[:400]}") from exc
    module_file = Path(getattr(module, "__file__", None) or "").resolve()
    if not module_file.is_file() or not module_file.is_relative_to(root):
        raise ProjectError("FACTOR_MODULE_INVALID", f"{module_name} is outside the project")
    definitions = {
        id(definition): definition
        for value in vars(module).values()
        if isinstance((definition := getattr(value, ATTRIBUTE, None)), FactorDef)
        and definition.module == module_name
    }
    if len(definitions) != 1:
        raise ProjectError(
            "FACTOR_MODULE_INVALID", f"{module_name} defines {len(definitions)} @factor functions"
        )
    package_dir = module_file.parent
    spec_path = package_dir / "factor.yaml"
    if not spec_path.is_file() or not spec_path.resolve().is_relative_to(root):
        raise ProjectError("FACTOR_SPEC_MISSING", str(spec_path))
    try:
        spec = load_factor_spec(spec_path)
    except ValueError as exc:
        code = (
            "FACTOR_SPEC_NOT_A_MAPPING" if str(exc) == "FACTOR_SPEC_NOT_A_MAPPING" else "FACTOR_SPEC_INVALID"
        )
        raise ProjectError(code, str(exc)[:500]) from exc
    except OSError as exc:
        raise ProjectError("FACTOR_SPEC_INVALID", str(exc)[:500]) from exc
    definition = next(iter(definitions.values()))
    try:
        params = definition.params.model_validate(spec.params)
    except ValueError as exc:
        raise ProjectError("FACTOR_PARAMS_INVALID", str(exc)[:500]) from exc
    return LoadedFactor(spec, definition, params, module, package_dir)


def load_factors(config: ProjectConfig) -> dict[str, LoadedFactor]:
    for directory in reversed(config.src_dirs):
        location = str(directory)
        if directory.is_dir() and location not in sys.path:
            sys.path.insert(0, location)
    loaded: dict[str, LoadedFactor] = {}
    for name in dict.fromkeys(config.factor_modules):
        factor = load_module_factor(name, config.root)
        if factor.spec.id in loaded:
            raise ProjectError("FACTOR_ID_DUPLICATE", factor.spec.id)
        loaded[factor.spec.id] = factor
    return loaded

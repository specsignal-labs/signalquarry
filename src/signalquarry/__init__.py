# SPDX-License-Identifier: Apache-2.0
"""SignalQuarry: agent-first strategy research with honest backtests and paper forward tests."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("signalquarry")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]

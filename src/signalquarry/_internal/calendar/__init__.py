# SPDX-License-Identifier: Apache-2.0
"""The NYSE trading calendar: which sessions exist, and which close early.

Pure computation, no network or filesystem access, so it is safe to call from the
deterministic backtest engine.
"""

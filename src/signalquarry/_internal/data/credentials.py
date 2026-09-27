# SPDX-License-Identifier: Apache-2.0
"""Market-data credentials: ``APCA_API_KEY_ID``/``APCA_API_SECRET_KEY`` or a 0600 credentials file.

The file is ``$SIGNALQUARRY_CONFIG_DIR/credentials.toml`` (default
``~/.config/signalquarry/credentials.toml``) with a ``[data]`` table holding
``key_id`` and ``secret_key``. Paper deployments use their own ``[paper.<alias>]``
tables (or ``SIGNALQUARRY_PAPER_KEY_ID``/``SIGNALQUARRY_PAPER_SECRET_KEY``), so data
and paper keys stay separate. Values are never printed; only their source is.
"""

from __future__ import annotations

import os
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SECRET_ENV = (
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "SIGNALQUARRY_PAPER_KEY_ID",
    "SIGNALQUARRY_PAPER_SECRET_KEY",
)
_LOADED: set[str] = set()


@dataclass(frozen=True)
class DataCredentials:
    key_id: str = field(repr=False)
    secret_key: str = field(repr=False)
    source: str = ""

    def __post_init__(self) -> None:
        _LOADED.update(value for value in (self.key_id, self.secret_key) if value)


def secret_values(environ: dict[str, str] | None = None) -> set[str]:
    """Every credential value this process has seen, for scrubbing output (short values skipped)."""
    env = os.environ if environ is None else environ
    values = {env.get(name, "").strip() for name in SECRET_ENV} | _LOADED
    return {value for value in values if len(value) >= 8}


def redact(text: str, secrets: set[str] | None = None) -> str:
    for value in sorted(secrets if secrets is not None else secret_values(), key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    return text


def config_dir() -> Path:
    return Path(
        os.environ.get("SIGNALQUARRY_CONFIG_DIR") or Path.home() / ".config" / "signalquarry"
    ).expanduser()


def load_data_credentials(environ: dict[str, str] | None = None) -> DataCredentials | None:
    env = os.environ if environ is None else environ
    key, secret = env.get("APCA_API_KEY_ID", "").strip(), env.get("APCA_API_SECRET_KEY", "").strip()
    if key and secret:
        return DataCredentials(key, secret, "environment")
    path = config_dir() / "credentials.toml"
    if not path.is_file():
        return None
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError("CREDENTIALS_FILE_PERMISSIONS_TOO_OPEN")
    data = tomllib.loads(path.read_text(encoding="utf-8")).get("data", {})
    key, secret = str(data.get("key_id", "")).strip(), str(data.get("secret_key", "")).strip()
    return DataCredentials(key, secret, str(path)) if key and secret else None


def load_paper_credentials(profile: str, environ: dict[str, str] | None = None) -> DataCredentials | None:
    """Keys for one paper deployment. ``profile`` is ``paper.<alias>``."""
    env = os.environ if environ is None else environ
    key = env.get("SIGNALQUARRY_PAPER_KEY_ID", "").strip()
    secret = env.get("SIGNALQUARRY_PAPER_SECRET_KEY", "").strip()
    if key and secret:
        return DataCredentials(key, secret, "environment")
    path = config_dir() / "credentials.toml"
    if not path.is_file():
        return None
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise PermissionError("CREDENTIALS_FILE_PERMISSIONS_TOO_OPEN")
    table: object = tomllib.loads(path.read_text(encoding="utf-8"))
    for part in profile.split("."):
        table = table.get(part, {}) if isinstance(table, dict) else {}
    if not isinstance(table, dict):
        return None
    key, secret = str(table.get("key_id", "")).strip(), str(table.get("secret_key", "")).strip()
    return DataCredentials(key, secret, f"{path} [{profile}]") if key and secret else None

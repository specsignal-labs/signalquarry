# SPDX-License-Identifier: Apache-2.0
"""Project only official CLI account credentials into an isolated run home."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

AUTH_FILES = {
    "codex": ".codex/auth.json",
    "claude": ".claude/.credentials.json",
    "copilot": ".copilot/config.json",
    "grok": ".grok/auth.json",
}
MAX_AUTH_BYTES = 256 * 1024


def _read_cache(root: Path, relative: str, agent_uid: int) -> dict:
    root = root.absolute()
    paths = [root, *[root / Path(relative).parts[0], root / relative]]
    for path in paths:
        try:
            info = path.lstat()
        except OSError:
            raise SystemExit("account login cache missing; complete the official CLI login first") from None
        if stat.S_ISLNK(info.st_mode) or info.st_uid == agent_uid or info.st_mode & 0o077:
            raise SystemExit("account login cache must be private and outside the agent identity")
    source = paths[-1]
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_AUTH_BYTES:
        raise SystemExit("account login cache must be a bounded regular file")
    try:
        text = source.read_text()
        if relative == ".copilot/config.json":
            # Official Copilot prepends full-line comments to its JSON cache.
            # Keep URLs inside strings intact; unsupported JSONC fails closed.
            text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))
        payload = json.loads(text)
    except (OSError, ValueError, RecursionError):
        raise SystemExit("invalid account login cache") from None
    if not isinstance(payload, dict):
        raise SystemExit("invalid account login cache")
    return payload


def _oauth_tokens(value: object) -> set[str]:
    if isinstance(value, str):
        return {value} if value.startswith("gho_") else set()
    if isinstance(value, dict):
        return set().union(*(_oauth_tokens(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_oauth_tokens(item) for item in value))
    return set()


def install_account_auth(
    agent: str, source_home: Path, home: Path, *, uid: int, gid: int
) -> tuple[dict[str, str], set[str]]:
    """Never copy settings, helpers, custom endpoints, other providers or whole homes."""
    if agent not in AUTH_FILES:
        raise SystemExit("unknown agent")
    payload = _read_cache(source_home, AUTH_FILES[agent], uid)
    env: dict[str, str] = {}
    if agent == "codex":
        tokens = payload.get("tokens")
        if payload.get("auth_mode") != "chatgpt" or not isinstance(tokens, dict):
            raise SystemExit("Codex requires ChatGPT account login; API authentication is disabled")
        if not tokens.get("access_token") or not tokens.get("refresh_token"):
            raise SystemExit("Codex ChatGPT login cache is incomplete")
        payload = {"auth_mode": "chatgpt", "tokens": tokens}
    elif agent == "claude":
        oauth = payload.get("claudeAiOauth")
        if not isinstance(oauth, dict) or not oauth.get("accessToken"):
            raise SystemExit("Claude requires claude.ai subscription login")
        if not oauth.get("subscriptionType"):
            raise SystemExit("Claude subscription type is missing; verify the subscription login")
        payload = {"claudeAiOauth": oauth}
    elif agent == "copilot":
        token_map = payload.get("authTokens")
        if token_map is not None:
            if not isinstance(token_map, dict) or not all(
                isinstance(scope, str) and scope.startswith("https://github.com:") for scope in token_map
            ):
                raise SystemExit("Copilot requires official GitHub OAuth cache scopes")
            tokens = _oauth_tokens(token_map)
        else:
            tokens = _oauth_tokens(payload.get("loggedInUsers", []))
        if len(tokens) != 1:
            raise SystemExit("Copilot requires one cached OAuth account; use a dedicated CLI login home")
        token = tokens.pop()
        # Official OAuth token from the CLI cache, never inherited PATs or BYOK keys.
        return {"COPILOT_GITHUB_TOKEN": token}, {token}
    else:
        accounts = {
            scope: item
            for scope, item in payload.items()
            if isinstance(item, dict)
            and item.get("auth_mode") in ("oidc", "web_login", "grok")
            and item.get("oidc_issuer") == "https://auth.x.ai"
            and isinstance(item.get("key"), str)
            and item["key"]
        }
        if len(accounts) != 1:
            raise SystemExit("Grok requires one official xAI OAuth login; API authentication is disabled")
        fields = {
            "key",
            "auth_mode",
            "create_time",
            "user_id",
            "email",
            "refresh_token",
            "expires_at",
            "oidc_issuer",
            "oidc_client_id",
            "coding_data_retention_opt_out",
            "principal_type",
            "principal_id",
            "team_id",
            "has_grok_code_access",
        }
        payload = {
            scope: {key: value for key, value in item.items() if key in fields}
            for scope, item in accounts.items()
        }
    target = home / AUTH_FILES[agent]
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # The run home is Linux tmpfs. Create owner-only before writing; never put
    # imported credentials on the container's persistent overlay filesystem.
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        encoded = (json.dumps(payload) + "\n").encode()
        while encoded:
            written = os.write(descriptor, encoded)
            encoded = encoded[written:]
    finally:
        os.close(descriptor)
    os.chown(target.parent, uid, gid)
    os.chown(target, uid, gid)
    if agent == "codex":
        config = target.parent / "config.toml"
        config.write_text('forced_login_method = "chatgpt"\ncli_auth_credentials_store = "file"\n')
        config.chmod(0o600)
        os.chown(config, uid, gid)
    elif agent == "claude":
        config = target.parent / "settings.json"
        config.write_text('{"forceLoginMethod":"claudeai"}\n')
        config.chmod(0o600)
        os.chown(config, uid, gid)
    secrets: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if (
                    isinstance(item, str)
                    and item
                    and (
                        "token" in key.lower()
                        or key in {"key", "user_id", "email"}
                        or "account" in key.lower()
                    )
                ):
                    secrets.add(item)
                else:
                    collect(item)

    collect(payload)
    return env, secrets

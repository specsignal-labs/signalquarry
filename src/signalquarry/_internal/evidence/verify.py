# SPDX-License-Identifier: Apache-2.0
"""Verify a SignalQuarry evidence bundle. Standard library only.

Vendorable with ``canonical.py``: copy both files side by side and run

    python verify.py BUNDLE_DIR [--max-tier public]

Checks: manifest shape, ``bundle_hash`` (canonical v2), every listed file's
SHA-256, safe relative paths, no unlisted files, and that no file exceeds the
bundle's (or the caller's maximum) disclosure tier. Exit 0 when valid, 1 when
invalid, 2 on usage errors. Output is one JSON object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

try:  # package import
    from signalquarry._internal.canonical import canonical_hash
except ImportError:  # vendored next to canonical.py
    from canonical import canonical_hash  # type: ignore[no-redef]

SCHEMA = "signalquarry-evidence-bundle/v1"
MANIFEST = "bundle.json"
TIERS = ("public", "nda")
MEDIA_TYPES = frozenset(
    {
        "application/json",
        "text/markdown",
        "image/svg+xml",
        "application/vnd.opentimestamps",
        "application/timestamp-reply",
    }
)
REQUIRED = ("schema", "bundle_id", "produced_at", "canonical", "tier", "producer", "files", "bundle_hash")
ALLOWED = frozenset(REQUIRED + ("supersedes",))
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX = re.compile(r"^[0-9a-f]{64}$")
_BUNDLE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,119}$")
_PATH = re.compile(r"^[A-Za-z0-9._/-]{1,240}$")


def _safe_path(text: str) -> bool:
    if not isinstance(text, str) or not _PATH.fullmatch(text):
        return False
    pure = PurePosixPath(text)
    return not pure.is_absolute() and ".." not in pure.parts and text != MANIFEST


def verify_bundle(root: Path, *, max_tier: str = "nda") -> dict[str, Any]:
    """Return ``{"valid": bool, "errors": [...], ...}`` for the bundle at ``root``."""
    errors: list[str] = []
    manifest_path = root / MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"valid": False, "errors": [f"MANIFEST_UNREADABLE:{type(exc).__name__}"]}
    if not isinstance(manifest, dict):
        return {"valid": False, "errors": ["MANIFEST_NOT_OBJECT"]}

    missing = [key for key in REQUIRED if key not in manifest]
    extra = sorted(set(manifest) - ALLOWED)
    errors += [f"FIELD_MISSING:{key}" for key in missing] + [f"FIELD_UNKNOWN:{key}" for key in extra]
    if manifest.get("schema") != SCHEMA:
        errors.append("SCHEMA_UNSUPPORTED")
    if manifest.get("canonical") != "v2":
        errors.append("CANONICAL_VERSION_UNSUPPORTED")
    if not isinstance(manifest.get("bundle_id"), str) or not _BUNDLE_ID.fullmatch(manifest["bundle_id"]):
        errors.append("BUNDLE_ID_INVALID")
    produced_at = manifest.get("produced_at")
    if not isinstance(produced_at, str) or not produced_at.endswith("Z"):
        errors.append("PRODUCED_AT_INVALID")
    tier = manifest.get("tier")
    if tier not in TIERS:
        errors.append("TIER_INVALID")
    if max_tier not in TIERS:
        raise ValueError("MAX_TIER_INVALID")
    elif tier in TIERS and TIERS.index(tier) > TIERS.index(max_tier):
        errors.append(f"TIER_ABOVE_MAXIMUM:{tier}")
    producer = manifest.get("producer")
    if not isinstance(producer, dict) or not isinstance(producer.get("framework_version"), str):
        errors.append("PRODUCER_INVALID")
    if "supersedes" in manifest and not (
        isinstance(manifest["supersedes"], str) and _HASH.fullmatch(manifest["supersedes"])
    ):
        errors.append("SUPERSEDES_INVALID")

    listed: set[str] = set()
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        errors.append("FILES_EMPTY_OR_INVALID")
        files = []
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "media_type", "tier"}:
            errors.append("FILE_ENTRY_INVALID")
            continue
        path = entry["path"]
        if not _safe_path(path):
            errors.append(f"FILE_PATH_UNSAFE:{path!s:.80}")
            continue
        if path in listed:
            errors.append(f"FILE_DUPLICATE:{path}")
        listed.add(path)
        if entry["media_type"] not in MEDIA_TYPES:
            errors.append(f"FILE_MEDIA_TYPE_INVALID:{path}")
        if entry["tier"] not in TIERS or (tier in TIERS and TIERS.index(entry["tier"]) > TIERS.index(tier)):
            errors.append(f"FILE_TIER_ABOVE_BUNDLE:{path}")
        if not isinstance(entry["sha256"], str) or not _HEX.fullmatch(entry["sha256"]):
            errors.append(f"FILE_HASH_INVALID:{path}")
            continue
        target = root / path
        if not target.is_file() or target.is_symlink():
            errors.append(f"FILE_MISSING:{path}")
            continue
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            errors.append(f"FILE_HASH_MISMATCH:{path}")

    present = {
        item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file() or item.is_symlink()
    } - {MANIFEST}
    errors += [f"FILE_UNLISTED:{path}" for path in sorted(present - listed)]

    body = {key: value for key, value in manifest.items() if key != "bundle_hash"}
    try:
        expected = canonical_hash(body)
    except (TypeError, ValueError) as exc:
        errors.append(f"MANIFEST_NOT_CANONICAL:{exc}")
        expected = None
    if expected is not None and manifest.get("bundle_hash") != expected:
        errors.append("BUNDLE_HASH_MISMATCH")

    return {
        "valid": not errors,
        "errors": errors,
        "bundle_id": manifest.get("bundle_id"),
        "bundle_hash": manifest.get("bundle_hash"),
        "tier": tier,
        "files": len(listed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--max-tier", choices=TIERS, default="nda")
    args = parser.parse_args(argv)
    if not args.bundle.is_dir():
        print(json.dumps({"valid": False, "errors": ["BUNDLE_DIR_MISSING"]}))
        return 2
    report = verify_bundle(args.bundle, max_tier=args.max_tier)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())

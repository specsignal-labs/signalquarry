# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import signalquarry._internal.canonical as canonical_module
import signalquarry._internal.evidence.verify as verify_module
from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.evidence.verify import verify_bundle


def _bundle(
    root: Path, files: dict[str, tuple[bytes, str]], *, tier: str = "public", **overrides: object
) -> Path:
    entries = []
    for path, (data, file_tier) in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        entries.append(
            {
                "path": path,
                "sha256": hashlib.sha256(data).hexdigest(),
                "media_type": "application/json",
                "tier": file_tier,
            }
        )
    manifest = {
        "schema": "signalquarry-evidence-bundle/v1",
        "bundle_id": "demo-bundle-1",
        "produced_at": "2026-09-25T12:00:00Z",
        "canonical": "v2",
        "tier": tier,
        "producer": {"framework_version": "0.1.0"},
        "files": entries,
        **overrides,
    }
    manifest["bundle_hash"] = canonical_hash(manifest)
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_valid_bundle(tmp_path: Path) -> None:
    report = verify_bundle(_bundle(tmp_path, {"studies/a/result.json": (b"{}", "public")}))
    assert report["valid"], report["errors"]
    assert report["files"] == 1


def test_tampered_file_and_unlisted_file_fail(tmp_path: Path) -> None:
    root = _bundle(tmp_path, {"result.json": (b"{}", "public")})
    (root / "result.json").write_bytes(b'{"x":1}')
    (root / "extra.json").write_bytes(b"{}")
    errors = verify_bundle(root)["errors"]
    assert "FILE_HASH_MISMATCH:result.json" in errors
    assert "FILE_UNLISTED:extra.json" in errors


def test_manifest_edit_breaks_bundle_hash(tmp_path: Path) -> None:
    root = _bundle(tmp_path, {"result.json": (b"{}", "public")})
    manifest = json.loads((root / "bundle.json").read_text())
    manifest["bundle_id"] = "demo-bundle-2"
    (root / "bundle.json").write_text(json.dumps(manifest))
    assert "BUNDLE_HASH_MISMATCH" in verify_bundle(root)["errors"]


def test_tier_rules(tmp_path: Path) -> None:
    public_with_nda = _bundle(tmp_path / "a", {"daily.json": (b"{}", "nda")})
    assert "FILE_TIER_ABOVE_BUNDLE:daily.json" in verify_bundle(public_with_nda)["errors"]
    nda = _bundle(tmp_path / "b", {"daily.json": (b"{}", "nda")}, tier="nda")
    assert verify_bundle(nda)["valid"]
    assert "TIER_ABOVE_MAXIMUM:nda" in verify_bundle(nda, max_tier="public")["errors"]


def test_unsafe_path_is_rejected(tmp_path: Path) -> None:
    root = _bundle(tmp_path, {"ok.json": (b"{}", "public")})
    manifest = json.loads((root / "bundle.json").read_text())
    manifest["files"][0]["path"] = "../escape.json"
    manifest.pop("bundle_hash")
    manifest["bundle_hash"] = canonical_hash(manifest)
    (root / "bundle.json").write_text(json.dumps(manifest))
    assert any(error.startswith("FILE_PATH_UNSAFE") for error in verify_bundle(root)["errors"])


def test_vendored_copy_runs_standalone(tmp_path: Path) -> None:
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    shutil.copy(canonical_module.__file__, vendor / "canonical.py")
    shutil.copy(verify_module.__file__, vendor / "verify.py")
    root = _bundle(tmp_path / "bundle", {"result.json": (b"{}", "public")})
    result = subprocess.run(
        [sys.executable, "-S", "verify.py", str(root), "--max-tier", "public"],
        cwd=vendor,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["valid"] is True

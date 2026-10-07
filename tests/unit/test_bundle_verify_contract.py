# SPDX-License-Identifier: Apache-2.0
"""Every defect the standalone bundle verifier reports, with its exact code, and its command line."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.evidence.verify import MEDIA_TYPES, main, verify_bundle

from .test_bundle_verify import _bundle

Edit = Callable[[dict[str, Any]], None]


def _fresh(tmp_path: Path, *, tier: str = "public", files: dict | None = None) -> Path:
    return _bundle(tmp_path, files or {"studies/a/result.json": (b"{}", "public")}, tier=tier)


def _rewrite(root: Path, edit: Edit, *, rehash: bool = True) -> None:
    manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    edit(manifest)
    if rehash:
        manifest.pop("bundle_hash", None)
        manifest["bundle_hash"] = canonical_hash(manifest)
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")


def _errors(root: Path, **options: Any) -> list[str]:
    return verify_bundle(root, **options)["errors"]


def test_a_valid_bundle_reports_its_identity(tmp_path: Path) -> None:
    root = _fresh(tmp_path)
    report = verify_bundle(root)
    manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    assert report == {
        "valid": True,
        "errors": [],
        "bundle_id": "demo-bundle-1",
        "bundle_hash": manifest["bundle_hash"],
        "tier": "public",
        "files": 1,
    }


@pytest.mark.parametrize("content", [None, "{broken", "[]", "5", '"text"', "null"])
def test_an_unreadable_or_non_object_manifest_stops_the_check(tmp_path: Path, content: str | None) -> None:
    (tmp_path / "x.json").write_text("{}")
    if content is not None:
        (tmp_path / "bundle.json").write_text(content, encoding="utf-8")
    report = verify_bundle(tmp_path)
    assert report["valid"] is False and len(report["errors"]) == 1 and set(report) == {"valid", "errors"}
    expected = {
        None: "MANIFEST_UNREADABLE:FileNotFoundError",
        "{broken": "MANIFEST_UNREADABLE:JSONDecodeError",
    }.get(content, "MANIFEST_NOT_OBJECT")
    assert report["errors"] == [expected]


def test_a_manifest_that_is_not_utf8_is_unreadable(tmp_path: Path) -> None:
    (tmp_path / "bundle.json").write_bytes(b"\xff\xfe{}")
    assert verify_bundle(tmp_path)["errors"] == ["MANIFEST_UNREADABLE:UnicodeDecodeError"]


def test_missing_and_unknown_manifest_fields_are_listed_in_order(tmp_path: Path) -> None:
    root = _fresh(tmp_path)
    _rewrite(root, lambda m: (m.pop("producer"), m.pop("canonical"), m.update(zeta=1, alpha=2)))
    errors = _errors(root)
    assert errors[:4] == [
        "FIELD_MISSING:canonical",
        "FIELD_MISSING:producer",
        "FIELD_UNKNOWN:alpha",
        "FIELD_UNKNOWN:zeta",
    ]
    assert "CANONICAL_VERSION_UNSUPPORTED" in errors and "PRODUCER_INVALID" in errors
    _rewrite(root, lambda m: m.pop("bundle_hash"), rehash=False)
    assert "FIELD_MISSING:bundle_hash" in _errors(root)


def test_supersedes_is_an_allowed_optional_hash(tmp_path: Path) -> None:
    root = _fresh(tmp_path)
    _rewrite(root, lambda m: m.update(supersedes="sha256:" + "a" * 64))
    assert verify_bundle(root)["valid"]
    for bad in ("sha256:" + "A" * 64, "sha256:abc", 5, None, "a" * 64):
        _rewrite(root, lambda m, bad=bad: m.update(supersedes=bad))
        assert _errors(root) == ["SUPERSEDES_INVALID"], bad


@pytest.mark.parametrize(
    ("edit", "code"),
    [
        (lambda m: m.update(schema="signalquarry-evidence-bundle/v2"), "SCHEMA_UNSUPPORTED"),
        (lambda m: m.update(canonical="v1"), "CANONICAL_VERSION_UNSUPPORTED"),
        (lambda m: m.update(bundle_id="ab"), "BUNDLE_ID_INVALID"),
        (lambda m: m.update(bundle_id="Demo-Bundle"), "BUNDLE_ID_INVALID"),
        (lambda m: m.update(bundle_id="-demo"), "BUNDLE_ID_INVALID"),
        (lambda m: m.update(bundle_id="x" * 121), "BUNDLE_ID_INVALID"),
        (lambda m: m.update(bundle_id=7), "BUNDLE_ID_INVALID"),
        (lambda m: m.update(produced_at="2026-09-25T12:00:00"), "PRODUCED_AT_INVALID"),
        (lambda m: m.update(produced_at=5), "PRODUCED_AT_INVALID"),
        (lambda m: m.update(tier="secret"), "TIER_INVALID"),
        (lambda m: m.update(tier=None), "TIER_INVALID"),
        (lambda m: m.update(producer="x"), "PRODUCER_INVALID"),
        (lambda m: m.update(producer={}), "PRODUCER_INVALID"),
        (lambda m: m.update(producer={"framework_version": 3}), "PRODUCER_INVALID"),
        (lambda m: m.update(files=[]), "FILES_EMPTY_OR_INVALID"),
        (lambda m: m.update(files="x"), "FILES_EMPTY_OR_INVALID"),
        (lambda m: m.update(files=None), "FILES_EMPTY_OR_INVALID"),
    ],
)
def test_each_header_defect_has_its_own_code(tmp_path: Path, edit: Edit, code: str) -> None:
    root = _fresh(tmp_path)
    _rewrite(root, edit)
    assert code in _errors(root)


def test_bundle_id_boundaries(tmp_path: Path) -> None:
    root = _fresh(tmp_path)
    for good in ("abc", "a" * 120, "a0._-", "0abc"):
        _rewrite(root, lambda m, good=good: m.update(bundle_id=good))
        assert "BUNDLE_ID_INVALID" not in _errors(root), good


@pytest.mark.parametrize("entry_change", [{"path": 5}, {"path": ""}, {"path": "a b.json"}, {"path": "/abs.json"},
                                           {"path": "a/../b.json"}, {"path": "bundle.json"}, {"path": "x" * 241},
                                           {"path": "ünï.json"}])  # fmt: skip
def test_unsafe_paths_are_named_and_not_read(tmp_path: Path, entry_change: dict[str, Any]) -> None:
    root = _fresh(tmp_path)
    _rewrite(root, lambda m: m["files"][0].update(entry_change))
    errors = _errors(root)
    assert any(error.startswith("FILE_PATH_UNSAFE:") for error in errors)
    assert not any(error.startswith(("FILE_MISSING", "FILE_HASH")) for error in errors)


def test_the_unsafe_path_in_an_error_is_truncated_to_eighty_characters(tmp_path: Path) -> None:
    root = _fresh(tmp_path)
    _rewrite(root, lambda m: m["files"][0].update(path="a b" * 100))
    (error,) = [e for e in _errors(root) if e.startswith("FILE_PATH_UNSAFE:")]
    assert error == "FILE_PATH_UNSAFE:" + ("a b" * 100)[:80]


def test_path_length_limit_is_inclusive(tmp_path: Path) -> None:
    long_name = "d/" * 117 + "f.json"
    assert len(long_name) == 240
    root = _fresh(tmp_path, files={long_name: (b"{}", "public")})
    assert verify_bundle(root)["valid"]


@pytest.mark.parametrize("entry", [None, "x", [], {}, {"path": "a.json"}, {"path": "a.json", "sha256": "0" * 64,
                                  "media_type": "application/json", "tier": "public", "extra": 1}])  # fmt: skip
def test_a_file_entry_must_have_exactly_four_fields(tmp_path: Path, entry: object) -> None:
    root = _fresh(tmp_path)
    _rewrite(root, lambda m: m["files"].append(entry))
    errors = _errors(root)
    assert errors.count("FILE_ENTRY_INVALID") == 1
    assert "FILE_UNLISTED:studies/a/result.json" not in errors


def test_duplicate_media_type_tier_and_hash_defects_are_reported_per_file(tmp_path: Path) -> None:
    root = _fresh(tmp_path)
    _rewrite(root, lambda m: m["files"].append(dict(m["files"][0])))
    assert _errors(root) == ["FILE_DUPLICATE:studies/a/result.json"]
    root2 = _fresh(tmp_path / "b")
    _rewrite(root2, lambda m: m["files"][0].update(media_type="text/plain"))
    assert _errors(root2) == ["FILE_MEDIA_TYPE_INVALID:studies/a/result.json"]
    for good in sorted(MEDIA_TYPES):
        _rewrite(root2, lambda m, good=good: m["files"][0].update(media_type=good))
        assert verify_bundle(root2)["valid"], good
    root3 = _fresh(tmp_path / "c")
    for bad in ("", "0" * 63, "0" * 65, "G" * 64, "A" * 64, "sha256:" + "0" * 64, 5, None):
        _rewrite(root3, lambda m, bad=bad: m["files"][0].update(sha256=bad))
        assert _errors(root3) == ["FILE_HASH_INVALID:studies/a/result.json"], bad


def test_file_tier_rules(tmp_path: Path) -> None:
    root = _fresh(tmp_path)
    _rewrite(root, lambda m: m["files"][0].update(tier="nda"))
    assert _errors(root) == ["FILE_TIER_ABOVE_BUNDLE:studies/a/result.json"]
    _rewrite(root, lambda m: m["files"][0].update(tier="secret"))
    assert _errors(root) == ["FILE_TIER_ABOVE_BUNDLE:studies/a/result.json"]
    nda = _fresh(tmp_path / "n", tier="nda", files={"a.json": (b"{}", "public")})
    assert verify_bundle(nda)["valid"]
    both = _fresh(tmp_path / "m", tier="nda", files={"a.json": (b"{}", "nda"), "b.json": (b"{}", "public")})
    assert verify_bundle(both)["valid"]
    assert _errors(both, max_tier="public") == ["TIER_ABOVE_MAXIMUM:nda"]
    assert verify_bundle(both, max_tier="nda")["valid"]


def test_an_unknown_bundle_tier_does_not_raise_the_maximum_error(tmp_path: Path) -> None:
    root = _fresh(tmp_path)
    _rewrite(root, lambda m: m.update(tier="secret"))
    errors = _errors(root)
    assert "TIER_INVALID" in errors and not any(e.startswith("TIER_ABOVE_MAXIMUM") for e in errors)


def test_an_unknown_maximum_tier_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="^MAX_TIER_INVALID$"):
        verify_bundle(_fresh(tmp_path), max_tier="secret")


def test_missing_files_symlinks_and_changed_content_are_distinguished(tmp_path: Path) -> None:
    root = _fresh(
        tmp_path,
        files={"a.json": (b"{}", "public"), "b.json": (b"{}", "public"), "c.json": (b"{}", "public")},
    )
    (root / "a.json").unlink()
    (root / "b.json").unlink()
    (root / "b.json").symlink_to(root / "c.json")
    (root / "c.json").write_bytes(b"{ }")
    errors = _errors(root)
    assert "FILE_MISSING:a.json" in errors and "FILE_MISSING:b.json" in errors
    assert "FILE_HASH_MISMATCH:c.json" in errors
    (root / "dir.json").mkdir()
    assert not any("dir.json" in error for error in _errors(root))


def test_unlisted_files_are_found_at_any_depth_including_symlinks_and_sorted(tmp_path: Path) -> None:
    root = _fresh(tmp_path)
    (root / "z.txt").write_text("z")
    (root / "deep" / "er").mkdir(parents=True)
    (root / "deep" / "er" / "a.txt").write_text("a")
    os.symlink(root / "z.txt", root / "link.json")
    errors = [e for e in _errors(root) if e.startswith("FILE_UNLISTED:")]
    assert errors == ["FILE_UNLISTED:deep/er/a.txt", "FILE_UNLISTED:link.json", "FILE_UNLISTED:z.txt"]
    assert verify_bundle(root)["files"] == 1


def test_the_bundle_hash_covers_every_other_field(tmp_path: Path) -> None:
    root = _fresh(tmp_path)
    assert "BUNDLE_HASH_MISMATCH" not in _errors(root)
    for edit in (
        lambda m: m.update(produced_at="2027-01-01T00:00:00Z"),
        lambda m: m["producer"].update(framework_version="9.9.9"),
        lambda m: m["files"][0].update(media_type="text/markdown"),
        lambda m: m.update(bundle_hash="sha256:" + "0" * 64),
        lambda m: m.update(bundle_hash=5),
    ):
        root2 = _fresh(tmp_path / str(id(edit)))
        _rewrite(root2, edit, rehash=False)
        assert "BUNDLE_HASH_MISMATCH" in _errors(root2)


def test_a_manifest_that_cannot_be_canonicalised_is_reported(tmp_path: Path) -> None:
    root = _fresh(tmp_path)
    manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    manifest["producer"]["framework_version"] = 1.5e400  # JSON infinity
    (root / "bundle.json").write_text(json.dumps(manifest).replace("Infinity", "1e999"), encoding="utf-8")
    errors = _errors(root)
    assert any(error.startswith("MANIFEST_NOT_CANONICAL:") for error in errors)
    assert "BUNDLE_HASH_MISMATCH" not in errors


def test_errors_accumulate_in_a_stable_order(tmp_path: Path) -> None:
    root = _fresh(tmp_path, tier="nda", files={"a.json": (b"{}", "nda")})
    (root / "extra.txt").write_text("x")
    _rewrite(
        root, lambda m: m.update(schema="other", bundle_id="X", produced_at="now", tier="nda"), rehash=False
    )
    assert _errors(root, max_tier="public") == [
        "SCHEMA_UNSUPPORTED",
        "BUNDLE_ID_INVALID",
        "PRODUCED_AT_INVALID",
        "TIER_ABOVE_MAXIMUM:nda",
        "FILE_UNLISTED:extra.txt",
        "BUNDLE_HASH_MISMATCH",
    ]


def test_the_command_line_reports_json_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _fresh(tmp_path)
    assert main([str(root)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["valid"] is True and out["files"] == 1
    assert capsys.readouterr().out == ""
    (root / "extra.txt").write_text("x")
    assert main([str(root), "--max-tier", "public"]) == 1
    assert json.loads(capsys.readouterr().out)["errors"] == ["FILE_UNLISTED:extra.txt"]
    assert main([str(tmp_path / "missing")]) == 2
    assert json.loads(capsys.readouterr().out) == {"valid": False, "errors": ["BUNDLE_DIR_MISSING"]}
    nda = _fresh(tmp_path / "nda", tier="nda", files={"a.json": (b"{}", "nda")})
    assert main([str(nda)]) == 0
    capsys.readouterr()
    assert main([str(nda), "--max-tier", "public"]) == 1
    capsys.readouterr()
    with pytest.raises(SystemExit) as stopped:
        main([str(nda), "--max-tier", "secret"])
    assert stopped.value.code == 2


def test_the_command_line_output_is_key_sorted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main([str(_fresh(tmp_path))])
    text = capsys.readouterr().out
    assert text.endswith("}\n")
    keys = list(json.loads(text))
    assert keys == sorted(keys)
    assert hashlib.sha256(text.encode()).hexdigest()

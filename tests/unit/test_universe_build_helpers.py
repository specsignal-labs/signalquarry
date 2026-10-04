# SPDX-License-Identifier: Apache-2.0
"""Policy limits, dollar-volume arithmetic, the private classification cache, and manifest verification."""

from __future__ import annotations

import gzip
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.data.library import LibraryError
from signalquarry._internal.data.universe_build import (
    UniverseBuildPolicy,
    _classification_cache_path,
    _dollar_volume,
    _median,
    load_classification_snapshot,
    store_classification_snapshot,
    verify_universe_manifest,
)

from .test_universe_build import _build, _inputs

D = Decimal


_UNSET = object()


def _policy(price: Any = _UNSET, age: Any = 0, percentile: Any = _UNSET) -> UniverseBuildPolicy:
    return UniverseBuildPolicy(
        D("1") if price is _UNSET else price, age, D("0") if percentile is _UNSET else percentile
    )


@pytest.mark.parametrize(
    ("policy", "detail"),
    [
        (_policy(price=1), "minimum price"),
        (_policy(price=1.0), "minimum price"),
        (_policy(price=D("0")), "minimum price"),
        (_policy(price=D("-0.01")), "minimum price"),
        (_policy(price=D("NaN")), "minimum price"),
        (_policy(price=D("Infinity")), "minimum price"),
        (_policy(age=-1), "minimum listing age"),
        (_policy(age=1.0), "minimum listing age"),
        (_policy(age=True), "minimum listing age"),
        (_policy(percentile=50), "minimum dollar-volume percentile"),
        (_policy(percentile=D("-0.0001")), "minimum dollar-volume percentile"),
        (_policy(percentile=D("100.0001")), "minimum dollar-volume percentile"),
        (_policy(percentile=D("NaN")), "minimum dollar-volume percentile"),
        (_policy(percentile=D("Infinity")), "minimum dollar-volume percentile"),
    ],
)
def test_each_policy_defect_reports_its_own_detail(policy: UniverseBuildPolicy, detail: str) -> None:
    with pytest.raises(LibraryError, match=f"^USAGE_INVALID:{detail}$"):
        policy.validate()


@pytest.mark.parametrize(
    "policy",
    [
        _policy(price=D("0.000001")),
        _policy(age=0),
        _policy(age=10_000),
        _policy(percentile=D("0")),
        _policy(percentile=D("100")),
        _policy(percentile=D("33.3333")),
    ],
)
def test_policy_limits_are_inclusive_where_stated(policy: UniverseBuildPolicy) -> None:
    policy.validate()


def test_dollar_volume_is_price_times_volume_with_exact_decimal_arithmetic() -> None:
    assert _dollar_volume(20_000_000, 1000.0) == D("20000")
    assert _dollar_volume(1, 1.0) == D("0.000001")
    assert _dollar_volume(123_456_789, 0.5) == D("61.7283945")
    assert _dollar_volume(10_000_000, 0) == D("0")
    assert _dollar_volume(10_000_000, "2.5") == D("25")
    assert _dollar_volume(3, 0.1) == D("0.0000003")  # the float 0.1 is read as 0.1, not its binary expansion


@pytest.mark.parametrize(
    "volume", [-0.0001, float("nan"), float("inf"), float("-inf"), "abc", None, [1], object()]
)
def test_a_negative_nonfinite_or_unreadable_volume_is_refused(volume: object) -> None:
    with pytest.raises(LibraryError, match="^DATA_PANEL_INVALID:volume$"):
        _dollar_volume(10_000_000, volume)


def test_the_median_of_even_and_odd_samples_and_unsorted_input() -> None:
    assert _median([D("3"), D("1"), D("2")]) == D("2")
    assert _median([D("4"), D("1"), D("3"), D("2")]) == D("2.5")
    assert _median([D("7")]) == D("7")
    assert _median([D("1"), D("2")]) == D("1.5")
    assert _median([D("5"), D("5"), D("1"), D("9")]) == D("5")


def test_cache_paths_are_content_addressed_under_a_fixed_directory(tmp_path: Path) -> None:
    digest = "cd" * 32
    assert _classification_cache_path(tmp_path, f"sha256:{digest}") == (
        tmp_path / "universe" / "classifications" / f"{digest}.json.gz"
    )
    for bad in (
        "",
        "sha256:xyz",
        "sha256:" + "A" * 64,
        "sha256:" + "a" * 63,
        "sha256:" + "a" * 65,
        "a" * 64,
        None,
        5,
    ):
        with pytest.raises(LibraryError, match="^DATA_MANIFEST_INVALID:classification hash$"):
            _classification_cache_path(tmp_path, bad)  # type: ignore[arg-type]


def _snapshot() -> tuple[dict[str, Any], str]:
    snapshot = _inputs()[2]
    return snapshot, canonical_hash(snapshot)


def test_the_stored_snapshot_is_compact_canonical_gzip_with_private_permissions(tmp_path: Path) -> None:
    snapshot, digest = _snapshot()
    path = store_classification_snapshot(tmp_path, snapshot, digest)
    assert path == _classification_cache_path(tmp_path, digest)
    body = gzip.decompress(path.read_bytes())
    assert body == json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert path.read_bytes() == gzip.compress(body, mtime=0)
    assert path.stat().st_mode & 0o777 == 0o600
    for directory in (path.parent, path.parent.parent):
        assert directory.is_dir()
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert not list(path.parent.glob("*.tmp"))


def test_storing_tightens_a_permissive_directory_and_file(tmp_path: Path) -> None:
    snapshot, digest = _snapshot()
    path = store_classification_snapshot(tmp_path, snapshot, digest)
    os.chmod(path.parent, 0o755)
    os.chmod(path, 0o644)
    store_classification_snapshot(tmp_path, snapshot, digest)
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600


def test_storing_requires_the_hash_of_the_snapshot_and_detects_a_different_cached_body(
    tmp_path: Path,
) -> None:
    snapshot, digest = _snapshot()
    with pytest.raises(LibraryError, match="^UNIVERSE_CLASSIFICATION_INVALID:classification hash$"):
        store_classification_snapshot(tmp_path, snapshot, "sha256:" + "0" * 64)
    assert not (tmp_path / "universe").exists()
    path = store_classification_snapshot(tmp_path, snapshot, digest)
    path.write_bytes(b"not gzip")
    with pytest.raises(LibraryError, match="^DATA_PAGE_CORRUPT:classification snapshot$"):
        store_classification_snapshot(tmp_path, snapshot, digest)
    path.write_bytes(gzip.compress(b"another body", mtime=0))
    with pytest.raises(LibraryError, match="^DATA_PAGE_CORRUPT:classification snapshot collision$"):
        store_classification_snapshot(tmp_path, snapshot, digest)


def test_load_reports_missing_unreadable_and_mismatching_content_distinctly(tmp_path: Path) -> None:
    snapshot, digest = _snapshot()
    with pytest.raises(
        LibraryError, match="^UNIVERSE_INPUT_UNAVAILABLE:classification snapshot cache missing$"
    ):
        load_classification_snapshot(tmp_path, digest)
    path = store_classification_snapshot(tmp_path, snapshot, digest)
    assert load_classification_snapshot(tmp_path, digest) == snapshot
    for content in (b"not gzip", gzip.compress(b"{not json"), gzip.compress(b"\xff\xfe")):
        path.write_bytes(content)
        with pytest.raises(LibraryError, match="^DATA_PAGE_CORRUPT:classification snapshot$"):
            load_classification_snapshot(tmp_path, digest)
    path.write_bytes(gzip.compress(b"[1, 2]"))
    with pytest.raises(LibraryError, match="^DATA_PAGE_CORRUPT:classification snapshot hash$"):
        load_classification_snapshot(tmp_path, digest)
    path.write_bytes(gzip.compress(b"{}"))
    with pytest.raises(LibraryError, match="^DATA_PAGE_CORRUPT:classification snapshot hash$"):
        load_classification_snapshot(tmp_path, digest)
    with pytest.raises(LibraryError, match="^DATA_MANIFEST_INVALID:classification hash$"):
        load_classification_snapshot(tmp_path, "nope")


def test_a_built_manifest_verifies_and_every_tampered_field_is_refused() -> None:
    manifest = _build()
    verify_universe_manifest(manifest)
    for key, value in (
        ("schema", "other"),
        ("member_symbols", "AAA"),
        ("member_symbols", ["AAA", ""]),
        ("member_symbols", ["AAA", 3]),
        ("member_symbols", ["BBB", "AAA"]),
        ("member_symbols", ["AAA", "AAA"]),
        ("members_hash", "sha256:" + "0" * 64),
        ("redistributable", True),
        ("redistributable", None),
        ("manifest_hash", 5),
        ("manifest_hash", "sha256:abc"),
        ("manifest_hash", "sha256:" + "0" * 64),
        ("known_at", "2000-01-01T00:00:00Z"),
    ):
        tampered = {**manifest, key: value}
        with pytest.raises(LibraryError, match="^DATA_MANIFEST_INVALID:point-in-time universe$"):
            verify_universe_manifest(tampered)
    for value in (None, [], "x", 5):
        with pytest.raises(LibraryError, match="^DATA_MANIFEST_INVALID:point-in-time universe$"):
            verify_universe_manifest(value)  # type: ignore[arg-type]
    missing = {key: item for key, item in manifest.items() if key != "members_hash"}
    with pytest.raises(LibraryError, match="^DATA_MANIFEST_INVALID:point-in-time universe$"):
        verify_universe_manifest(missing)

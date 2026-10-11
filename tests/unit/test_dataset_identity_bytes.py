# SPDX-License-Identifier: Apache-2.0
"""The dataset identity is hashed quickly, and to the bytes the canonical encoder gives."""

from __future__ import annotations

import hashlib
from datetime import date

import numpy as np
import pytest

from signalquarry._internal.canonical import canonical_bytes
from signalquarry._internal.data import dataset as module
from signalquarry._internal.data.synthetic import synthetic_dataset


def _slow(values: np.ndarray) -> bytes:
    digest = hashlib.sha256()
    digest.update(canonical_bytes(values.tolist()))
    return digest.digest()


def _fast(values: np.ndarray) -> bytes:
    digest = hashlib.sha256()
    module._hash_array(digest, values)
    return digest.digest()


@pytest.mark.parametrize(
    "values",
    [
        np.array([], dtype=np.int64),
        np.array([0, -1, 2**62, 123456789], dtype=np.int64),
        np.array([True, False, True]),
        np.array([0.0, -0.0, 1.5, 1e-7, 1e21, 123456.789, 2.0**60, 1 / 3]),
        np.arange(module.IDENTITY_ARRAY_CHUNK * 2 + 3, dtype=np.int64),
        np.linspace(0.0, 1.0, module.IDENTITY_ARRAY_CHUNK + 1),
        np.array([7, 8], dtype=np.uint32),
    ],
)
def test_arrays_hash_to_the_canonical_bytes(values: np.ndarray) -> None:
    assert _fast(values) == _slow(values)


def test_a_non_finite_value_is_refused_as_before() -> None:
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="CANONICAL_FLOAT_NOT_FINITE"):
            _fast(np.array([1.0, bad]))


def test_the_synthetic_identity_is_the_recorded_one() -> None:
    # The demo dataset's identity appears in recorded runs and documentation.
    data = synthetic_dataset(date(2014, 1, 2), date(2025, 12, 31), symbols=("SYNA",))
    assert data.identity() == "sha256:6229c5b059545c501be654386fcc482441ec74bac8085c53bb87145489022947"

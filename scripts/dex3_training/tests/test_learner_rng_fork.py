"""Paired repeats may change exactly the PRNG leaf, never learner parameters."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from fork_learner_rng import rekey_arrays


def envelope(arrays):
    return [
        {"key": k, "shape": v.shape, "dtype": v.dtype.str} for k, v in arrays.items()
    ]


def test_fork_preserves_every_non_rng_leaf_and_original_rng():
    arrays = {
        "state.0000": np.arange(12, dtype=np.float32).reshape(3, 4),
        "state.0001": np.array([2000], dtype=np.int32),
        "state.0002": np.array([123, 456], dtype=np.uint32),
    }
    new, key = rekey_arrays(arrays, envelope(arrays), np.array([0, 101], np.uint32))
    assert key == "state.0002"
    for name in ["state.0000", "state.0001"]:
        assert new[name].tobytes() == arrays[name].tobytes()
    np.testing.assert_array_equal(arrays[key], [123, 456])
    np.testing.assert_array_equal(new[key], [0, 101])


def test_ambiguous_rng_and_same_key_are_not_relabelled_as_independent():
    arrays = {"state.0000": np.array([0, 42], dtype=np.uint32)}
    with pytest.raises(ValueError, match="differ"):
        rekey_arrays(arrays, envelope(arrays), arrays["state.0000"])
    arrays["state.0001"] = arrays["state.0000"].copy()
    with pytest.raises(ValueError, match="Ambiguous"):
        rekey_arrays(arrays, envelope(arrays), np.array([0, 101], np.uint32))

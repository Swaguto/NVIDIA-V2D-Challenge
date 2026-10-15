"""Frozen collision geometry cannot be replaced or regenerated silently."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2d_dex3_flashchord import RUNTIME_COMMIT
from v2d_dex3_flashchord.reference import bind_collision_cache


def fixture(tmp_path):
    file = tmp_path / "collision-cache/convex/geometry.npz"
    file.parent.mkdir(parents=True)
    np.savez(file, n=np.array(1))
    name = file.relative_to(tmp_path).as_posix()
    manifest = dict(
        collision_cache=dict(
            directory="collision-cache", runtime_commit=RUNTIME_COMMIT, files=[name]
        ),
        files={name: hashlib.sha256(file.read_bytes()).hexdigest()},
    )
    return file, manifest


def test_geometry_cache_is_bound_to_the_readonly_input_pack(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASH_CHORD_CACHE_DIR", "/untrusted/other-cache")
    _, manifest = fixture(tmp_path)
    bind_collision_cache(tmp_path, manifest)
    import os

    assert os.environ["FLASH_CHORD_CACHE_DIR"] == str(tmp_path / "collision-cache")


def test_missing_cache_fails_before_training(tmp_path):
    with pytest.raises(ValueError, match="Frozen collision cache required"):
        bind_collision_cache(tmp_path, {})


def test_modified_cache_is_rejected(tmp_path):
    file, manifest = fixture(tmp_path)
    with file.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="hash/path mismatch"):
        bind_collision_cache(tmp_path, manifest)


def test_undeclared_cache_or_runtime_drift_is_rejected(tmp_path):
    file, manifest = fixture(tmp_path)
    np.savez(file.with_name("extra.npz"), n=np.array(1))
    with pytest.raises(ValueError, match="undeclared geometry"):
        bind_collision_cache(tmp_path, manifest)
    manifest["collision_cache"]["runtime_commit"] = "wrong"
    with pytest.raises(ValueError, match="Invalid frozen"):
        bind_collision_cache(tmp_path, manifest)

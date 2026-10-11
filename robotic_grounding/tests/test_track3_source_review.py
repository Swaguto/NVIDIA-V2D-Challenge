# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check visible-surface ownership independently of observation coverage."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/retarget"))
from track3_review_source import mask_source_hash, raster, resolve_source  # noqa: E402


def test_occluded_surface_never_replaces_visible_surface() -> None:
    """An opaque object hides a hand regardless of render order or winding."""
    intrinsic = np.array([[10.0, 0, 5], [0, 10.0, 5], [0, 0, 1.0]])
    triangle = np.array([[-0.4, -0.4, 1.0], [0.4, -0.4, 1.0], [0.0, 0.4, 1.0]])
    for reverse in (False, True):
        depth = np.full((11, 11), np.inf)
        owner = np.full((11, 11), -1, dtype=np.int16)
        sequence = [(triangle, 0), (triangle * 2, 1)]
        if reverse:
            sequence.reverse()
        for vertices, label in sequence:
            raster(vertices, np.array([[2, 1, 0]]), intrinsic, depth, owner, label)
        assert owner[5, 5] == 0
        assert depth[5, 5] == pytest.approx(1.0)
        assert owner[0, 0] == -1
        assert np.isinf(depth[0, 0])


def test_perspective_depth_matches_ray_plane_intersection() -> None:
    """Use an independent plane/ray intersection, not screen-depth interpolation."""
    intrinsic = np.array([[10.0, 0, 5], [0, 10.0, 5], [0, 0, 1.0]])
    vertices = np.array([[-0.5, -0.5, 1.0], [1.0, -1.0, 2.0], [0.0, 1.5, 3.0]])
    depth = np.full((11, 11), np.inf)
    owner = np.full((11, 11), -1, dtype=np.int16)
    raster(vertices, np.array([[0, 1, 2]]), intrinsic, depth, owner, 4)
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    for y, x in np.argwhere(owner == 4):
        ray = np.linalg.solve(intrinsic, np.array([x, y, 1.0]))
        expected = np.dot(normal, vertices[0]) / np.dot(normal, ray)
        assert depth[y, x] == pytest.approx(expected, abs=1e-12)


def test_relocated_files_use_longest_explicit_root(tmp_path: Path) -> None:
    """Relocation changes local paths without changing which source file is used."""
    mappings = [
        {"source_prefix": "/cache", "local_prefix": str(tmp_path / "all")},
        {"source_prefix": "/cache/episode", "local_prefix": str(tmp_path / "episode")},
    ]
    assert (
        resolve_source("/cache/episode/depth/000004.npy", mappings)
        == tmp_path / "episode/depth/000004.npy"
    )
    assert resolve_source("/cached/000004.npy", mappings) == Path("/cached/000004.npy")


def test_relocation_rejects_parent_traversal(tmp_path: Path) -> None:
    """A recorded source path cannot leave the explicitly mapped data directory."""
    with pytest.raises(ValueError, match="escape"):
        resolve_source(
            "/cache/../other/file.png",
            [{"source_prefix": "/cache", "local_prefix": str(tmp_path)}],
        )


def test_mask_lock_distinguishes_frames_and_identities() -> None:
    """A valid PNG from another frame or hand must not satisfy this mask lock."""
    hashes = {
        "/cache/hands/masks/left/000004.png": "left-four",
        "/cache/hands/masks/left/000005.png": "left-five",
        "/cache/hands/masks/right/000004.png": "right-four",
        "/cache/objects/pot/masks/000004.png": "pot-four",
    }
    assert mask_source_hash(hashes, 4, "left", True) == "left-four"
    assert mask_source_hash(hashes, 4, "right", True) == "right-four"
    assert mask_source_hash(hashes, 4, "pot", False) == "pot-four"
    with pytest.raises(ValueError, match="semantic mask lock"):
        mask_source_hash(hashes, 6, "left", True)


def test_conflicting_semantic_mask_locks_are_refused() -> None:
    """Two stages cannot silently declare different bytes for the same mask."""
    hashes = {
        "/original/hands/masks/left/000004.png": "original",
        "/prepared/hands/left/masks/000004.png": "different",
    }
    with pytest.raises(ValueError, match="semantic mask lock"):
        mask_source_hash(hashes, 4, "left", True)

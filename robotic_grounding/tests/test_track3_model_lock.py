# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detect physically different robot assets before collision or simulation checks."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/robotic_grounding"))

from robotic_grounding.retarget.track3_model_lock import (
    robot_asset_lock,
    verify_robot_assets,
)


def model(tmp_path: Path) -> Path:
    """Make URDFs with separate visual and collision dependencies."""
    (tmp_path / "urdfs/g1").mkdir(parents=True)
    (tmp_path / "meshes").mkdir()
    for filename in ("hand.obj", "hand.stl"):
        (tmp_path / "meshes" / filename).write_text("test mesh bytes")
    for side in ("left", "right"):
        (tmp_path / f"urdfs/g1/dex3_{side}.urdf").write_text(
            '<robot name="test"><link name="palm">'
            '<visual><geometry><mesh filename="package://hand.stl"/></geometry></visual>'
            '<collision><geometry><mesh filename="package://hand.obj"/></geometry></collision>'
            "</link></robot>"
        )
    return tmp_path


@pytest.mark.parametrize(
    "changed", ["meshes/hand.obj", "meshes/hand.stl", "urdfs/g1/dex3_left.urdf"]
)
def test_geometry_or_urdf_changes_refused(tmp_path: Path, changed: str) -> None:
    """Refuse a different mesh or URDF even when other assets still match."""
    assets = model(tmp_path)
    contract = {"robot_assets": robot_asset_lock(assets)}
    assert verify_robot_assets(contract, assets)
    path = assets / changed
    path.write_text(path.read_text() + "\n ")
    with pytest.raises(ValueError, match="differ"):
        verify_robot_assets(contract, assets)


def test_lfs_pointer_and_legacy_contract_refused(tmp_path: Path) -> None:
    """Require retrieved assets and an explicit legacy diagnostic opt-in."""
    assets = model(tmp_path)
    with pytest.raises(ValueError, match="no robot asset lock"):
        verify_robot_assets({}, assets)
    assert verify_robot_assets({}, assets, allow_legacy=True) is False
    (assets / "meshes/hand.obj").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
    )
    with pytest.raises(ValueError, match="LFS pointer"):
        robot_asset_lock(assets)


def test_lock_portable_between_directories_and_incomplete_lock_refused(
    tmp_path: Path,
) -> None:
    """Ignore machine paths but require the complete dependency closure."""
    first = model(tmp_path / "first")
    second = model(tmp_path / "second")
    contract = {"robot_assets": robot_asset_lock(first)}
    assert verify_robot_assets(contract, second)
    del contract["robot_assets"]["files_sha256"]["meshes/hand.obj"]
    with pytest.raises(ValueError, match="differ"):
        verify_robot_assets(contract, second)

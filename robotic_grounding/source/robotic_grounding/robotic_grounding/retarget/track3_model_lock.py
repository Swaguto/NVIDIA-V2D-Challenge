# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind Dex3 IK, collision checks and playback to the same robot assets."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .track3_bundle import SIDES, digest


def robot_asset_lock(assets: Path) -> dict:
    """Hash both URDFs and every referenced visual/collision mesh, portably."""
    assets = assets.resolve()
    paths = set()
    for side in SIDES:
        urdf = assets / f"urdfs/g1/dex3_{side}.urdf"
        paths.add(urdf)
        for mesh in ET.parse(urdf).getroot().iter("mesh"):
            filename = mesh.attrib["filename"]
            if not filename.startswith("package://"):
                raise ValueError(f"Expected package-relative Dex3 mesh: {filename}")
            path = (assets / "meshes" / filename.removeprefix("package://")).resolve()
            if not path.is_relative_to(assets / "meshes"):
                raise ValueError("Dex3 mesh escapes its asset directory")
            paths.add(path)
    if len(paths) <= len(SIDES):
        raise ValueError("Dex3 model has no mesh dependencies")
    files = {}
    for path in sorted(paths):
        with path.open("rb") as handle:
            if handle.read(80).startswith(b"version https://git-lfs"):
                raise ValueError(
                    f"Dex3 asset is an unretrieved LFS pointer: {path.name}"
                )
        files[path.relative_to(assets).as_posix()] = digest(path)
    return {"schema": "v2d.dex3.robot-assets.v1", "files_sha256": files}


def verify_robot_assets(
    contract: dict, assets: Path, allow_legacy: bool = False
) -> bool:
    """Refuse changed models; an explicit legacy diagnostic carries no asset proof."""
    expected = contract.get("robot_assets")
    if expected is None:
        if allow_legacy:
            return False
        raise ValueError(
            "Retarget has no robot asset lock; rerun IK or explicitly allow the unlocked legacy diagnostic"
        )
    if expected != robot_asset_lock(assets):
        raise ValueError("Robot URDF or mesh assets differ from the IK model")
    return True

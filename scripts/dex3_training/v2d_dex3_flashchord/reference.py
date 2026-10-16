"""Load explicitly approved, fully confident contiguous dual-hand windows.

No hole filling, dropped internal frames, or fabricated contacts. Unknown contact
arrays are API placeholders; Dex3ObjectiveConfig prohibits label-based rewards.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from v2d_dex3_flashchord import RUNTIME_COMMIT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_collision_cache(root, manifest, *, required=True):
    cache = manifest.get("collision_cache")
    if not cache:
        if required:
            raise ValueError(
                "Frozen collision cache required; run freeze_collision_cache.py"
            )
        return
    directory = (root / cache["directory"]).resolve()
    if (
        not directory.is_relative_to(root.resolve())
        or directory == root.resolve()
        or cache["runtime_commit"] != RUNTIME_COMMIT
        or not cache["files"]
    ):
        raise ValueError("Invalid frozen collision cache contract")
    declared = set()
    for name in cache["files"]:
        file = (root / name).resolve()
        if (
            not file.is_relative_to(directory)
            or file.suffix != ".npz"
            or name not in manifest["files"]
            or sha(file) != manifest["files"][name]
        ):
            raise ValueError("Frozen collision cache hash/path mismatch")
        declared.add(file)
    if declared != {p.resolve() for p in directory.rglob("*.npz")}:
        raise ValueError("Frozen collision cache contains undeclared geometry")
    os.environ["FLASH_CHORD_CACHE_DIR"] = str(directory)


def load_dex3(
    path,
    *,
    control_fps=None,
    motion_speed=1.0,
    source_frame_playback=False,
    require_frozen_collisions=True,
):
    path = Path(path).resolve()
    root = path.parent
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] not in {
        "v2d.dex3-training-window.v1",
        "v2d.dex3-training-window.v2",
        "v2d.dex3-training-task.v3",
    }:
        raise ValueError("Unsupported Dex3 reference manifest")
    for relative, expected in manifest["files"].items():
        file = (root / relative).resolve()
        if not file.is_relative_to(root) or sha(file) != expected:
            raise ValueError(f"Reference asset checksum/path mismatch: {relative}")
    if path.name not in manifest["files"] or sha(path) != manifest["files"][path.name]:
        raise ValueError("Reference parquet is not declared")
    bind_collision_cache(root, manifest, required=require_frozen_collisions)
    from flash_chord.data.mano_sharpa import ManoSharpaReference
    from flash_chord.data.reference import (
        ObjectAssetSpec,
        ObjectBodySpec,
        ReferenceMetadata,
    )
    import pyarrow.parquet as pq

    row = pq.ParquetFile(path).read().to_pylist()[0]
    fps = float(row["fps"])
    if motion_speed != 1 or (control_fps is not None and control_fps != fps):
        raise ValueError(
            "Dex3 confidence window currently requires source-rate playback"
        )
    with np.load(root / "targets.npz", allow_pickle=False) as source:
        data = {k: source[k].copy() for k in source.files}
    count = len(data["frame_indices"])
    if count < 2 or not np.all(np.diff(data["frame_indices"]) == 1):
        raise ValueError("Confidence window is not contiguous")
    inferred_task = manifest["schema"] == "v2d.dex3-training-task.v3"
    if inferred_task and (
        manifest.get("target_contract") != "inferred_robot_grasp_commands.v1"
        or manifest.get("allows_inferred_robot_targets") is not True
        or manifest.get("source_observations_changed") is not False
        or manifest.get("contact_labels_fabricated") is not False
    ):
        raise ValueError("Invalid inferred task target contract")
    if (not inferred_task and not data["confidence_valid"].all()) or not np.allclose(
        np.diff(data["timestamps_s"]), 1 / fps
    ):
        raise ValueError("Confidence or time-axis gap in training reference")
    for key, values in data.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite training target {key}")
    sides = ("left", "right")
    objects = manifest["objects"]
    assets = tuple(
        ObjectAssetSpec(
            name=o["name"],
            urdf_path=str(root / o["urdf"]),
            bodies=(ObjectBodySpec(o["name"], None),),
            root_reference_name=o["name"],
        )
        for o in objects
    )

    def quat(v):
        return v[..., [6, 3, 4, 5]]

    return ManoSharpaReference(
        _metadata=ReferenceMetadata(
            source_path=str(path),
            source_fps=fps,
            schema_version="dex3_dual_hand_v1",
            motion_kind="dual_hand",
            robot_name="dex3",
            sequence_id=manifest["sequence_id"],
            source_dataset="local_development",
            coord_frame="inferred support aligned world; metres; wxyz",
        ),
        _fps=fps,
        _wrist_pos_w={s: data["robot_qpos"][:, i, :3] for i, s in enumerate(sides)},
        _wrist_quat_w={s: quat(data["robot_qpos"][:, i]) for i, s in enumerate(sides)},
        _finger_joint_pos={
            s: data["robot_qpos"][:, i, 7:] for i, s in enumerate(sides)
        },
        _finger_joint_names=manifest["joint_names"],
        _object_body_pos_w=data["object_poses"][:, :, :3],
        _object_body_quat_w=quat(data["object_poses"]),
        _object_articulation=np.zeros((count, 0), dtype=np.float32),
        _contact_pos_w={s: np.zeros((count, 1, 3), dtype=np.float32) for s in sides},
        _contact_normal_w={s: np.zeros((count, 1, 3), dtype=np.float32) for s in sides},
        _contact_part_ids={s: np.zeros((count, 1), dtype=np.int32) for s in sides},
        _object_name="+".join(o["name"] for o in objects),
        _object_body_names=[o["name"] for o in objects],
        _object_mesh_paths=[str(root / o["mesh"]) for o in objects],
        _object_urdf_paths=[str(root / o["urdf"]) for o in objects],
        _object_mesh_radius=np.array(
            [o["radius_m"] for o in objects], dtype=np.float32
        ),
        _object_assets=assets,
        _frame_pos_w={s: data[f"{s}_frames"] for s in sides},
        _frame_names=manifest["frame_names"],
    )

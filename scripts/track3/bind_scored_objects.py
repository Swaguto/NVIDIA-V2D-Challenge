"""Attach reconstructed objects to unchanged scored human observations."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reconstruction/modules/v2d_issue6"))
from issue6.manifest import PATH_KEYS, label_free_video_contract, load
from issue6.track3_adapter import observed_object_state
from reconstructed_view import verify_geometry, FINAL_STAGES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(manifest, data, results, out):
    if out.exists():
        raise FileExistsError(out)
    m = load(manifest)
    contract = m.get("input_contract", {})
    if (
        not label_free_video_contract(contract)
        or contract.get("hands_only") is not True
        or contract.get("reference_pose_tables_read") is not False
        or m["objects"]
    ):
        raise ValueError(
            "Expected independent scored human observations without object geometry"
        )
    names = contract["official_object_slot_names"]
    proof = verify_geometry(data, m["episode"], names)
    if any(row["stage"] not in FINAL_STAGES for row in proof["objects"].values()):
        raise ValueError(
            "Final object integration requires completed participant reconstruction"
        )
    meshes = {
        name: (data / "mesh" / name / (name + ".glb")).resolve() for name in names
    }
    digests = {name: sha(path) for name, path in meshes.items()}
    m["objects"] = [
        dict(id=name, mesh=str(meshes[name]), provenance="predicted") for name in names
    ]
    episode_root = results / f"e{m['episode']:03d}"
    for frame in m["frames"]:
        states = {
            name: observed_object_state(
                episode_root, m["episode"], name, frame["id"], m["color_camera"]["id"]
            )
            for name in names
        }
        for name, state in states.items():
            # Image foreground is independent evidence even when 3-D tracking
            # rejects a pose. Keep it for hand occlusion/depth alignment; never
            # turn the missing object pose into an observed contact state.
            mask = episode_root / "masks" / name / f"{frame['id']:06d}.png"
            if mask.is_file():
                state["mask"] = str(mask)
                state["mask_provenance"] = (
                    "predicted_image_foreground_independent_of_pose"
                )
            if state["valid"]:
                record = json.loads(Path(state["pose_path"]).read_text())
                if (
                    record.get("object") != name
                    or record.get("mesh_sha256") != digests[name]
                ):
                    raise ValueError(
                        "Object observations are bound to different geometry"
                    )
        frame["objects"] = states
    contract.update(
        hands_only=False,
        object_geometry_read=True,
        human_observations_manifest_sha256=sha(manifest),
        reconstructed_geometry_provenance_sha256=sha(data / "geometry-provenance.json"),
        reconstructed_geometry_provenance=proof,
        human_pixels_depth_cameras_changed=False,
        image_foreground_retained_without_object_pose=True,
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    def portable(value):
        if isinstance(value, list):
            return [portable(v) for v in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            if key in PATH_KEYS and item and Path(item).is_absolute():
                result[key] = os.path.relpath(item, out.parent)
            elif key == "model_files":
                result[key] = [os.path.relpath(p, out.parent) for p in item]
            else:
                result[key] = portable(item)
        return result

    out.write_text(json.dumps(portable(m), indent=2) + "\n")
    load(out)
    return dict(
        schema="v2d.bind-scored-human-object-observations.v1",
        episode=m["episode"],
        frames=len(m["frames"]),
        object_slot_names=names,
        mesh_sha256=digests,
        human_observations_changed=False,
        reference_pose_labels_read=False,
        input_manifest_sha256=sha(manifest),
        output_manifest_sha256=sha(out),
        provided_meshes_read=False,
        physical_task_accepted=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["manifest", "data", "results", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(bind(args.manifest, args.data, args.results, args.out)))

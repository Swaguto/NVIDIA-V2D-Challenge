"""Bind one scored episode to its own reconstructed geometry; never fall back to CAD."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

FINAL_STAGES = {"multi_view_tsdf", "sam3d_metric_depth"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_geometry(data, episode, names):
    if (
        not names
        or len(names) != len(set(names))
        or any(not re.fullmatch(r"[a-zA-Z0-9_]+", n) for n in names)
    ):
        raise ValueError("Invalid object names in geometry view")
    path = data / "geometry-provenance.json"
    if not path.is_file():
        raise ValueError("Scored tracking requires a video-reconstructed geometry view")
    proof = json.loads(path.read_text())
    if (
        proof.get("schema") != "v2d.reconstructed-geometry-view.v1"
        or proof["episode"] != episode
        or proof["object_slot_names"] != names
    ):
        raise ValueError("Geometry view differs from scored episode/object slots")
    for name in names:
        item = proof["objects"][name]
        manifest = data / item["manifest"]
        if not manifest.is_file() or sha(manifest) != item["manifest_sha256"]:
            raise ValueError("Reconstruction manifest changed")
        m = json.loads(manifest.read_text())
        if (
            m.get("provided_meshes_read") is not False
            or m.get("reference_pose_labels_read") is not False
            or m["episode"] != episode
            or m["object"] != name
            or m["units"] != "metres"
            or m.get("body_origin_frozen") is not True
            or m["stage"] not in ({"depth_seed"} | FINAL_STAGES)
            or sha(data / "mesh" / name / f"{name}.glb") != m["mesh_sha256"]
        ):
            raise ValueError("Scored mesh is not bound to its video reconstruction")
        if m["stage"] == "sam3d_metric_depth":
            required = {
                "input.png",
                "mask.png",
                "pointmap.npy",
                "input-intrinsics.json",
                "raw.glb",
                "raw-transform.json",
            }
            if (
                m.get("model_revision") != "2e73555018d2741ccd486e56c24fac41155a1dc6"
                or m.get("unseen_surfaces_are_inferred") is not True
                or m.get("anchor_pose_is_inferred") is not True
                or m.get("measured_pointmap_pixels", 0) < 200
                or m.get("coordinate_frame") != "rectified_camera"
                or not required.issubset(m.get("source_files_sha256", {}))
            ):
                raise ValueError(
                    "SAM3D mesh lacks its metric source and inferred-pose contract"
                )
            for relative, digest in m["source_files_sha256"].items():
                source = (manifest.parent / relative).resolve()
                if (
                    not source.is_relative_to(manifest.parent.resolve())
                    or sha(source) != digest
                ):
                    raise ValueError("SAM3D source reconstruction input changed")
    return proof


def prepare(data, geometry, episode, names, out, allow_seed=False):
    if (
        not names
        or len(names) != len(set(names))
        or any(not re.fullmatch(r"[a-zA-Z0-9_]+", n) for n in names)
    ):
        raise ValueError("Invalid object names in geometry view")
    data, geometry, out = data.resolve(), geometry.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError(out)
    objects = {}
    for name in names:
        folder = geometry / name
        manifest = folder / "manifest.json"
        m = json.loads(manifest.read_text())
        if m["stage"] == "depth_seed" and not allow_seed:
            raise ValueError(
                "Depth seeds need explicit --allow-seed for bootstrap tracking"
            )
        if (
            m.get("provided_meshes_read") is not False
            or m.get("reference_pose_labels_read") is not False
            or m["episode"] != episode
            or m["object"] != name
            or sha(folder / "mesh.glb") != m["mesh_sha256"]
        ):
            raise ValueError("Wrong reconstructed object or mesh digest")
        objects[name] = {
            "manifest": os.path.relpath(manifest, out),
            "manifest_sha256": sha(manifest),
            "stage": m["stage"],
        }
    out.mkdir(parents=True)
    for key in ("videos", "meta"):
        (out / key).symlink_to(
            os.path.relpath(data / key, out), target_is_directory=True
        )
    for name in names:
        folder = out / "mesh" / name
        folder.mkdir(parents=True)
        (folder / f"{name}.glb").symlink_to(
            os.path.relpath(geometry / name / "mesh.glb", folder)
        )
    proof = {
        "schema": "v2d.reconstructed-geometry-view.v1",
        "episode": episode,
        "object_slot_names": names,
        "objects": objects,
        "provided_mesh_fallback": False,
        "bootstrap_seed_allowed": allow_seed,
        "submission_ready": False,
    }
    (out / "geometry-provenance.json").write_text(json.dumps(proof, indent=2) + "\n")
    verify_geometry(out, episode, names)
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "geometry", "roster", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--allow-seed", action="store_true")
    args = parser.parse_args()
    names = json.loads(args.roster.read_text())["episodes"][str(args.episode)]
    print(
        json.dumps(
            prepare(
                args.data, args.geometry, args.episode, names, args.out, args.allow_seed
            )
        )
    )

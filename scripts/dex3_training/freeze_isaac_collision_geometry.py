"""Use exact recorded Isaac convexes in the native training URDFs.

Only collision meshes are replaced. Authored visual surfaces, body/joint frames,
source trajectories, inertias, masses and motor limits remain unchanged. This
aligns a grasp's fitted geometry across engines; it does not certify retention.
"""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).parents[1] / "dex3_physics"))
from cooked_grasp_audit import world_convexes


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_hulls(parent, link, colliders, prefix):
    if re.fullmatch(r"[A-Za-z0-9_-]+", prefix) is None:
        raise ValueError("Unsafe collision mesh name")
    for old in list(link.findall("collision")):
        link.remove(old)
    records = []
    for ci, c in enumerate(colliders):
        for hi, vertices in enumerate(world_convexes(c, [0, 0, 0, 0, 0, 0, 1])):
            vertices = np.asarray(vertices, float)
            if not np.isfinite(vertices).all():
                raise ValueError("Nonfinite recorded hull")
            hull = ConvexHull(vertices)
            name = f"collision-frozen/{prefix}-{ci}-{hi}.obj"
            path = parent / name
            path.parent.mkdir(parents=True, exist_ok=True)
            text = "".join(
                "v " + " ".join(format(v, ".17g") for v in point) + "\n"
                for point in vertices
            )
            for tri, equation in zip(hull.simplices, hull.equations, strict=True):
                tri = tri.copy()
                points = vertices[tri]
                if (
                    np.cross(points[1] - points[0], points[2] - points[0])
                    @ equation[:3]
                    < 0
                ):
                    tri[[1, 2]] = tri[[2, 1]]
                text += "f " + " ".join(str(int(v) + 1) for v in tri) + "\n"
            path.write_text(text)
            collision = ET.SubElement(link, "collision", name=f"frozen-{ci}-{hi}")
            ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(ET.SubElement(collision, "geometry"), "mesh", filename=name)
            records.append(
                dict(
                    file=name,
                    sha256=sha(path),
                    vertices=len(vertices),
                    faces=len(hull.simplices),
                    minimum_body_local_m=vertices.min(0).tolist(),
                    maximum_body_local_m=vertices.max(0).tolist(),
                )
            )
    if not records:
        raise ValueError("No recorded collision hulls for link")
    return records


def freeze(inputs, cooked, out):
    original = json.loads((inputs / "manifest.json").read_text())
    for name, expected in original["files"].items():
        p = (inputs / name).resolve()
        if not p.is_relative_to(inputs.resolve()) or sha(p) != expected:
            raise ValueError("Input bytes/path changed")
    colliders = json.loads(cooked.read_text())["colliders"]
    source_report_path = cooked.parent / "report.json"
    source_report = json.loads(source_report_path.read_text())
    sample_for_object = {}
    sample_for_hand = {}
    for index, row in enumerate(source_report["provenance"]):
        sample = row["sample"]
        sample_for_object.setdefault(sample["object"], index)
        sample_for_hand.setdefault(sample["side"], index)
    if out.exists():
        raise FileExistsError(out)
    shutil.copytree(inputs, out)
    records = {}
    preserved = {}
    for side in ("left", "right"):
        asset = f"hand-{sample_for_hand[side]}"
        path = out / f"robot/urdfs/g1/dex3_{side}.urdf"
        tree = ET.parse(path)
        before = ET.tostring(tree.getroot())
        skeleton = [ET.tostring(j) for j in tree.findall("joint")]
        inertias = [ET.tostring(i) for i in tree.findall(".//inertial")]
        for c in [c for c in colliders if c["asset"] == asset]:
            name = c["body_name"]
            link = tree.find(f"link[@name='{name}']")
            if link is None:
                raise ValueError("Recorded body does not exist in the robot")
            records[side + "/" + name] = write_hulls(path.parent, link, [c], name)
        if skeleton != [ET.tostring(j) for j in tree.findall("joint")] or inertias != [
            ET.tostring(i) for i in tree.findall(".//inertial")
        ]:
            raise ValueError("Collision conversion changed mechanics")
        tree.write(path, encoding="utf-8", xml_declaration=True)
        preserved[side] = dict(
            joint_frames_preserved=True,
            inertias_preserved=True,
            before_sha256=hashlib.sha256(before).hexdigest(),
            after_sha256=sha(path),
        )
    for oi in range(len(original["objects"])):
        asset = f"object-{sample_for_object[oi]}"
        path = out / original["objects"][oi]["urdf"]
        tree = ET.parse(path)
        links = tree.findall("link")
        if len(links) != 1:
            raise ValueError(
                "Exact collision adapter currently requires one rigid object link"
            )
        inertia = ET.tostring(links[0].find("inertial"))
        records["object/" + str(oi)] = write_hulls(
            path.parent,
            links[0],
            [c for c in colliders if c["asset"] == asset],
            f"object{oi}",
        )
        if inertia != ET.tostring(links[0].find("inertial")):
            raise ValueError("Object inertia changed")
        tree.write(path, encoding="utf-8", xml_declaration=True)
    receipt = dict(
        schema="v2d.exact-recorded-collision-transfer.v1",
        source_cooked_sha256=sha(cooked),
        source_static_report_sha256=sha(source_report_path),
        source_manifest_sha256=sha(inputs / "manifest.json"),
        source_code_sha256=sha(Path(__file__)),
        hand_mechanics=preserved,
        hulls=records,
        masses_and_inertias_changed=False,
        joint_frames_changed=False,
        source_observations_changed=False,
        collision_vertices_in_body_coordinates=True,
        requires_scene_decompose_objects=False,
        physical_task_accepted=False,
    )
    (out / "collision-geometry.json").write_text(json.dumps(receipt, indent=2) + "\n")
    m = copy.deepcopy(original)
    m.update(
        collision_geometry_contract="recorded_isaac_convexes_in_body_frame.v1",
        collision_geometry_receipt_sha256=sha(out / "collision-geometry.json"),
        collision_geometry_source_sha256=sha(cooked),
        collision_decomposition_required=False,
        robot_urdf_change="Recorded cooked convex collision meshes; original visual meshes, joints, inertia and masses preserved",
        training_runtime_retention_verified=False,
    )
    m["files"] = {
        p.relative_to(out).as_posix(): sha(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p != out / "manifest.json"
    }
    (out / "manifest.json").write_text(json.dumps(m, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "cooked", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    r = freeze(a.inputs, a.cooked, a.out)
    print(json.dumps({k: v for k, v in r.items() if k != "hulls"}, indent=2))

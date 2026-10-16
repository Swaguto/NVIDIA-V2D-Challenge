"""Compare imported MuJoCo mesh vertices against the recorded body-local hulls."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vertex_distance(a, b):
    return float(max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()))


def audit(run, inputs):
    report = json.loads((run / "report.json").read_text())
    if sha(inputs / "manifest.json") != report["source_manifest_sha256"]:
        raise ValueError("Input manifest differs from simulated inputs")
    manifest = json.loads((inputs / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        p = (inputs / name).resolve()
        if not p.is_relative_to(inputs.resolve()) or sha(p) != expected:
            raise ValueError("Input bytes/path differ")
    labels = json.loads((run / "body-labels.json").read_text())
    worlds = report["metrics"]["world_count"]
    stride = len(labels) // worlds
    labels = labels[:stride]
    with np.load(run / "native-model-readback.npz") as f:
        model = {k: f[k].copy() for k in f.files}
    if len(model["body_mass"]) != stride + 1:
        raise ValueError("Unsupported native/body label mapping")
    receipt = json.loads((inputs / "collision-geometry.json").read_text())
    rows = []
    for key, hulls in receipt["hulls"].items():
        side, name = key.split("/", 1)
        if side == "object":
            asset = manifest["objects"][int(name)]
            label = asset["name"] + "/object"
            parent = inputs / Path(asset["urdf"]).parent
        else:
            label = f"dex3_{side}/{name}"
            parent = inputs / "robot/urdfs/g1"
        bi = labels.index(label) + 1  # MuJoCo includes world body 0.
        expected = []
        for hull in hulls:
            path = parent / hull["file"]
            if sha(path) != hull["sha256"]:
                raise ValueError("Hull changed")
            expected.append(
                np.array(
                    [
                        [float(v) for v in line.split()[1:]]
                        for line in path.read_text().splitlines()
                        if line.startswith("v ")
                    ]
                )
            )
        actual = []
        for gi in np.flatnonzero(
            (model["geom_bodyid"] == bi) & (model["geom_type"] == 7)
        ):
            mesh = model["geom_dataid"][gi]
            start, count = model["mesh_vertadr"][mesh], model["mesh_vertnum"][mesh]
            q = model["geom_quat"][gi]
            actual.append(
                Rotation.from_quat(q[[1, 2, 3, 0]]).apply(
                    model["mesh_vert"][start : start + count]
                )
                + model["geom_pos"][gi]
            )
        if len(actual) != len(expected):
            raise ValueError("Imported convex part count differs: " + label)
        costs = np.array([[vertex_distance(a, b) for b in actual] for a in expected])
        a, b = linear_sum_assignment(costs)
        maximum = float(costs[a, b].max())
        rows.append(
            dict(
                body=label,
                convex_parts=len(actual),
                maximum_symmetric_vertex_distance_m=maximum,
                matches_within_1um=maximum <= 1e-6,
            )
        )
    return dict(
        schema="v2d.native-collision-import-audit.v1",
        bodies=rows,
        all_imported_hulls_match=bool(rows)
        and all(r["matches_within_1um"] for r in rows),
        source_manifest_sha256=report["source_manifest_sha256"],
        native_model_sha256=sha(run / "native-model-readback.npz"),
        analysis_source_sha256=sha(Path(__file__)),
        scope="Per-convex-part body-local vertex sets after importer centering/rotation; does not establish identical contact manifolds, inertia, material response or physics engines",
        physical_task_accepted=False,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "inputs", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = audit(a.run, a.inputs)
    with a.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2))

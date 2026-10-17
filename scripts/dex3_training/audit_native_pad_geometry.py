"""Report loaded pad-normal opposition from unchanged native contact readbacks."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def weighted_normal(normals, force):
    normals, force = np.asarray(normals, float), np.asarray(force, float)
    if normals.shape != force.shape or normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("Need matching native normal and force vectors")
    if not np.isfinite(normals).all() or not np.isfinite(force).all():
        raise ValueError("Nonfinite contact geometry must remain a failed readback")
    weights = np.abs(np.sum(normals * force, axis=1))
    vector = (normals * weights[:, None]).sum(axis=0)
    length = np.linalg.norm(vector)
    return (vector / length if length > 1e-12 else np.zeros(3)), float(weights.sum())


def audit(run, inputs, grasp_audit):
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    report = json.loads((run / "report.json").read_text())
    retention = json.loads(grasp_audit.read_text())
    if retention["source_report_sha256"] != sha(run / "report.json"):
        raise ValueError("Retention and contact readback reports differ")
    if report["source_manifest_sha256"] != sha(inputs / "manifest.json"):
        raise ValueError("Native readback belongs to other inputs")
    manifest = json.loads((inputs / "manifest.json").read_text())
    if sha(inputs / "targets.npz") != manifest["files"]["targets.npz"]:
        raise ValueError("Source clock changed")
    with np.load(inputs / "targets.npz") as source:
        origin = float(source["timestamps_s"][0])
    with np.load(run / "native-contacts-readback.npz") as source:
        contacts = {k: source[k].copy() for k in source.files}
    labels = json.loads((run / "body-labels.json").read_text())
    stride = len(labels) // report["metrics"]["world_count"]
    spec = json.loads((inputs / "task-grips.json").read_text())
    phases = {p["sample"]: p for rows in spec["phases"].values() for p in rows}
    rows = []
    for grasp in retention["grasps"]:
        offset = grasp["world"] * stride
        obj = labels[:stride].index(manifest["objects"][phases[grasp["sample"]]["object"]]["name"] + "/object") + offset
        names = grasp["required_pad_names"]
        pads = [labels[:stride].index("dex3_" + grasp["side"] + "/" + n) + offset for n in names]
        first = round((grasp["start_source_time_s"] - origin) * report["control_fps"])
        last = round((grasp["end_source_time_s"] - origin) * report["control_fps"])
        normal, load = [], []
        for frame in range(first, last + 1):
            start, end = contacts["frame_offsets"][frame:frame + 2]
            a, b = contacts["body0_ids"][start:end], contacts["body1_ids"][start:end]
            ns, ls = [], []
            for pad in pads:
                pair = ((a == pad) & (b == obj)) | ((b == pad) & (a == obj))
                # Native normals point shape0 -> shape1. Orient toward the
                # assigned object, independently of contact shape ordering.
                sign = np.where(b[pair] == obj, 1., -1.)[:, None]
                n, f = weighted_normal(contacts["contact_normal_w"][start:end][pair] * sign,
                                       contacts["contact_force_w"][start:end][pair])
                ns.append(n)
                ls.append(f)
            normal.append(ns)
            load.append(ls)
        normal, load = np.asarray(normal), np.asarray(load)
        pairs = []
        for i, j in itertools.combinations(range(len(pads)), 2):
            loaded = ((load[:, i] >= .5) & (load[:, j] >= .5)
                      & (np.linalg.norm(normal[:, i], axis=1) > .5)
                      & (np.linalg.norm(normal[:, j], axis=1) > .5))
            dots = np.sum(normal[:, i] * normal[:, j], axis=1)
            pairs.append(dict(pads=[names[i], names[j]], loaded_samples=int(loaded.sum()),
                              median_loaded_normal_dot=float(np.median(dots[loaded])) if loaded.any() else None,
                              opposed_and_loaded_fraction=float((loaded & (dots <= -.7)).mean())))
        rows.append(dict(sample=grasp["sample"], side=grasp["side"], world=grasp["world"],
                         sampled_frames=len(normal), pairs=pairs))
    return dict(schema="v2d.native-pad-normal-geometry.v1", grasps=rows,
                contact_readback_sha256=sha(run / "native-contacts-readback.npz"),
                source_report_sha256=sha(run / "report.json"),
                source_manifest_sha256=report["source_manifest_sha256"],
                source_retention_audit_sha256=sha(grasp_audit),
                analysis_source_sha256=sha(Path(__file__)),
                opposition_dot_threshold=-.7, minimum_normal_load_N=.5,
                acceptance_thresholds_changed=False, physical_task_accepted=False,
                scope="Last physics substep per control frame; opposition alone does not establish retention")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "inputs", "grasp-audit", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.inputs, args.grasp_audit)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")

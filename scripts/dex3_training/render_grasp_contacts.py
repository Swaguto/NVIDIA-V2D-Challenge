"""Plot sampled native pad contacts in the moving object's local frame."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation

from render_native_readback import faces


def render(run, inputs, grasp_audit, out, sample="left-object0-grasp0", world=0):
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((run / "report.json").read_text())
    audit = json.loads(grasp_audit.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if audit["source_report_sha256"] != sha(run / "report.json"):
        raise ValueError("Contact and grasp readbacks differ")
    if report["source_manifest_sha256"] != sha(inputs / "manifest.json"):
        raise ValueError("Input and contact readbacks differ")
    manifest = json.loads((inputs / "manifest.json").read_text())
    if sha(inputs / "task-grips.json") != manifest["files"]["task-grips.json"]:
        raise ValueError("Grasp specification changed")
    grasp = next(g for g in audit["grasps"] if g["sample"] == sample and g["world"] == world)
    labels = json.loads((run / "body-labels.json").read_text())
    stride = len(labels) // report["metrics"]["world_count"]
    side = grasp["side"]
    spec = json.loads((inputs / "task-grips.json").read_text())
    phase = next(p for p in spec["phases"][side] if p["sample"] == sample)
    # The saved assignment array follows contact-layout side order. Resolve
    # the object by the named grasp instead of assuming left/right array order.
    obj = labels[:stride].index(manifest["objects"][phase["object"]]["name"] + "/object")
    pads = {labels[:stride].index(f"dex3_{side}/" + name): name for name in grasp["required_pad_names"]}
    with np.load(run / "native-state-readback.npz", allow_pickle=False) as f:
        poses = f["body_poses"]
    if not labels[obj].endswith("/object"):
        raise ValueError("Assigned object body mapping is invalid")
    with np.load(run / "native-model-readback.npz", allow_pickle=False) as f:
        model = {k: f[k] for k in f.files}
    with np.load(run / "native-contacts-readback.npz", allow_pickle=False) as f:
        contacts = {k: f[k] for k in f.files}
    palette = ["#ef795f", "#7cdb94", "#8b9cff"]
    colors = dict(zip(pads, palette, strict=True))
    meshes = []
    for gi, body_id in enumerate(model["geom_bodyid"]):
        body = int(body_id) - 1
        if body < 0 or model["geom_type"][gi] != 7:
            continue
        if body != obj and not labels[body].startswith(f"dex3_{side}/"):
            continue
        mesh = model["geom_dataid"][gi]
        start, count = model["mesh_vertadr"][mesh], model["mesh_vertnum"][mesh]
        vertices = Rotation.from_quat(model["geom_quat"][gi][[1, 2, 3, 0]]).apply(model["mesh_vert"][start:start + count]) + model["geom_pos"][gi]
        meshes.append((body, vertices, faces(vertices)))
    fps = report["control_fps"]
    if sha(inputs / "targets.npz") != manifest["files"]["targets.npz"]:
        raise ValueError("Source clock changed")
    with np.load(inputs / "targets.npz", allow_pickle=False) as source:
        origin = float(source["timestamps_s"][0])
    times = np.linspace(grasp["start_source_time_s"] + .1, grasp["end_source_time_s"] - .1, 6)
    frames = np.rint((times - origin) * fps).astype(int)
    if np.any(frames < 0) or np.any(frames >= len(poses)):
        raise ValueError("Grasp clock lies outside the saved readback")
    center = None
    plt.rcParams.update({"text.color": "#e7effa", "axes.labelcolor": "#b5c7da", "xtick.color": "#b5c7da", "ytick.color": "#b5c7da"})
    fig = plt.figure(figsize=(16, 10), facecolor="#111b2a")
    for panel, frame in enumerate(frames):
        ax = fig.add_subplot(2, 3, panel + 1, projection="3d", facecolor="#111b2a")
        object_pose = poses[frame, obj + world * stride]
        inverse = Rotation.from_quat(object_pose[3:]).inv()
        if center is None:
            palm = labels[:stride].index(f"dex3_{side}/{side}_hand_palm_link")
            center = inverse.apply(poses[frame, palm + world * stride, :3] - object_pose[:3])
        for body, vertices, triangles in meshes:
            pose = poses[frame, body + world * stride]
            points = inverse.apply(Rotation.from_quat(pose[3:]).apply(vertices) + pose[:3] - object_pose[:3])
            color = colors.get(body, "#92b7cb")
            ax.add_collection3d(Poly3DCollection(points[triangles], facecolors=color,
                edgecolors="#485d73", linewidths=.2, alpha=.15 if body == obj else .6))
        begin, end = contacts["frame_offsets"][frame:frame + 2]
        a, b = contacts["body0_ids"][begin:end], contacts["body1_ids"][begin:end]
        load_text = []
        for pad, name in pads.items():
            global_pad, global_obj = pad + world * stride, obj + world * stride
            selected = ((a == global_pad) & (b == global_obj)) | ((b == global_pad) & (a == global_obj))
            force = contacts["contact_force_w"][begin:end][selected]
            normal = contacts["contact_normal_w"][begin:end][selected]
            load = np.abs(np.sum(force * normal, axis=1)).sum()
            load_text.append(name.split("_hand_")[1].split("_")[0] + f" {load:.1f} N")
            if not selected.any():
                continue
            points = inverse.apply(contacts["contact_pos_w"][begin:end][selected] - object_pose[:3])
            ax.scatter(*points.T, s=35, c=colors[pad], edgecolors="white", linewidths=.5, depthshade=False)
        ax.set(xlim=(center[0]-.1, center[0]+.1), ylim=(center[1]-.1, center[1]+.1), zlim=(center[2]-.1, center[2]+.1), xlabel="object X (m)", ylabel="object Y (m)", zlabel="object Z (m)")
        ax.set_box_aspect([1, 1, 1])
        ax.tick_params(labelsize=7)
        for axis, value in zip((ax.xaxis, ax.yaxis, ax.zaxis), center, strict=True):
            axis.set_ticks(np.round(np.linspace(value-.1, value+.1, 3), 3))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((.08, .12, .19, 1))
        ax.view_init(elev=20, azim=-65)
        ax.set_title(f"{origin + frame/fps:.2f} s · " + " / ".join(load_text), fontsize=10)
    fig.suptitle(f"{sample}, world {world}: native collision bodies and required-pad contacts\nObject-local view; dots mark sampled contacts; labels show summed normal load. No new simulation.", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    receipt = dict(schema="v2d.native-grasp-contact-render.v1", source_report_sha256=sha(run / "report.json"),
                   source_manifest_sha256=sha(inputs / "manifest.json"),
                   contact_readback_sha256=sha(run / "native-contacts-readback.npz"),
                   model_readback_sha256=sha(run / "native-model-readback.npz"),
                   state_readback_sha256=sha(run / "native-state-readback.npz"),
                   image_sha256=sha(out), rendering_source_sha256=sha(Path(__file__)),
                   sample=sample, world=world, frames=frames.tolist(), new_simulation=False,
                   semantic_handle_contact_certified=False, physics_step_contact_continuity_certified=False)
    out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "inputs", "grasp-audit", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--sample", default="left-object0-grasp0")
    parser.add_argument("--world", default=0, type=int)
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.inputs, args.grasp_audit, args.out, args.sample, args.world)))

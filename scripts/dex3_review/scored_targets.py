"""Load complete inferred scored object targets without changing observation masks."""

import json

import numpy as np
from scipy.spatial.transform import Rotation


def load_world_targets(root, episode, names, frame_ids, mesh_hashes):
    poses, observed, provenance, files = [], [], {}, []
    for name in names:
        folder = (
            root
            / f"e{episode:03d}/ego_object_poses/e{episode:03d}"
            / name
            / "object_to_world"
        )
        values, valid, methods = [], [], {}
        for frame in frame_ids:
            path = folder / f"{int(frame):06d}.json"
            record = json.loads(
                path.read_text()
            )  # A gap is an error; never an identity target.
            q = np.asarray(record["rotation"], dtype=float)
            t = np.asarray(record["translation"], dtype=float)
            if (
                record.get("episode") != episode
                or record.get("frame_index") != int(frame)
                or record.get("object") != name
                or record.get("mesh_sha256") != mesh_hashes[name]
                or record.get("coordinate_frame") != "fixed_world"
                or record.get("gt_use") != "independent"
                or record.get("status") not in {"observed", "inferred"}
                or not np.allclose(record.get("scale", 0), 1)
                or q.shape != (4,)
                or t.shape != (3,)
                or not np.isfinite(q).all()
                or not np.isfinite(t).all()
                or not np.isclose(np.linalg.norm(q), 1, atol=1e-4)
            ):
                raise ValueError(f"Unbound or invalid scored object target: {path}")
            directly_observed = record["status"] == "observed"
            if not directly_observed and not record.get("inference_method"):
                raise ValueError("Inferred poses require their actual inference method")
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            pose[:3, 3] = t
            values.append(pose)
            valid.append(directly_observed)
            methods[str(int(frame))] = (
                "observed"
                if directly_observed
                else "inferred:" + record["inference_method"]
            )
            files.append((name, int(frame), path))
        poses.append(np.asarray(values))
        observed.append(np.asarray(valid, dtype=bool))
        provenance[name] = methods
    return np.stack(poses, axis=1), np.stack(observed, axis=1), provenance, files

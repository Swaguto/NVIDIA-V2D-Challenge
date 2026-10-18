"""Independent camera motion transports a tracked body prior between image frames."""

import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation


def load_sequence(folder, episode, count, camera="ego_cam_c"):
    poses = []
    digest = hashlib.sha256()
    for frame in range(count):
        path = folder / f"{frame:06d}.json"
        content = path.read_bytes()
        record = json.loads(content)
        q = np.asarray(record["rotation"], float)
        t = np.asarray(record["translation"], float)
        if (
            record.get("episode") != episode
            or record.get("frame_index") != frame
            or record.get("camera_frame") != camera
            or record.get("coordinate_frame") != "rectified_camera"
            or record.get("gt_use") != "independent"
            or record.get("status") != "observed"
            or q.shape != (4,)
            or t.shape != (3,)
            or not np.isfinite(q).all()
            or not np.isfinite(t).all()
            or not np.isclose(np.linalg.norm(q), 1, atol=1e-4)
        ):
            raise ValueError(
                "Camera prior needs complete independent observed rectified stereo cameras"
            )
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        pose[:3, 3] = t
        poses.append(pose)
        digest.update(path.name.encode() + b"\0" + content)
    return np.asarray(poses), digest.hexdigest()


def transport(camera_from_body, previous_camera_from_world, current_camera_from_world):
    """Preserve the prior world body pose while changing the observing camera."""
    return (
        current_camera_from_world
        @ np.linalg.inv(previous_camera_from_world)
        @ camera_from_body
    )

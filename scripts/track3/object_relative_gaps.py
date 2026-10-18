"""Conservative, optional robot-command hypotheses through missing hand intervals.

Endpoint proximity does not establish contact. This leaves observed commands,
object references and observation confidence unchanged and never extrapolates.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


def complete_object_relative_gaps(commands, anchors, times, objects):
    """Replace an interior wrist gap only when both endpoints support one object.

    ``objects`` contains ordered, fixed-scale body vertices, world transforms,
    and a mask of directly observed object poses. Finger interpolation is kept.
    Thresholds describe a development hypothesis, not a contact certificate.
    """
    q = np.asarray(commands, float)
    anchors, times = np.asarray(anchors), np.asarray(times, float)
    n = len(times)
    if (
        q.shape != (n, 2, 14)
        or anchors.shape != (n, 2)
        or anchors.dtype != bool
        or n < 2
        or not np.isfinite(q).all()
        or not np.isfinite(times).all()
        or not (np.diff(times) > 0).all()
        or not np.allclose(np.linalg.norm(q[:, :, 3:7], axis=-1), 1, atol=1e-5)
    ):
        raise ValueError(
            "Need complete ordered commands and explicit hand anchor flags"
        )
    thresholds = dict(
        maximum_endpoint_vertex_distance_m=0.12,
        minimum_object_distance_margin_m=0.01,
        maximum_relative_endpoint_translation_m=0.03,
        maximum_relative_endpoint_rotation_rad=float(np.deg2rad(30)),
        maximum_gap_duration_s=5.0,
    )
    prepared = []
    for obj in objects:
        v, poses, observed = (
            np.asarray(obj[k]) for k in ("vertices", "world", "observed")
        )
        if (
            v.ndim != 2
            or v.shape[1] != 3
            or not len(v)
            or not np.isfinite(v).all()
            or poses.shape != (n, 4, 4)
            or observed.shape != (n,)
            or observed.dtype != bool
        ):
            raise ValueError(
                "Need fixed body vertices, world poses and object observation flags"
            )
        known = poses[observed]
        r = known[:, :3, :3]
        if (
            not np.isfinite(known).all()
            or not np.allclose(known[:, 3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(np.einsum("nji,njk->nik", r, r), np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(r), 1, atol=1e-5)
        ):
            raise ValueError("Observed object transforms must be finite and rigid")
        prepared.append((cKDTree(v), poses, observed))
    out = q.copy()
    slots = np.full((n, 2), -1, dtype=np.int32)
    decisions = []
    for side in range(2):
        rows = np.flatnonzero(anchors[:, side])
        for start, end in zip(rows[:-1], rows[1:]):
            if end == start + 1:
                continue
            rec = dict(
                hand=side, start_frame=int(start), end_frame=int(end), applied=False
            )
            decisions.append(rec)
            if times[end] - times[start] > thresholds["maximum_gap_duration_s"]:
                rec["reason"] = "gap exceeds bounded duration"
                continue
            distances = []
            for tree, poses, observed in prepared:
                if not observed[start] or not observed[end]:
                    distances.append([np.inf, np.inf])
                    continue
                distances.append(
                    [
                        float(
                            tree.query(
                                np.einsum(
                                    "ji,j->i",
                                    poses[f, :3, :3],
                                    q[f, side, :3] - poses[f, :3, 3],
                                    optimize=False,
                                )
                            )[0]
                        )
                        for f in (start, end)
                    ]
                )
            if not distances:
                rec["reason"] = "no observed object candidates"
                continue
            distances = np.asarray(distances)
            chosen = distances.argmin(axis=0)
            if chosen[0] != chosen[1]:
                rec["reason"] = "endpoint object association differs"
                continue
            slot = int(chosen[0])
            nearest = distances[slot]
            if np.max(nearest) > thresholds["maximum_endpoint_vertex_distance_m"]:
                rec["reason"] = "endpoint wrist too far from object"
                continue
            alternatives = np.delete(distances, slot, axis=0)
            if (
                len(alternatives)
                and np.min(alternatives - nearest)
                < thresholds["minimum_object_distance_margin_m"]
            ):
                rec["reason"] = "ambiguous endpoint object association"
                continue
            _, poses, observed = prepared[slot]
            if not observed[start : end + 1].all():
                rec["reason"] = "object has an unobserved pose within hand gap"
                continue
            local_p = np.array(
                [
                    np.einsum(
                        "ji,j->i",
                        poses[f, :3, :3],
                        q[f, side, :3] - poses[f, :3, 3],
                        optimize=False,
                    )
                    for f in (start, end)
                ]
            )
            object_r = Rotation.from_matrix(poses[[start, end], :3, :3])
            local_r = object_r.inv() * Rotation.from_quat(q[[start, end], side, 3:7])
            drift = float(np.linalg.norm(local_p[1] - local_p[0]))
            angle = float((local_r[0].inv() * local_r[1]).magnitude())
            rec.update(
                object_slot=slot,
                endpoint_vertex_distances_m=nearest.tolist(),
                relative_translation_drift_m=drift,
                relative_rotation_drift_rad=angle,
            )
            if (
                drift > thresholds["maximum_relative_endpoint_translation_m"]
                or angle > thresholds["maximum_relative_endpoint_rotation_rad"]
            ):
                rec["reason"] = "endpoint relative wrist pose inconsistent"
                continue
            indices = np.arange(start + 1, end)
            alpha = (times[indices] - times[start]) / (times[end] - times[start])
            positions = local_p[0] + alpha[:, None] * (local_p[1] - local_p[0])
            out[indices, side, :3] = (
                np.einsum(
                    "nij,nj->ni", poses[indices, :3, :3], positions, optimize=False
                )
                + poses[indices, :3, 3]
            )
            out[indices, side, 3:7] = (
                Rotation.from_matrix(poses[indices, :3, :3])
                * Slerp(times[[start, end]], local_r)(times[indices])
            ).as_quat()
            slots[indices, side] = slot
            rec.update(
                applied=True,
                reason="consistent observed object and relative endpoint wrist poses",
                inferred_frames=len(indices),
            )
    np.testing.assert_array_equal(out[anchors], q[anchors])
    np.testing.assert_array_equal(out[:, :, 7:], q[:, :, 7:])
    return (
        out,
        slots,
        dict(
            policy="observed-object-relative-interior-gaps-v1",
            thresholds=thresholds,
            decisions=decisions,
            inferred_hand_slots=(slots >= 0).sum(axis=0).tolist(),
            observed_commands_changed=False,
            object_poses_changed=False,
            original_source_confidence_preserved=True,
            contact_assumed_confirmed=False,
            physical_task_accepted=False,
        ),
    )

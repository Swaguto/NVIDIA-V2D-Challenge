"""Experimental torque allocation from current physical contact patches.

This candidate is disabled by default. Both tested posture modes failed the
full e002 carry; analytic feasibility is not grasp or training acceptance.
"""

import numpy as np

from grip_force import point_jacobian, solve_gravity_contacts


def patch_readback(normal_data, sensor_indices, object_index):
    """Retain signed scalar evidence for selected patches, excluding capacity."""
    forces, _points, normals, _distances, counts, starts = map(np.asarray, normal_data)
    forces = forces.reshape(-1)
    result = []
    for index in sensor_indices:
        count, start = (
            int(counts[index, object_index]),
            int(starts[index, object_index]),
        )
        item = dict(sensor_index=index, count=count, start=start)
        if count > 0 and start >= 0 and start + count <= len(forces):
            f, n = forces[start : start + count], normals[start : start + count]
            if np.isfinite(f).all() and np.isfinite(n).all():
                item.update(
                    signed_scalar_min_N=float(f.min()),
                    signed_scalar_max_N=float(f.max()),
                    scalar_magnitude_sum_N=float(np.abs(f).sum()),
                    resultant_force_on_hand_N=(f[:, None] * n).sum(0).tolist(),
                )
        result.append(item)
    return result


def contact_patches(normal_data, sensor_indices, object_index, minimum_force_N=0.5):
    if not 0.05 <= minimum_force_N <= 0.5:
        raise ValueError("Invalid minimum force for live contact geometry")
    magnitudes, points, normals, _distances, counts, starts = [
        np.asarray(x) for x in normal_data
    ]
    magnitudes = magnitudes.reshape(-1)
    centers, inward = [], []
    for index in sensor_indices:
        count, start = (
            int(counts[index, object_index]),
            int(starts[index, object_index]),
        )
        if count <= 0 or start < 0 or start + count > len(magnitudes):
            raise ValueError("A required live pad contact is missing or truncated")
        signed = magnitudes[start : start + count]
        weights = np.abs(signed)
        p, n = points[start : start + count], normals[start : start + count]
        if (
            not np.isfinite(np.r_[weights, p.ravel(), n.ravel()]).all()
            or weights.sum() < minimum_force_N
        ):
            raise ValueError("Live pad pressure is insufficient or invalid")
        # The backend's signed scalar and normal jointly define force.  Use
        # magnitude only for the centroid, preserving the sign in the reaction.
        reaction = (n * signed[:, None]).sum(0)
        if np.linalg.norm(reaction) < minimum_force_N:
            raise ValueError("Live patch has cancelling or ambiguous normals")
        centers.append((p * weights[:, None]).sum(0) / weights.sum())
        # The reconstructed reaction acts on the sensor (hand).  Its opposite
        # is the force on the object, regardless of scalar/normal sign convention.
        inward.append(-reaction / np.linalg.norm(reaction))
    return np.asarray(centers), np.asarray(inward)


def motor_effort(
    points,
    inward,
    jacobians,
    object_com,
    gravity_compensation,
    mass_kg=0.3,
    desired_wrench_stage=None,
):
    gravity = np.asarray(gravity_compensation, float)
    if gravity.shape != (7,) or not np.isfinite(gravity).all():
        raise ValueError("Expected seven intrinsic joint gravity efforts")
    _forces, squeeze, plan = solve_gravity_contacts(
        points,
        inward,
        jacobians,
        object_com,
        mass_kg,
        normal_cap_N=10,
        friction=1.25,
        minimum_normal_N=2,
        desired_wrench_stage=desired_wrench_stage,
    )
    effort = squeeze + gravity
    if np.max(abs(effort)) > 0.760001:
        raise ValueError(
            "Live contact allocation plus finger gravity exceeds the original motor limit"
        )
    return effort, dict(
        plan,
        joint_gravity_compensation_Nm=gravity.tolist(),
        total_commanded_motor_effort_Nm=effort.tolist(),
    )


def allocate(
    kin,
    root,
    simulator_names,
    positions,
    pads,
    sensor_indices,
    object_index,
    normal_data,
    object_com,
    joint_gravity,
    desired_wrench_stage=None,
    minimum_patch_force_N=0.5,
):
    """Return actual simulator joint order; no state or object force is written."""
    if set(simulator_names) != set(kin.names):
        raise ValueError("Live effort joint names differ from the original URDF")
    q = dict(zip(simulator_names, positions, strict=True))
    poses = kin.poses(root, q)
    points, inward = contact_patches(
        normal_data, sensor_indices, object_index, minimum_patch_force_N
    )
    jacobians = []
    for pad, point in zip(pads, points, strict=True):
        pose = poses[pad["link"]]
        local = pose[:3, :3].T @ (point - pose[:3, 3])
        jacobians.append(point_jacobian(kin, root, q, pad["link"], local))
    gravity = np.asarray(joint_gravity)[[simulator_names.index(n) for n in kin.names]]
    effort, details = motor_effort(
        points,
        inward,
        np.asarray(jacobians),
        object_com,
        gravity,
        desired_wrench_stage=desired_wrench_stage,
    )
    details.update(
        contact_points_world_m=points.tolist(),
        inward_normals_world=inward.tolist(),
        command_scope="measured current contact geometry, nominal load, real joint torques; not accepted",
    )
    return effort[[kin.names.index(n) for n in simulator_names]], details

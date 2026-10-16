"""Robot-only payload motor commands, with a loss-of-grasp fade and shared cap."""

import warp as wp


@wp.kernel
def payload_position_bias(
    timestep: wp.array(dtype=wp.int32),
    frames: int,
    hands: int,
    dofs: int,
    bodies: int,
    object_ids: wp.array(dtype=wp.int32),
    reference: wp.array(dtype=wp.transform),
    body_pose: wp.array(dtype=wp.transform),
    pos_dofs: wp.array(dtype=wp.int32),
    wrench: wp.array(dtype=wp.spatial_vector),
    stiffness: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.float32),
):
    world, hand = wp.tid()
    frame = wp.clamp(timestep[world], 0, frames - 1)
    ri = frame * hands + hand
    error = wp.length(
        wp.transform_get_translation(body_pose[world * bodies + object_ids[hand]])
        - wp.transform_get_translation(reference[ri])
    )
    fade = wp.clamp((0.06 - error) / 0.03, 0.0, 1.0)
    fade = fade * fade * (3.0 - 2.0 * fade)
    for axis in range(3):
        dof = world * dofs + pos_dofs[hand * 3 + axis]
        target[dof] = target[dof] + fade * wrench[ri][axis] / stiffness[dof]


@wp.kernel
def payload_orientation_effort(
    timestep: wp.array(dtype=wp.int32),
    frames: int,
    hands: int,
    dofs: int,
    bodies: int,
    object_ids: wp.array(dtype=wp.int32),
    hand_ids: wp.array(dtype=wp.int32),
    reference: wp.array(dtype=wp.transform),
    body_pose: wp.array(dtype=wp.transform),
    com: wp.array(dtype=wp.vec3),
    rot_dofs: wp.array(dtype=wp.int32),
    wrench: wp.array(dtype=wp.spatial_vector),
    caps: wp.array(dtype=wp.float32),
    effort: wp.array(dtype=wp.float32),
):
    world, hand = wp.tid()
    frame = wp.clamp(timestep[world], 0, frames - 1)
    ri = frame * hands + hand
    oi = world * bodies + object_ids[hand]
    obj = body_pose[oi]
    palm = body_pose[world * bodies + hand_ids[hand]]
    error = wp.length(
        wp.transform_get_translation(obj) - wp.transform_get_translation(reference[ri])
    )
    fade = wp.clamp((0.06 - error) / 0.03, 0.0, 1.0)
    fade = fade * fade * (3.0 - 2.0 * fade)
    value = wrench[ri]
    force = wp.vec3(value[0], value[1], value[2])
    moment = wp.vec3(value[3], value[4], value[5])
    lever = wp.transform_point(obj, com[oi]) - wp.transform_get_translation(palm)
    local = wp.quat_rotate_inv(
        wp.transform_get_rotation(palm), moment + wp.cross(lever, force)
    )
    for axis in range(3):
        dof = world * dofs + rot_dofs[hand * 3 + axis]
        effort[dof] = wp.clamp(
            effort[dof] + fade * local[axis], -caps[hand], caps[hand]
        )

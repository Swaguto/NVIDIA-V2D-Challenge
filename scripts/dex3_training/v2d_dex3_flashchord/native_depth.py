"""Read sampled assigned-hand/object depth without changing contacts or states."""

import warp as wp


@wp.kernel
def maximum_assigned_depth(
    count: wp.array(dtype=wp.int32),
    shape0: wp.array(dtype=wp.int32),
    shape1: wp.array(dtype=wp.int32),
    point0: wp.array(dtype=wp.vec3),
    point1: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=wp.int32),
    pose: wp.array(dtype=wp.transform),
    bodies: int,
    hands: int,
    owner: wp.array(dtype=wp.int32),
    object_id: wp.array(dtype=wp.int32),
    depth: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if i >= count[0] or shape0[i] < 0 or shape1[i] < 0:
        return
    a, b = shape_body[shape0[i]], shape_body[shape1[i]]
    if a < 0 or b < 0 or a // bodies != b // bodies:
        return
    local_a, local_b = a % bodies, b % bodies
    hand = int(-1)
    if owner[local_a] >= 0:
        if local_b == object_id[owner[local_a]]:
            hand = owner[local_a]
    if owner[local_b] >= 0:
        if local_a == object_id[owner[local_b]]:
            hand = owner[local_b]
    if hand >= 0:
        gap = wp.dot(
            normal[i],
            wp.transform_point(pose[b], point1[i])
            - wp.transform_point(pose[a], point0[i]),
        )
        wp.atomic_max(depth, (a // bodies) * hands + hand, wp.max(0.0, -gap))

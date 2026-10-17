"""Sum native contact forces and moments about each assigned object's COM.

Fixed contact slots retain their original observations. This additional readout
preserves distributed contact couples and native torsional/rolling moments.
"""

import warp as wp


@wp.kernel
def accumulate_contact_wrench(
    count: wp.array(dtype=wp.int32),
    shape0: wp.array(dtype=wp.int32),
    shape1: wp.array(dtype=wp.int32),
    point0: wp.array(dtype=wp.vec3),
    point1: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.spatial_vector),
    shape_body: wp.array(dtype=wp.int32),
    shape_hand: wp.array(dtype=wp.int32),
    shapes_per_world: int,
    bodies_per_world: int,
    hands: int,
    assigned_objects: wp.array(dtype=wp.int32),
    poses: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    result_force: wp.array(dtype=wp.vec3),
    result_moment: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if i >= count[0]:
        return
    s0, s1 = shape0[i], shape1[i]
    if s0 < 0 or s1 < 0 or s0 >= shape_body.shape[0] or s1 >= shape_body.shape[0]:
        return
    world = s0 // shapes_per_world
    if s1 // shapes_per_world != world:
        return
    b0, b1 = shape_body[s0], shape_body[s1]
    if b0 < 0 or b1 < 0:
        return
    h0 = shape_hand[s0 % shapes_per_world]
    h1 = shape_hand[s1 % shapes_per_world]
    hand, obj = int(-1), int(-1)
    position, f, moment = wp.vec3(), wp.vec3(), wp.vec3()
    if h0 >= 0 and b1 % bodies_per_world == assigned_objects[h0]:
        hand, obj = h0, b1
        position = wp.transform_point(poses[b1], point1[i])
        f = -wp.spatial_top(force[i])
        moment = -wp.spatial_bottom(force[i])
    elif h1 >= 0 and b0 % bodies_per_world == assigned_objects[h1]:
        hand, obj = h1, b0
        position = wp.transform_point(poses[b0], point0[i])
        f = wp.spatial_top(force[i])
        moment = wp.spatial_bottom(force[i])
    if hand >= 0:
        center = wp.transform_point(poses[obj], body_com[obj])
        slot = world * hands + hand
        wp.atomic_add(result_force, slot, f)
        wp.atomic_add(result_moment, slot, moment + wp.cross(position - center, f))


class RawContactWrench:
    def __init__(self, contact, model, assigned_objects, hands, device):
        self.contact, self.model = contact, model
        self.assigned_objects, self.hands = assigned_objects, hands
        self.force = wp.zeros(contact.world_count * hands, dtype=wp.vec3, device=device)
        self.moment = wp.zeros_like(self.force)
        original_update = contact.update

        def update(model, state, contacts):
            original_update(model, state, contacts)
            self.compute(state, contacts)

        contact.update = update

    def compute(self, state, contacts):
        if contacts.force is None:
            raise ValueError("Raw wrench rewards need native contact forces")
        self.force.zero_()
        self.moment.zero_()
        wp.launch(
            accumulate_contact_wrench,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.force,
                self.model.shape_body,
                self.contact.shape_hand_ids,
                self.contact.shapes_per_world,
                self.contact.bodies_per_world,
                self.hands,
                self.assigned_objects,
                state.body_q,
                self.model.body_com,
                self.force,
                self.moment,
            ],
        )

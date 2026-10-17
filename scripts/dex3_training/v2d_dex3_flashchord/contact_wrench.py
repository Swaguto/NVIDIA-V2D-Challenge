"""Sum native contact forces and moments about each assigned object's COM.

Fixed contact slots retain their original observations. This additional readout
preserves distributed contact couples and native torsional/rolling moments.
"""

import warp as wp
from .native_contact_binding import invert_body_mapping


@wp.kernel
def accumulate_contact_wrench(
    count: wp.array(dtype=wp.int32),
    shape0: wp.array(dtype=wp.int32),
    shape1: wp.array(dtype=wp.int32),
    contact_position: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.spatial_vector),
    shape_body: wp.array(dtype=wp.int32),
    shape_hand: wp.array(dtype=wp.int32),
    shapes_per_world: int,
    bodies_per_world: int,
    hands: int,
    assigned_objects: wp.array(dtype=wp.int32),
    object_to_native: wp.array(dtype=wp.int32),
    native_com: wp.array2d(dtype=wp.vec3),
    result_force: wp.array(dtype=wp.vec3),
    result_moment: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if i >= count[0]:
        return
    s0 = int(shape0[i])
    s1 = int(shape1[i])
    if s0 < 0 or s1 < 0 or s0 >= shape_body.shape[0] or s1 >= shape_body.shape[0]:
        return
    world = s0 // shapes_per_world
    if s1 // shapes_per_world != world:
        return
    b0 = int(shape_body[s0])
    b1 = int(shape_body[s1])
    if b0 < 0 or b1 < 0:
        return
    h0 = int(shape_hand[s0 % shapes_per_world])
    h1 = int(shape_hand[s1 % shapes_per_world])
    hand, obj = int(-1), int(-1)
    f, moment = wp.vec3(), wp.vec3()
    if h0 >= 0 and b1 % bodies_per_world == assigned_objects[h0]:
        hand = h0
        obj = b1
        f = -wp.spatial_top(force[i])
        moment = -wp.spatial_bottom(force[i])
    elif h1 >= 0 and b0 % bodies_per_world == assigned_objects[h1]:
        hand = h1
        obj = b0
        f = wp.spatial_top(force[i])
        moment = wp.spatial_bottom(force[i])
    if hand >= 0:
        # Read both quantities at the force solve's time. Newton's returned
        # state is post-integration; its local surface points plus that state
        # cannot recover the native force application point at the earlier time.
        position = contact_position[i]
        center = native_com[world, object_to_native[obj]]
        slot = world * hands + hand
        wp.atomic_add(result_force, slot, f)
        wp.atomic_add(result_moment, slot, moment + wp.cross(position - center, f))


class RawContactWrench:
    def __init__(self, contact, model, assigned_objects, hands, device):
        self.contact, self.model = contact, model
        self.assigned_objects, self.hands = assigned_objects, hands
        binding = getattr(model, "dex3_native_contact_readout", None)
        if binding is None:
            raise ValueError("Raw wrench rewards require native contact-time readout")
        self.native = binding.data
        inverse = invert_body_mapping(
            binding.body_to_newton.numpy(), contact.bodies_per_world
        )
        for world in range(contact.world_count):
            for obj in assigned_objects.numpy():
                if not 0 <= obj < contact.bodies_per_world or inverse[
                    world * contact.bodies_per_world + obj
                ] < 0:
                    raise ValueError("Assigned grip object has no native COM mapping")
        self.object_to_native = wp.array(inverse, dtype=wp.int32, device=device)
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
                self.native.contact.pos,
                contacts.force,
                self.model.shape_body,
                self.contact.shape_hand_ids,
                self.contact.shapes_per_world,
                self.contact.bodies_per_world,
                self.hands,
                self.assigned_objects,
                self.object_to_native,
                self.native.xipos,
                self.force,
                self.moment,
            ],
        )

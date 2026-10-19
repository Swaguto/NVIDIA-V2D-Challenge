"""Strict summary of native same-hand collision records, including empty knots."""
import numpy as np


def summarize_self_contacts(frames, hand_bodies, *, reference_frame_count,
                            tolerance_m=0.001):
    """Assess both hands over the complete declared clock.

    Records concern native active collision geometry, not measured forces or
    arbitrary mesh intersections. The producer must preserve native filters,
    provide every knot and bind body names/poses to the actual model.
    """
    if (isinstance(reference_frame_count, (bool, np.bool_))
            or not isinstance(reference_frame_count, (int, np.integer))
            or reference_frame_count <= 0 or len(frames) != reference_frame_count):
        raise ValueError('Need every declared reference knot')
    if (isinstance(tolerance_m, (bool, np.bool_)) or not np.isscalar(tolerance_m)
            or not np.isfinite(tolerance_m) or tolerance_m < 0):
        raise ValueError('Need a finite nonnegative tolerance')
    if set(hand_bodies) != {'left', 'right'}:
        raise ValueError('Need explicit left and right body registries')
    registries = {}
    for side in ('left', 'right'):
        names = hand_bodies[side]
        if (not isinstance(names, (list, tuple)) or len(names) < 2
                or any(not isinstance(n, str) or not n for n in names)
                or len(set(names)) != len(names)):
            raise ValueError('Need distinct native body names per hand')
        registries[side] = set(names)
    if registries['left'] & registries['right']:
        raise ValueError('Hand body registries overlap')
    by_hand = {side: np.zeros(reference_frame_count) for side in registries}
    pairs = {}
    for index, row in enumerate(frames):
        frame = row['reference_frame']
        if isinstance(frame, (bool, np.bool_)) or not isinstance(frame, (int, np.integer)) or frame != index:
            raise ValueError('Need every ordered reference frame exactly once')
        if not isinstance(row['contacts'], list):
            raise ValueError('Need explicit contact lists, including empty lists')
        for contact in row['contacts']:
            side = contact['hand']; bodies = contact['native_bodies']; gap = contact['signed_gap_m']
            if (side not in registries or not isinstance(bodies, (list, tuple))
                    or len(bodies) != 2 or len(set(bodies)) != 2
                    or not all(b in registries[side] for b in bodies)):
                raise ValueError('Self-contact bodies must belong to the declared hand')
            if isinstance(gap, (bool, np.bool_)) or not np.isscalar(gap) or not np.isfinite(gap):
                raise ValueError('Need finite native signed contact gaps')
            penetration = max(0., -float(gap))
            by_hand[side][index] = max(by_hand[side][index], penetration)
            key = (side, *sorted(bodies))
            depths = pairs.setdefault(key, {})
            depths[index] = max(depths.get(index, 0.), penetration)
    hand_summary = {side: dict(
        frames_over_tolerance=int(np.count_nonzero(depths > tolerance_m)),
        maximum_penetration_m=float(depths.max()),
        worst_reference_frame=int(depths.argmax()),
    ) for side, depths in by_hand.items()}
    pair_summary = [dict(hand=key[0], native_bodies=list(key[1:]),
        contact_frames=len(depths),
        frames_over_tolerance=sum(d > tolerance_m for d in depths.values()),
        maximum_penetration_m=max(depths.values()),
        worst_reference_frame=max(depths, key=depths.get))
        for key, depths in sorted(pairs.items())]
    return dict(reference_knots=int(reference_frame_count), reporting_tolerance_m=float(tolerance_m),
        summary_by_hand=hand_summary, summary_by_pair=pair_summary,
        hand_self_contact_gate_passed=all(x['frames_over_tolerance'] == 0 for x in hand_summary.values()),
        native_filters_require_producer_binding=True, forces_measured=False,
        dynamic_or_grasp_success_certified=False)

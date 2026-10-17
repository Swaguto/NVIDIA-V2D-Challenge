"""Permit reference-state practice with zero assistance and fixed objectives."""

import math


def objective_weights(config):
    names = (
        "object_keypoints",
        "hand_keypoints",
        "hand_joint_pos",
        "contact_wrench_support",
        "missed_contact",
        "unintended_contact",
        "termination",
        "action_rate_l2",
        "action_l2",
        "contact_force_l2",
        "force_closure",
    )
    result = {
        name: float(config[name]["weight"]) if config[name]["enabled"] else 0.0
        for name in names
    }
    for name, field in [
        ("robot_pad_support", "pad_support_weight"),
        ("robot_grip_stability", "grip_stability_weight"),
        ("robot_carried_load", "carried_load_weight"),
        ("robot_carried_height", "carried_height_weight"),
        ("robot_excess_pad_force", "excess_pad_force_weight"),
    ]:
        if field in config:
            result[name] = float(config[field])
    return result


def validate(curriculum, weights):
    if curriculum is None:
        return
    if curriculum.get("_target_") != "flash_chord.lifecycle.curriculum.FixedCurriculum":
        raise ValueError("Only an explicit fixed reset curriculum is supported")
    thresholds, stages = curriculum["thresholds"], curriculum["stages"]
    if len(stages) < 2 or len(thresholds) != len(stages) or thresholds[0] <= 0:
        raise ValueError(
            "Need an ordered reset curriculum with a frame-zero final stage"
        )
    if any(a >= b for a, b in zip(thresholds, thresholds[1:])):
        raise ValueError("Reset curriculum thresholds must increase")
    probabilities = []
    for stage in stages:
        if (
            stage["voc_scale"] != 0
            or stage.get("immediate_first_frame_probability") is not None
        ):
            raise ValueError(
                "Reset curriculum may not add object or settling assistance"
            )
        p = stage.get("reset_to_first_frame_probability")
        if p is None or not math.isfinite(p) or not 0 < p <= 1:
            raise ValueError("Every stage needs a bounded frame-zero reset probability")
        probabilities.append(p)
        declared = stage["objective_weights"]
        if set(declared) != set(weights) or any(
            not math.isfinite(float(v))
            or not math.isclose(float(v), float(weights[k]), rel_tol=1e-6, abs_tol=1e-8)
            for k, v in declared.items()
        ):
            raise ValueError("A reset curriculum must preserve every objective weight")
        for name in (
            "contact_wrench_support",
            "missed_contact",
            "unintended_contact",
            "force_closure",
        ):
            if declared.get(name, 0) != 0:
                raise ValueError("Unknown human contact labels remain disabled")
    if probabilities != sorted(probabilities) or probabilities[-1] != 1.0:
        raise ValueError("Reset curriculum must finish with only frame-zero starts")

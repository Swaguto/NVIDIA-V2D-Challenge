"""Validate explicit training curricula; deployment always uses zero assistance."""

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


def assistance_contract(config):
    """Opt in to bounded training assistance without relaxing archived recipes."""
    mode = config.get("training_assistance", "disabled")
    if mode not in ("disabled", "decaying_voc"):
        raise ValueError("Unknown training assistance mode")
    assisted = mode == "decaying_voc"
    if config["logging"]["mode"] != "disabled" or config["env"]["voc_scale"] != 0:
        raise ValueError("Training requires local logging and a zero default VOC")
    if not assisted:
        if config["voc_scale_override"] != 0 or config["reset"]["voc_decay_steps"] != 0:
            raise ValueError("Unassisted training may not add object or reset assistance")
        return None
    if config["voc_scale_override"] is not None:
        raise ValueError("Decaying training VOC must use its declared curriculum")
    reset = config["reset"]
    if (
        reset["_target_"] != "flash_chord.lifecycle.reset.ReferenceResetConfig"
        or not 0 < reset["voc_decay_steps"] <= 20
        or reset["voc_decay_mode"] != "linear"
        or reset["first_frame_voc_threshold"] <= 1.0
    ):
        raise ValueError("Assisted training needs bounded linear reference settling")
    curriculum = config.get("curriculum")
    if not curriculum or len(curriculum["stages"]) < 2:
        raise ValueError("Assisted training needs an explicit decaying curriculum")
    scales = [s["voc_scale"] for s in curriculum["stages"]]
    if (
        any(not math.isfinite(v) or not 0 <= v <= 1 for v in scales)
        or scales != sorted(scales, reverse=True)
        or scales[0] <= 0
        or scales[-1] != 0
    ):
        raise ValueError("Training VOC must decay from (0,1] to zero")
    # FixedCurriculum enters stage i at threshold i-1, not threshold i.
    first_zero = scales.index(0)
    zero_start = curriculum["thresholds"][first_zero - 1]
    maximum = config["training"]["max_iterations"]
    if zero_start > 0.75 * maximum:
        raise ValueError("Reserve at least a quarter of training for zero target VOC")
    return dict(
        mode=mode,
        training_only=True,
        target_scales=scales,
        target_zero_from_iteration=zero_start,
        reset_stabilization_steps=reset["voc_decay_steps"],
        reset_stabilization_seconds=reset["voc_decay_steps"] / config["sim"]["fps"],
        reset_assistance_continues_in_zero_target_stage=True,
        evaluation_voc_scale=0.0,
        evaluation_settling_steps=0,
        evaluation_starts_at_frame_zero=True,
    )


def validate(curriculum, weights, *, allow_training_assistance=False):
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
            (stage["voc_scale"] != 0 and not allow_training_assistance)
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

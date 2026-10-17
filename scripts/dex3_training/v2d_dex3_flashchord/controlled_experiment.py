"""Validate one declared training factor while preserving the complete learner.

These are controlled forks, not unchanged continuations. Inputs, physics,
actions, observations, optimizer settings and reset distributions stay bound by
the original resume validator. Final evaluation composition stays upstream.
"""

import math
from copy import deepcopy


def normalize(parent, candidate):
    config = deepcopy(candidate)
    spec = config.pop("controlled_experiment")
    if parent.get("controlled_experiment") is not None:
        raise ValueError("Start a controlled fork from the original control checkpoint")
    if spec.get("schema") != "v2d.single-factor-experiment.v1":
        raise ValueError("Unknown controlled experiment schema")
    kind = spec.get("kind")
    change = {"kind": kind}
    if kind == "control":
        allowed = {"schema", "kind"}
    elif kind == "objective_weight":
        allowed = {"schema", "kind", "term", "multiplier"}
        name, scale = spec.get("term"), spec.get("multiplier")
        if (
            name not in ("object_keypoints", "contact_wrench_support")
            or isinstance(scale, bool)
            or not isinstance(scale, (float, int))
            or not math.isfinite(scale)
            or not 0.25 <= scale <= 4
            or scale == 1
        ):
            raise ValueError("Invalid single objective weight experiment")
        old = parent["objective"][name]
        new = config["objective"][name]
        if (
            not old["enabled"]
            or old["weight"] <= 0
            or new["weight"] != old["weight"] * scale
        ):
            raise ValueError("Objective weight does not match the declared factor")
        new["weight"] = old["weight"]
        previous = parent["curriculum"]["stages"]
        stages = config["curriculum"]["stages"]
        if len(stages) != len(previous):
            raise ValueError("Controlled curriculum length changed")
        for before, after in zip(previous, stages, strict=True):
            if (
                after["objective_weights"][name]
                != before["objective_weights"][name] * scale
            ):
                raise ValueError(
                    "Every curriculum stage must use the same weight factor"
                )
            after["objective_weights"][name] = before["objective_weights"][name]
        change.update(
            term=name,
            parent_weight=old["weight"],
            multiplier=scale,
            candidate_weight=old["weight"] * scale,
        )
    elif kind == "reset_stabilization":
        allowed = {"schema", "kind", "steps"}
        steps = spec.get("steps")
        old_steps = parent["reset"]["voc_decay_steps"]
        if (
            type(steps) is not int
            or not 0 < steps < old_steps <= 20
            or config["reset"]["voc_decay_steps"] != steps
            or config["env"]["reset"]["voc_decay_steps"] != steps
        ):
            raise ValueError(
                "Reset experiment may only shorten the bounded settling period"
            )
        config["reset"]["voc_decay_steps"] = old_steps
        config["env"]["reset"]["voc_decay_steps"] = parent["env"]["reset"][
            "voc_decay_steps"
        ]
        change.update(parent_steps=old_steps, candidate_steps=steps)
    else:
        raise ValueError("Unsupported controlled experiment factor")
    if set(spec) != allowed:
        raise ValueError("Unexpected controlled experiment fields")
    # Hydra also resolves objective aliases under env; undo only that same
    # scalar there. Any extra change remains visible to strict comparison.
    if kind == "objective_weight":
        name = spec["term"]
        old = parent["env"]["objective"][name]
        if (
            config["env"]["objective"][name]["weight"]
            != old["weight"] * spec["multiplier"]
        ):
            raise ValueError("Resolved environment objective does not match the factor")
        config["env"]["objective"][name]["weight"] = old["weight"]
    # TrainingConfig embeds the resolved curriculum as well.
    if kind == "objective_weight":
        for before, after in zip(
            parent["training"]["curriculum"]["stages"],
            config["training"]["curriculum"]["stages"],
            strict=True,
        ):
            name = spec["term"]
            if (
                after["objective_weights"][name]
                != before["objective_weights"][name] * spec["multiplier"]
            ):
                raise ValueError(
                    "Resolved learner curriculum does not match the factor"
                )
            after["objective_weights"][name] = before["objective_weights"][name]
    return config, dict(
        schema=spec["schema"],
        **change,
        unchanged_experiment=False,
        complete_learner_restored=True,
        new_training_counters_continue_parent=True,
        simulator_state_restored=False,
        evaluation_composition_changed=False,
    )

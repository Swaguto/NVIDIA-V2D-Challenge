"""Attribute saved physics-rollout rewards without changing rewards or scores.

Terms may average multiple active hands. A time window is not a per-hand audit.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run, checkpoint, start_s=2.4, end_s=11.5):
    report = json.loads((run / "report.json").read_text())
    if report["checkpoint_sha256"] != digest(checkpoint):
        raise ValueError("Reward checkpoint differs from measured policy")
    with checkpoint.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < length < 16 * 1024 * 1024:
            raise ValueError("Invalid checkpoint header size")
        metadata = json.loads(stream.read(length))["__metadata__"]
    if metadata["user.dex3_input_manifest_sha256"] != report["source_manifest_sha256"]:
        raise ValueError("Reward input manifest differs")
    config = json.loads(metadata["user.resolved_config_json"])
    names = json.loads((run / "objective-term-labels.json").read_text())
    objective = config["objective"]
    weights = {
        name: float(objective[name]["weight"]) if objective[name]["enabled"] else 0.0
        for name in names if not name.startswith("robot_")
    }
    for name, field in (
        ("robot_pad_support", "pad_support_weight"),
        ("robot_grip_stability", "grip_stability_weight"),
        ("robot_carried_load", "carried_load_weight"),
        ("robot_carried_height", "carried_height_weight"),
        ("robot_excess_pad_force", "excess_pad_force_weight"),
    ):
        if name in names:
            weights[name] = float(objective[field])
    with np.load(run / "native-state-readback.npz", allow_pickle=False) as saved:
        terms = saved["objective_terms"].reshape(report["frame_count"], -1, len(names))
        score = saved["objective_score"]
    fps = report["control_fps"]
    contribution = terms * np.array([weights[n] for n in names]) / fps
    if not np.isfinite(contribution).all() or not np.allclose(contribution.sum(-1), score, atol=1e-6, rtol=1e-5):
        raise ValueError("Saved reward terms do not reconstruct actual rollout scores")
    times = np.arange(len(terms)) / fps
    selected = (times >= start_s) & (times < end_s)
    if not selected.any():
        raise ValueError("Empty reward analysis window")
    windows = {}
    for title, mask in (("full_rollout", np.ones(len(terms), bool)), ("selected_time_window", selected)):
        average = contribution[mask].mean(axis=(0, 1))
        positive = np.maximum(average, 0).sum()
        windows[title] = dict(
            frames=int(mask.sum()),
            weighted_reward_mean_per_env_step=dict(zip(names, average.tolist(), strict=True)),
            share_of_positive_mean_reward={n: float(max(0, v) / positive) if positive else 0.0 for n, v in zip(names, average, strict=True)},
        )
    return dict(
        schema="v2d.saved-policy-reward-balance.v1",
        source_report_sha256=digest(run / "report.json"),
        state_sha256=digest(run / "native-state-readback.npz"),
        checkpoint_sha256=digest(checkpoint),
        analysis_source_sha256=digest(Path(__file__)),
        selected_time_window_s=[start_s, end_s],
        averages_all_worlds_and_active_hands=True,
        saved_rewards_reconstructed=True,
        reward_balance_is_not_physical_acceptance_or_causal_proof=True,
        windows=windows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.checkpoint)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["windows"], indent=2))

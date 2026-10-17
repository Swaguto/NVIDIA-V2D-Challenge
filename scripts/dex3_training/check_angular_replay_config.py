"""Check resolved replay contracts before allocating any simulation worlds."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import yaml


def resolved(experiment):
    text = subprocess.check_output([
        sys.executable, str(Path(__file__).with_name("fit_phase_policy.py")),
        "--config-dir", "/opt/dex3-extension/configs", "experiment=" + experiment,
        "action.wrist_payload_gain=0.5", "--cfg", "job", "--resolve",
    ], text=True)
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("task:"))
    return yaml.safe_load("\n".join(lines[start:]))


if __name__ == "__main__":
    baseline = resolved("dex3_repaired_payload")
    candidate = resolved("dex3_repaired_angular")
    action = deepcopy(candidate["action"])
    for key in ("wrist_angular_reference_velocity", "wrist_angular_kd",
                "maximum_reference_wrist_angular_speed_rad_s"):
        action.pop(key)
    assert action == baseline["action"], "Replay action semantics changed"
    assert candidate["scene"] == baseline["scene"]
    assert candidate["task"] == baseline["task"]
    assert candidate["sim"] == baseline["sim"]
    embodiment = deepcopy(candidate["embodiment"])
    ball = embodiment["config"]["joint_overrides"]["wrist_rotvec$"]
    assert ball["kd"] == 0
    ball["kd"] = baseline["embodiment"]["config"]["joint_overrides"]["wrist_rotvec$"]["kd"]
    assert embodiment == baseline["embodiment"]
    result = dict(raw_action_contract_unchanged=True, geometry_and_source_clock_unchanged=True,
                  angular_reference_enabled=candidate["action"]["wrist_angular_reference_velocity"],
                  native_passive_wrist_damping_zero=True, physical_task_accepted=False)
    Path("/outputs/config-preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)

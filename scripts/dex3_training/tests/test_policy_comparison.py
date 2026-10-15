import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import pytest

spec = importlib.util.spec_from_file_location(
    "comparison", Path(__file__).parents[1] / "compare_policies.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path

    blob = tmp_path / "hull.npz"
    blob.write_bytes(b"frozen geometry")
    manifest = write(
        "manifest.json",
        dict(
            collision_cache={"files": ["hull.npz"]},
            files={"hull.npz": module.digest(blob)},
            selected_frame_count=75,
        ),
    )
    sha = module.digest(manifest)
    controls = write(
        "controls.json",
        dict(
            input_manifest_sha256=sha,
            collision_cache_locked=True,
            installed_dispatch_loads_dual_hand=True,
            unknown_contact_reward_rejected=True,
            implicit_resampling_rejected=True,
            changed_manifest_rejected=True,
        ),
    )
    config = dict(
        env={"voc_scale": 0},
        voc_scale_override=0,
        curriculum=None,
        reset={"voc_decay_steps": 0},
        objective={
            name: {"enabled": False, "weight": 0}
            for name in (
                "contact_wrench_support",
                "missed_contact",
                "unintended_contact",
                "force_closure",
            )
        },
    )
    checkpoints, reports = [], []
    for iteration in [0, 1000]:
        meta = {
            "user.resolved_config_json": json.dumps(config),
            "user.dex3_runtime_patches_sha256": "patch",
            "user.dex3_image_context_sha256": "image",
            "user.policy_schema_json": "schema",
            "user.dex3_input_manifest_sha256": sha,
            "user.environment_steps": str(iteration * 1536),
            "user.iteration": str(iteration),
        }
        header = json.dumps({"__metadata__": meta}).encode()
        checkpoint = tmp_path / f"model_{iteration}.safetensors"
        checkpoint.write_bytes(struct.pack("<Q", len(header)) + header)
        checkpoints.append(checkpoint)
        reports.append(
            write(
                f"report_{iteration}.json",
                dict(
                    checkpoint_sha256=module.digest(checkpoint),
                    environment_steps=iteration * 1536,
                    step_count=75,
                    reset_mode="explicit",
                    settling_steps=0,
                    start_frame=0,
                    sim_control_fps=20,
                    non_finite_worlds=[],
                    world_count=4,
                    reference_tracking={"success_rate": 0.0},
                    metrics=dict(
                        chord_sr=0.0,
                        mppe_cm=19,
                        mean_add_m=0.1,
                        object_position_error_per_body_m=[0.01, 0.18],
                    ),
                ),
            )
        )
    return checkpoints, reports, controls


def test_failed_learning_stays_failed_and_geometry_tampering_is_rejected(tmp_path):
    checkpoints, reports, controls = fixture(tmp_path)
    result = module.compare(tmp_path, checkpoints, reports, controls)
    assert result["controlled_comparison"]
    assert not result["primary_success_improved"]
    assert not result["physical_task_accepted"]
    (tmp_path / "hull.npz").write_bytes(b"different decomposition")
    with pytest.raises(ValueError, match="Private input bytes differ"):
        module.compare(tmp_path, checkpoints, reports, controls)


def test_report_cannot_claim_another_checkpoint(tmp_path):
    checkpoints, reports, controls = fixture(tmp_path)
    report = json.loads(reports[1].read_text())
    report["checkpoint_sha256"] = hashlib.sha256(b"other").hexdigest()
    reports[1].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Evaluation receipt differs"):
        module.compare(tmp_path, checkpoints, reports, controls)

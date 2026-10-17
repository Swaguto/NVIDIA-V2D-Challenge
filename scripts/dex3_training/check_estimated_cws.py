"""GPU gate for estimated-reference CWS, with independent CPU support values."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
import warp as wp
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.objectives.contact import contact_wrench_support_objective
from flash_chord.objectives.wrench import sample_wrench_basis
from check_training_curriculum import build
from v2d_dex3_flashchord.phase_prior import model_signature


def cpu_support(points, normals, basis, radius, mu):
    if not len(points):
        return np.zeros(len(basis))
    n = normals.astype(float)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    sign = np.where(n[:, 2] >= 0, 1, -1)
    a = -1 / (sign + n[:, 2])
    b = n[:, 0] * n[:, 1] * a
    t1 = np.stack([1 + sign * n[:, 0] ** 2 * a, sign * b, -sign * n[:, 0]], axis=-1)
    t2 = np.stack([b, sign + n[:, 1] ** 2 * a, -n[:, 1]], axis=-1)
    t1 /= np.linalg.norm(t1, axis=-1, keepdims=True)
    t2 /= np.linalg.norm(t2, axis=-1, keepdims=True)
    theta = np.arange(8) * 2 * np.pi / 8
    rays = n[:, None] + mu * (np.cos(theta)[None, :, None] * t1[:, None] + np.sin(theta)[None, :, None] * t2[:, None])
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    rays = np.concatenate([rays, n[:, None]], axis=1)
    moment = np.cross(points[:, None], rays) / radius
    wrenches = np.concatenate([rays, moment], axis=-1)
    return np.maximum(0, np.einsum("crj,bj->crb", wrenches, basis).max(axis=(0, 1)))


@hydra.main(version_base="1.3", config_path="/opt/flash-chord/src/flash_chord/configs", config_name="train")
def main(cfg):
    manifest_path = Path(cfg.task.parquet).parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert cfg.objective.estimated_contact_guidance and not manifest["contacts_known"]
    OmegaConf.update(cfg, "embodiment.config.reference_manifest_sha256", hashlib.sha256(manifest_path.read_bytes()).hexdigest(), force_add=True)
    with wp.ScopedDevice("cuda:0"):
        env, setup = build(cfg)
        objective = env.objective.base
        assert objective.num_basis == 512 and objective.num_edges == 8
        ref = objective.ref_wrench_support.numpy().reshape(objective.num_frames, objective.num_hands, objective.num_bodies, objective.num_basis)
        root = manifest_path.parent
        with np.load(root / manifest["estimated_contact_reference"]["data"]) as saved:
            points, normals, parts, valid = [saved[k].copy() for k in ["points_object_m", "normal_force_directions_object", "part_ids", "source_geometry_valid"]]
        errors = []
        for frame in (0, 60, 100, 140, 200, 300):
            for hand, side in enumerate(objective.config.sides):
                source_hand = ("left", "right").index(side)
                for body in range(objective.num_bodies):
                    selected = parts[frame, source_hand] == body + 1
                    basis = sample_wrench_basis(512, 1.0, objective.config.wrench_basis_seed + body)
                    expected = cpu_support(points[frame, source_hand, selected], normals[frame, source_hand, selected], basis, setup.reference.object_mesh_radius()[body], objective.config.friction_coefficient)
                    errors.append(float(np.max(np.abs(expected - ref[frame, hand, body]))))
                    if not valid[frame, source_hand]:
                        assert not ref[frame, hand, body].any()
        assert max(errors) < 1e-5, errors
        assert ref[60].max() > 0
        # The real upstream objective must give zero to an absent robot contact
        # set and one to its identical, active reference support set.
        target = wp.array(np.tile(ref[60].ravel(), 4), dtype=wp.float32)
        zero = wp.zeros_like(target)
        output = wp.zeros(4, dtype=wp.float32)
        args = [4, objective.num_hands, objective.num_bodies, 512]
        for current, expected in ((zero, 0.0), (target, 1.0)):
            wp.launch(contact_wrench_support_objective, dim=4, inputs=[target, current, *args[1:], objective.config.contact_wrench_support.tolerance, objective.config.contact_wrench_support.var], outputs=[output])
            np.testing.assert_allclose(output.numpy(), expected, atol=1e-6)
        try:
            replace(env.config.objective, estimated_contact_guidance=False)
        except ValueError:
            pass
        else:
            raise AssertionError("Default unknown-contact guard was bypassed")
        bad_reference = replace(setup.reference, _metadata=replace(setup.reference.metadata, schema_version="dex3_dual_hand_v1", source_dataset="local_development"))
        try:
            env.config.objective.build(None, None, None, bad_reference, None, None, None, 4, 0.05)
        except ValueError:
            pass
        else:
            raise AssertionError("Unbound reference enabled estimated CWS")
        evaluated = OmegaConf.create(apply_evaluation_composition(OmegaConf.to_container(cfg, resolve=True), world_count=4))
        evaluation, _ = build(evaluated)
        evaluation.reset(frame_id=0)
        evaluation.capture_step()
        evaluation.reset(frame_id=0)
        for _ in range(4):
            evaluation.step(wp.zeros_like(evaluation.action_input))
            assert np.all(evaluation.object_control_scale.numpy() == 0)
            assert np.isfinite(evaluation.state_0.body_q.numpy()).all()
            assert np.isfinite(evaluation.objective.score.numpy()).all()
        result = dict(schema="v2d.estimated-cws-gpu-contract.v1", input_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), native_physics_model_sha256=model_signature(setup.scene.model, 4), independent_cpu_reference_support_max_absolute_error=max(errors), unknown_frames_have_zero_reference_support=True, absent_robot_contacts_reward_zero=True, matching_active_support_reward_one=True, default_unknown_contact_guard_preserved=True, unbound_reference_rejected=True, captured_unassisted_evaluation_steps=4, policy_observation_dim=evaluation.observation_dim, estimated_not_measured_contact_reference=True, physical_task_accepted=False)
        Path("/outputs/estimated-cws-gpu-contract.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

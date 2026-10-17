"""Compare captured real-scene wrench readout with independent CPU decoding."""

from dataclasses import replace
from pathlib import Path
import hashlib
import json

import hydra
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import warp as wp
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene


@hydra.main(
    version_base="1.3",
    config_path="/opt/flash-chord/src/flash_chord/configs",
    config_name="train",
)
def main(cfg):
    with wp.ScopedDevice("cuda:0"):
        manifest = Path(cfg.task.parquet).parent / "manifest.json"
        OmegaConf.update(
            cfg,
            "embodiment.config.reference_manifest_sha256",
            hashlib.sha256(manifest.read_bytes()).hexdigest(),
            force_add=True,
        )
        config = replace(instantiate_typed(cfg.env, RLEnvConfig), auto_reset=False)
        embodiment = instantiate_typed(cfg.embodiment, Embodiment)
        setup = setup_scene(
            parquet=cfg.task.parquet,
            control_fps=config.sim.fps,
            motion_speed=cfg.task.motion_speed,
            embodiment=embodiment,
            collision=instantiate_typed(cfg.collision, CollisionPolicy),
            world_count=2,
            include_support=cfg.scene.support,
            decompose_objects=cfg.scene.decompose_objects,
            contact_friction=cfg.scene.contact_friction,
        )
        env = RLEnv(setup.scene, setup.reference, config=config)
        assert env.objective.measured_wrench is not None
        env.reset(frame_id=140)
        env.capture_step()
        env.reset(frame_id=140)
        labels = env.model.body_label
        spec = json.loads((manifest.parent / "task-grips.json").read_text())
        objects = json.loads(manifest.read_text())["objects"]
        active_pairs = {
            (side, objects[phases[0]["object"]]["name"] + "/object")
            for side, phases in spec["phases"].items()
        }
        mapped = env.model.shape_body.numpy()
        bodies = env.model.body_count // 2
        contacts_seen = 0
        largest_error = 0.0
        largest_com_time_offset = 0.0
        for _ in range(6):
            env.step(wp.zeros_like(env.action_input))
            pose = env.state_0.body_q.numpy()
            reconstructed_com = pose[:, :3] + Rotation.from_quat(pose[:, 3:]).apply(
                env.model.body_com.numpy()
            )
            # Use native inertial-frame positions as an independent COM source.
            native_com = env.solver.mjw_data.xipos.numpy()
            body_map = env.solver.mjc_body_to_newton.numpy()
            com = np.full_like(reconstructed_com, np.nan)
            for world in range(2):
                for native_body, newton_body in enumerate(body_map[world]):
                    if newton_body >= 0:
                        com[newton_body] = native_com[world, native_body]
            # Native contacts/COM precede integration; returned Newton poses
            # follow integration. This displacement is diagnostic, not an
            # equality gate between different times.
            valid_com = np.isfinite(com).all(axis=1)
            largest_com_time_offset = max(
                largest_com_time_offset,
                float(np.max(np.linalg.norm(
                    com[valid_com] - reconstructed_com[valid_com], axis=1
                ))),
            )
            c = env.contacts
            count = int(c.rigid_contact_count.numpy()[0])
            a, b = (np.full(count, -1, dtype=int) for _ in range(2))
            for shape, body in [
                (c.rigid_contact_shape0.numpy()[:count], a),
                (c.rigid_contact_shape1.numpy()[:count], b),
            ]:
                valid = (shape >= 0) & (shape < len(mapped))
                body[valid] = mapped[shape[valid]]
            raw = c.force.numpy()[:count]
            # Independent force application point from MuJoCo itself, not a
            # reconstruction using the GPU helper's Newton surface points.
            contact_position = env.solver.mjw_data.contact.pos.numpy()[:count]
            expected_force = np.zeros((4, 3))
            expected_moment = np.zeros((4, 3))
            for i in range(count):
                for obj, hand, sign in [
                    (a[i], b[i], 1),
                    (b[i], a[i], -1),
                ]:
                    if obj < 0 or hand < 0:
                        continue
                    for hi, side in enumerate(env.contact.layout.sides):
                        if not labels[hand].startswith("dex3_" + side + "/"):
                            continue
                        # Compare labels within each replicated world rather
                        # than reusing the GPU's shape-hand/object mapping.
                        local_name = labels[obj % bodies]
                        if (side, local_name) not in active_pairs:
                            continue
                        world = obj // bodies
                        position = contact_position[i]
                        force = sign * raw[i, :3]
                        expected_force[world * 2 + hi] += force
                        expected_moment[world * 2 + hi] += (
                            np.cross(position - com[obj], force) + sign * raw[i, 3:]
                        )
                        contacts_seen += 1
            actual = env.objective.measured_wrench
            # Run the readout once more and prove its native input buffers did
            # not change. It must only clear/write its own resultant arrays.
            native_buffers = [
                env.solver.mjw_data.contact.pos, env.solver.mjw_data.xipos,
                env.solver.mjw_data.qpos, env.solver.mjw_data.qvel,
                env.solver.mjw_data.efc.force, c.force,
            ]
            before = [buffer.numpy() for buffer in native_buffers]
            actual.compute(env.state_0, c)
            for buffer, saved in zip(native_buffers, before):
                np.testing.assert_array_equal(buffer.numpy(), saved)
            for observed, expected in [
                (actual.force.numpy(), expected_force),
                (actual.moment.numpy(), expected_moment),
            ]:
                np.testing.assert_allclose(observed, expected, atol=2e-4, rtol=2e-5)
                largest_error = max(
                    largest_error, float(np.max(np.abs(observed - expected)))
                )
        assert contacts_seen > 0, "No real hand/object contacts exercised the mapping"
        print(
            json.dumps(
                dict(
                    actual_scene_captured_steps=6,
                    actual_assigned_contacts_decoded=contacts_seen,
                    maximum_absolute_force_or_moment_difference=largest_error,
                    independent_CPU_decode_matches_GPU_wrench=True,
                    independent_native_contact_midpoint_and_inertial_COM_used=True,
                    native_input_buffers_unchanged_by_readout=True,
                    maximum_force_time_to_returned_state_COM_displacement_m=largest_com_time_offset,
                    support_and_other_object_contacts_excluded=True,
                    evaluation_auto_reset=False,
                    physics_modified=False,
                )
            )
        )


if __name__ == "__main__":
    main()

"""Compile frozen collision geometry with native MuJoCo on CPU before PPO.

This catches invalid mesh fragments without launching a GPU learner. It does
not step the simulator or certify contacts, grasping, or policy performance.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path


def compile_scene(inputs):
    import newton
    import warp as wp
    from flash_chord.embodiments.base import JointControlSpec
    from flash_chord.scene.builder import build_scene
    from flash_chord.scene.collision import CollisionPolicy
    from v2d_dex3_flashchord import RUNTIME_COMMIT
    from v2d_dex3_flashchord.embodiment import Dex3Hands, Dex3HandsConfig
    from v2d_dex3_flashchord.reference import load_dex3
    from v2d_dex3_flashchord.sim import NativeContactSimConfig

    inputs = inputs.resolve()
    manifest_sha = hashlib.sha256((inputs / "manifest.json").read_bytes()).hexdigest()
    reference = load_dex3(inputs / "reference.parquet", control_fps=20)
    embodiment = Dex3Hands(
        Dex3HandsConfig(
            left_urdf_path=inputs / "robot/urdfs/g1/dex3_left.urdf",
            right_urdf_path=inputs / "robot/urdfs/g1/dex3_right.urdf",
            reference_manifest_sha256=manifest_sha,
            default_joint=JointControlSpec(
                kp=8, kd=0.2, armature=0.00149, effort_limit=0.76, velocity_limit=23
            ),
        )
    )
    cfg = NativeContactSimConfig(
        substeps=20, nconmax=512, njmax=2048, enable_multiccd=False
    )
    with wp.ScopedDevice("cpu"):
        scene = build_scene(
            embodiment,
            reference,
            world_count=1,
            support_usda=inputs / "support.usda",
            collision=CollisionPolicy(ground=False),
            decompose_objects=True,
            contact_friction=1.0,
        )
        # Native CPU and GPU backends share the same MuJoCo model compiler.
        # Keep solid-mesh inertia checks enabled; no shell-inertia workaround.
        solver = newton.solvers.SolverMuJoCo(
            scene.model,
            use_mujoco_cpu=True,
            solver=cfg.solver,
            integrator=cfg.integrator,
            njmax=cfg.njmax,
            nconmax=cfg.nconmax,
            impratio=cfg.impratio,
            cone=cfg.cone,
            iterations=cfg.iterations,
            ls_iterations=cfg.ls_iterations,
            use_mujoco_contacts=cfg.use_mujoco_contacts,
            enable_multiccd=cfg.enable_multiccd,
        )
    return scene, reference, solver, manifest_sha, RUNTIME_COMMIT


def check(inputs):
    start = time.monotonic()
    _, _, solver, manifest_sha, runtime_commit = compile_scene(inputs)
    return {
        "schema": "v2d.scored-native-geometry-preflight.v1",
        "status": "passed",
        "input_manifest_sha256": manifest_sha,
        "runtime_commit": runtime_commit,
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_geom_count": int(solver.mj_model.ngeom),
        "native_mesh_count": int(solver.mj_model.nmesh),
        "wall_seconds": time.monotonic() - start,
        "native_mesh_checks_relaxed": False,
        "physics_stepped": False,
        "physical_task_accepted": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = check(a.inputs)
    except Exception as error:
        a.out.write_text(
            json.dumps(
                {
                    "schema": "v2d.scored-native-geometry-preflight.v1",
                    "status": "failed",
                    "error": str(error),
                    "input_manifest_sha256": hashlib.sha256(
                        (a.inputs / "manifest.json").read_bytes()
                    ).hexdigest()
                    if (a.inputs / "manifest.json").is_file()
                    else None,
                    "producer_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "physical_task_accepted": False,
                    "native_mesh_checks_relaxed": False,
                },
                indent=2,
            )
            + "\n"
        )
        raise
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

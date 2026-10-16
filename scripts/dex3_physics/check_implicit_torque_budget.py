"""Measure an isolated fixed-base hinge's motor torque from inertial acceleration.

No source episode or object is loaded. Zero gravity, zero friction, no collision,
constant scalar inertia and zero initial velocity make I*delta_velocity/dt the
average net actuator torque. This checks the pinned backend's drive+feedforward
limit semantics; ImplicitActuator.applied_effort itself is only an estimate.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--encode-in-drive", action="store_true")
    parser.add_argument(
        "--contact-solver", choices=["default", "grip"], default="default"
    )
    args = parser.parse_args()
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, device="cpu", enable_cameras=False)
    try:
        import numpy as np
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproduction"))
        from _dex3_wrist_control import encode_finger_feedforward
        from _dex3_grip_control import contact_solver_options
        from isaaclab_physx.physics import PhysxCfg

        dt = 0.001
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=dt,
                device="cpu",
                use_fabric=False,
                gravity=(0, 0, 0),
                physics=PhysxCfg(**contact_solver_options(args.contact_solver)),
            )
        )
        stage = sim.stage
        UsdGeom.Xform.Define(stage, "/World/TorqueProbe")
        for name in ["root", "rotor"]:
            cube = UsdGeom.Cube.Define(stage, "/World/TorqueProbe/" + name)
            cube.CreateSizeAttr(0.1)
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
            mass = UsdPhysics.MassAPI.Apply(cube.GetPrim())
            mass.CreateMassAttr(0.1)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.001, 0.001, 0.001))
            mass.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
            body = PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim())
            body.CreateDisableGravityAttr(True)
        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/TorqueProbe/fixed")
        fixed.CreateBody1Rel().SetTargets([Sdf.Path("/World/TorqueProbe/root")])
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/TorqueProbe/hinge")
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/TorqueProbe/root")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/TorqueProbe/rotor")])
        joint.CreateAxisAttr("Z")
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(8.0)
        drive.CreateDampingAttr(0.0)
        drive.CreateMaxForceAttr(0.76)
        drive.CreateTargetPositionAttr(0.0)
        UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/TorqueProbe",
                spawn=None,
                actuators={
                    "hinge": ImplicitActuatorCfg(
                        joint_names_expr=["hinge"],
                        effort_limit_sim=0.76,
                        velocity_limit_sim=10.0,
                        stiffness=8.0,
                        damping=0.0,
                    )
                },
            )
        )
        sim.reset()
        view = robot.root_physx_view
        if not robot.is_fixed_base or robot.num_joints != 1:
            raise RuntimeError("Expected a fixed-base one-DOF torque probe")
        zero = torch.zeros((1, 1), dtype=torch.float32)
        robot.write_joint_state_to_sim(zero, zero)
        mass = view.get_generalized_mass_matrices().numpy().copy()
        if mass.shape != (1, 1, 1) or not np.isclose(mass[0, 0, 0], 0.001, rtol=1e-4):
            raise RuntimeError(f"Unexpected probe inertia {mass.tolist()}")
        rows = []
        cases = [(1.0, 0.0), (1.0, 0.55), (1.0, -0.55)]
        if args.encode_in_drive:
            cases += [(-1.0, -0.55), (0.02, 0.0), (0.02, 0.55), (0.02, -0.55)]
        for reference, effort in cases:
            robot.write_joint_state_to_sim(zero, zero)
            robot.set_joint_position_target(zero)
            robot.set_joint_effort_target(zero)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            robot.write_joint_state_to_sim(zero, zero)
            target = (
                float(encode_finger_feedforward([reference], [effort], [8.0])[0])
                if args.encode_in_drive
                else reference
            )
            external = 0.0 if args.encode_in_drive else effort
            robot.set_joint_position_target(
                torch.tensor([[target]], dtype=torch.float32)
            )
            robot.set_joint_effort_target(
                torch.tensor([[external]], dtype=torch.float32)
            )
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            velocity = float(robot.data.joint_vel[0, 0].cpu().numpy())
            rows.append(
                dict(
                    posture_reference_rad=reference,
                    feedforward_Nm=effort,
                    separate_joint_effort_target_Nm=external,
                    implicit_drive_position_target_rad=target,
                    measured_average_net_motor_torque_Nm=float(
                        mass[0, 0, 0] * velocity / dt
                    ),
                    velocity_rad_s=velocity,
                )
            )
        # Calibrate the unsaturated drive without feedforward, then check the
        # signed feedforward scaling against that measured response. PhysX TGS
        # subiterations need not equal one full-dt implicit Euler update.
        equivalence = None
        calibrated_gain = None
        if args.encode_in_drive:
            baseline = rows[-3]
            calibrated_gain = baseline["measured_average_net_motor_torque_Nm"] / (
                8.0 * baseline["posture_reference_rad"]
            )
            # Zero damping/inertia-only response must lie between the full-step
            # implicit response and the initial continuous spring torque.
            lower_gain = 1.0 / (1.0 + 8.0 * dt**2 / float(mass[0, 0, 0]))
            equivalence = lower_gain <= calibrated_gain <= 1.0001 and all(
                np.isclose(
                    row["measured_average_net_motor_torque_Nm"],
                    calibrated_gain
                    * (8.0 * row["posture_reference_rad"] + row["feedforward_Nm"]),
                    atol=0.0002,
                    rtol=0,
                )
                for row in rows[-2:]
            )
        result = dict(
            schema="v2d.implicit-torque-budget-probe.v1",
            dt_s=dt,
            inertia_kg_m2=float(mass[0, 0, 0]),
            drive_max_force_readback_Nm=view.get_dof_max_forces().numpy().tolist(),
            cases=rows,
            combined_0_76Nm_limit_observed=all(
                abs(r["measured_average_net_motor_torque_Nm"]) <= 0.7601 for r in rows
            )
            and bool(
                np.isclose(
                    rows[0]["measured_average_net_motor_torque_Nm"], 0.76, atol=0.002
                )
            ),
            drive_only_calibration_pass=bool(
                np.isclose(
                    rows[0]["measured_average_net_motor_torque_Nm"], 0.76, atol=0.002
                )
            ),
            source_episode_loaded=False,
            contact_solver=args.contact_solver,
            implicit_actuator_estimate_is_direct_measurement=False,
            unsaturated_control_law_verified=None
            if equivalence is None
            else bool(equivalence),
            unsaturated_drive_only_measured_gain=calibrated_gain,
            unsaturated_verification_scope="Drive-only inertial calibration and signed feedforward scaling; no assumption of single-step TGS integration",
            feedforward_encoding="drive_target_bias"
            if args.encode_in_drive
            else "separate_joint_effort",
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        sim.stop()
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()

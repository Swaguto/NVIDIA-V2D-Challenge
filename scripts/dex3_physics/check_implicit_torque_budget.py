"""Measure an isolated fixed-base hinge's motor torque from inertial acceleration.

No source episode or object is loaded. Zero gravity, zero friction, no collision,
constant scalar inertia and zero initial velocity make I*delta_velocity/dt the
average net actuator torque. This checks the pinned backend's drive+feedforward
limit semantics; ImplicitActuator.applied_effort itself is only an estimate.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
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

        dt = 0.001
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=dt, device="cpu", use_fabric=False, gravity=(0, 0, 0)
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
        for effort in [0.0, 0.55, -0.55]:
            robot.write_joint_state_to_sim(zero, zero)
            robot.set_joint_position_target(zero)
            robot.set_joint_effort_target(zero)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            robot.write_joint_state_to_sim(zero, zero)
            robot.set_joint_position_target(torch.ones((1, 1), dtype=torch.float32))
            robot.set_joint_effort_target(torch.tensor([[effort]], dtype=torch.float32))
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            velocity = float(robot.data.joint_vel[0, 0].cpu().numpy())
            rows.append(
                dict(
                    feedforward_Nm=effort,
                    measured_average_net_motor_torque_Nm=float(
                        mass[0, 0, 0] * velocity / dt
                    ),
                    velocity_rad_s=velocity,
                )
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
            implicit_actuator_estimate_is_direct_measurement=False,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        sim.stop()
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()

"""Opt-in native contact manifolds; all other upstream solver settings are kept."""

from dataclasses import dataclass
import newton
from flash_chord.runtime.sim import SimConfig
from .native_contact_binding import NativeContactReadout
from .phase_prior import simulator_signature


@dataclass
class NativeContactSimConfig(SimConfig):
    enable_multiccd: bool = False

    def __post_init__(self):
        if not isinstance(self.enable_multiccd, bool):
            raise ValueError("enable_multiccd must be an explicit boolean")


def build_solver(model, cfg):
    model.dex3_simulator_contract_sha256 = simulator_signature(cfg)
    solver = newton.solvers.SolverMuJoCo(
        model,
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
    # Contacts and xipos are both from the force solve, before integration.
    # Returned Newton body poses have already been integrated and cannot be
    # used to reconstruct the COM for that earlier contact wrench.
    model.dex3_native_contact_readout = NativeContactReadout(
        solver.mjw_data, solver.mjc_body_to_newton
    )
    return solver

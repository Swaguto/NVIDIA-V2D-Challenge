"""Opt-in native contact manifolds; all other upstream solver settings are kept."""

from dataclasses import dataclass
import newton
from flash_chord.runtime.sim import SimConfig


@dataclass
class NativeContactSimConfig(SimConfig):
    enable_multiccd: bool = False

    def __post_init__(self):
        if not isinstance(self.enable_multiccd, bool):
            raise ValueError("enable_multiccd must be an explicit boolean")


def build_solver(model, cfg):
    return newton.solvers.SolverMuJoCo(
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

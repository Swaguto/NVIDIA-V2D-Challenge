"""Necessary load-balance check for assumed active contact geometry.

The independent component friction box is an OUTER approximation: it permits
forces outside the native elliptic or pyramidal cone. Normal force is unlimited.
Infeasibility rejects only these frozen contacts, friction and required wrench.
Feasibility is not a certificate of contact activation, force closure, motor
authority, dynamics, source accuracy or task success. Returned forces are solved,
never measured. Use ``wrench_capacity.capacity`` for measured force budgets.

All spatial quantities use one world frame. Wrenches are [force N, moment Nm]
about the supplied COM. Friction order follows MuJoCo's contact convention:
two sliding coefficients (dimensionless), spin and two rolling coefficients (m).
This module has no simulator dependency and does not call forward dynamics.
"""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import numpy as np
from scipy.optimize import linprog


def _array(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class ContactGeometry:
    """An assumed active contact, with basis rows acting ON the object.

    The first row points from the hand into the object; the next two are
    tangents. An orthogonal basis of either handedness is allowed because
    reversing all native contact axes for geom1 reverses its handedness.
    Dimensions 1/3/4/6 allow normal / sliding / spin / rolling components.
    Supplying this geometry is not evidence of active contact or measured force.
    """

    point_world_m: np.ndarray
    frame_into_object_world: np.ndarray
    dimension: int
    friction: np.ndarray

    def __post_init__(self):
        if (
            isinstance(self.dimension, (bool, np.bool_))
            or not isinstance(self.dimension, Integral)
            or self.dimension not in (1, 3, 4, 6)
        ):
            raise ValueError("Contact dimension must be integer 1, 3, 4 or 6")
        point = _array(self.point_world_m, (3,), "contact point")
        frame = _array(self.frame_into_object_world, (3, 3), "contact frame")
        friction = _array(self.friction, (5,), "contact friction")
        if not np.allclose(frame @ frame.T, np.eye(3), atol=1e-7, rtol=0):
            raise ValueError("Contact frame rows must be orthonormal")
        if (friction < 0).any():
            raise ValueError("Contact friction must be nonnegative")
        object.__setattr__(self, "point_world_m", point)
        object.__setattr__(self, "frame_into_object_world", frame)
        object.__setattr__(self, "friction", friction)

    @classmethod
    def from_mujoco_geometry(
        cls, *, point_world_m, frame_geom1_to_geom2_world, dimension, friction,
        object_is_geom2,
    ):
        """Convert the native contact basis, without reading or solving forces.

        MuJoCo mjContact.pos is the midpoint and frame[0:3] points from geom1
        to geom2. The caller must bind the actual object geom before conversion.
        This sign conversion is valid for the symmetric component bounds here.
        """
        if not isinstance(object_is_geom2, (bool, np.bool_)):
            raise ValueError("object_is_geom2 must be an explicit boolean")
        frame = _array(frame_geom1_to_geom2_world, (3, 3), "native contact frame")
        return cls(point_world_m, frame if object_is_geom2 else -frame, dimension, friction)


def outer_contact_load_feasibility(
    contacts: Sequence[ContactGeometry], com_world_m, required_wrench_world,
    *, moment_length_m=0.1,
):
    """Balance a required contact wrench using an optimistic friction box.

    For static support, the required contact force is minus gravity force, not
    gravity itself. All moments must be about ``com_world_m``. At dimension 4,
    spin is about the normal; at dimension 6, rolling is about the two tangents.
    The LP minimizes total normal force without imposing a force or motor cap.
    ``moment_length_m`` conditions moment equations only; it is not a lever arm
    or a feasibility threshold. Numerical/solver failures raise RuntimeError;
    they must not be interpreted as feasible or as proven infeasible.
    """
    com = _array(com_world_m, (3,), "COM")
    load = _array(required_wrench_world, (6,), "required wrench")
    if (
        isinstance(moment_length_m, (bool, np.bool_))
        or not isinstance(moment_length_m, Real)
        or not np.isfinite(moment_length_m)
        or moment_length_m <= 0
        or not np.isfinite(1 / moment_length_m)
    ):
        raise ValueError("moment_length_m must be finite and positive")
    contacts = tuple(contacts)
    if not all(isinstance(c, ContactGeometry) for c in contacts):
        raise ValueError("contacts must contain validated ContactGeometry records")
    result = dict(
        friction_model="independent_component_outer_box",
        contacts=len(contacts),
        contact_geometry_assumed_active=True,
        force_origin="solved_not_measured",
        normal_force_limits_applied=False,
        motor_limits_applied=False,
        certifies_force_closure_or_dynamic_or_task_success=False,
    )
    if not contacts:
        feasible = bool(np.all(load == 0))
        return dict(
            result, feasible_in_outer_model=feasible,
            status="zero_load_without_contacts" if feasible else "no_contacts_for_load",
            solved_contact_components=[] if feasible else None,
            minimum_total_normal_force_N=0.0 if feasible else None,
        )

    count = sum(c.dimension for c in contacts)
    wrench = np.zeros((6, count))
    normal_indices, inequalities, bounds = [], [], []
    offset = 0
    for contact in contacts:
        dim, basis = contact.dimension, contact.frame_into_object_world
        normal_indices.append(offset)
        force_count = min(dim, 3)
        wrench[:3, offset:offset + force_count] = basis[:force_count].T
        wrench[3:, offset:offset + force_count] = np.cross(
            contact.point_world_m - com, basis[:force_count]
        ).T
        if dim >= 4:
            wrench[3:, offset + 3] = basis[0]
        if dim == 6:
            wrench[3:, offset + 4:offset + 6] = basis[1:].T
        bounds.append((0, None))
        for component in range(1, dim):
            bounds.append((None, None))
            for sign in (-1.0, 1.0):
                row = np.zeros(count)
                row[offset + component] = sign
                row[offset] = -contact.friction[component - 1]
                inequalities.append(row)
        offset += dim

    objective = np.zeros(count)
    objective[normal_indices] = 1
    scale = np.r_[np.ones(3), np.full(3, 1 / moment_length_m)]
    inequality = np.array(inequalities) if inequalities else None
    solved = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.zeros(len(inequalities)) if inequalities else None,
        A_eq=wrench * scale[:, None], b_eq=load * scale,
        bounds=bounds, method="highs",
    )
    if solved.status == 2:
        return dict(
            result, feasible_in_outer_model=False, status="infeasible",
            solved_contact_components=None, minimum_total_normal_force_N=None,
        )
    if not solved.success or solved.status != 0:
        raise RuntimeError(f"Contact load solve unresolved: {solved.message}")
    x = np.asarray(solved.x)
    if x.shape != (count,) or not np.isfinite(x).all():
        raise RuntimeError("Contact load solve returned invalid components")
    residual = wrench @ x - load
    # Check forces and moments independently in their stated SI units. Solver
    # success alone is insufficient, especially for an ill-conditioned load.
    tolerance = 1e-7 + 1e-9 * np.abs(load)
    components, offset = [], 0
    invalid = np.any(np.abs(residual) > tolerance)
    for contact in contacts:
        part = x[offset:offset + contact.dimension]
        invalid |= part[0] < -1e-7
        invalid |= np.any(np.abs(part[1:]) - contact.friction[:contact.dimension - 1] * part[0] > 1e-7)
        components.append(part.tolist())
        offset += contact.dimension
    if invalid:
        raise RuntimeError("Contact load solve failed independent equilibrium/friction checks")
    return dict(
        result, feasible_in_outer_model=True, status="optimal",
        solved_contact_components=components,
        component_order="normal_N, sliding1_N, sliding2_N, spin_Nm, rolling1_Nm, rolling2_Nm (truncated by dimension)",
        minimum_total_normal_force_N=float(x[normal_indices].sum()),
        wrench_residual_world=residual.tolist(),
    )

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
    return _solve_outer_load(
        contacts, com_world_m, required_wrench_world, moment_length_m=moment_length_m,
    )


def contact_effort_map(contact, linear_jacobian_m_per_rad, angular_jacobian):
    """Map forces ON the object to counteracting revolute actuator effort.

    Both Jacobians must describe the same actual hand contact point and pose,
    in world axes, with columns in one declared revolute-joint order. The hand
    receives the opposite wrench, so its holding actuator effort is +J^T times
    the object wrench. This does not verify contact activation or pose binding.
    """
    if not isinstance(contact, ContactGeometry):
        raise ValueError("Need validated contact geometry")
    linear = np.asarray(linear_jacobian_m_per_rad)
    if linear.ndim != 2 or linear.shape[0] != 3 or not linear.shape[1]:
        raise ValueError("Need linear Jacobian[3, revolute joints]")
    linear = _array(linear, linear.shape, "linear Jacobian")
    angular = _array(angular_jacobian, linear.shape, "angular Jacobian")
    basis, dim = contact.frame_into_object_world, contact.dimension
    result = np.zeros((linear.shape[1], dim))
    nf = min(dim, 3)
    result[:, :nf] = linear.T @ basis[:nf].T
    if dim >= 4:
        result[:, 3] = angular.T @ basis[0]
    if dim == 6:
        result[:, 4:6] = angular.T @ basis[1:].T
    return result


def outer_actuated_contact_load_feasibility(
    contacts, com_world_m, required_wrench_world, *, effort_maps,
    free_effort_Nm, effort_bounds_Nm, moment_length_m=0.1,
):
    """Outer load check with pose-bound, caller-verified joint effort limits.

    At the supplied concrete pose, required effort is free_effort + sum(map @
    contact_components). Free effort includes gravity/dynamics not canceled by
    passive forces. Gravity compensation that is inside an actuator clamp must
    still consume effort. Bounds may explicitly include known passive stiction;
    that interpretation and native cap provenance are the caller's responsibility.

    This tests only the stated actuation model. It certifies neither actual
    contacts, exact friction, joint-stop/self-contact assistance, motion nor task
    success. These are solved forces, not measured normal-force budgets.
    """
    contacts = tuple(contacts)
    if not all(isinstance(c, ContactGeometry) for c in contacts):
        raise ValueError("contacts must contain validated ContactGeometry records")
    free = np.asarray(free_effort_Nm)
    if free.ndim != 1 or not len(free):
        raise ValueError("Need free effort for a nonempty revolute-joint order")
    free = _array(free, free.shape, "free effort")
    limits = _array(effort_bounds_Nm, (len(free), 2), "effort bounds")
    if (limits[:, 0] > limits[:, 1]).any():
        raise ValueError("Effort lower bounds must not exceed upper bounds")
    maps = tuple(effort_maps)
    if len(maps) != len(contacts):
        raise ValueError("Need one effort map per contact")
    maps = [_array(m, (len(free), c.dimension), "contact effort map")
            for m, c in zip(maps, contacts, strict=True)]
    matrix = np.concatenate(maps, axis=1) if maps else np.zeros((len(free), 0))
    return _solve_outer_load(
        contacts, com_world_m, required_wrench_world,
        moment_length_m=moment_length_m, actuation=(matrix, free, limits),
    )


def _solve_outer_load(
    contacts, com_world_m, required_wrench_world, *, moment_length_m, actuation=None,
):
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
        motor_limits_applied=actuation is not None,
        certifies_force_closure_or_dynamic_or_task_success=False,
    )
    if actuation is not None:
        effort_matrix, free, limits = actuation
        result['effort_map_pose_binding_and_native_caps_require_external_verification'] = True
    if not contacts:
        effort_ok = actuation is None or bool(np.all((free >= limits[:, 0]) & (free <= limits[:, 1])))
        feasible = bool(np.all(load == 0)) and effort_ok
        empty = dict(
            result, feasible_in_outer_model=feasible,
            status=("zero_load_without_contacts" if feasible else
                    "no_contacts_for_load" if effort_ok else "free_effort_outside_bounds"),
            solved_contact_components=[] if feasible else None,
            minimum_total_normal_force_N=0.0 if feasible else None,
        )
        if actuation is not None:
            empty['solved_required_joint_effort_Nm'] = free.tolist() if feasible else None
        return empty

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
    right_hand_side = [0.0] * len(inequalities)
    if actuation is not None:
        inequalities.extend(effort_matrix)
        inequalities.extend(-effort_matrix)
        right_hand_side.extend(limits[:, 1] - free)
        right_hand_side.extend(free - limits[:, 0])
    inequality = np.array(inequalities) if inequalities else None
    solved = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.asarray(right_hand_side) if inequalities else None,
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
    if actuation is not None:
        effort = free + effort_matrix @ x
        if not np.isfinite(effort).all() or np.any(effort < limits[:, 0] - 1e-7) or np.any(effort > limits[:, 1] + 1e-7):
            raise RuntimeError("Contact load solve failed independent joint effort checks")
        result['solved_required_joint_effort_Nm'] = effort.tolist()
    return dict(
        result, feasible_in_outer_model=True, status="optimal",
        solved_contact_components=components,
        component_order="normal_N, sliding1_N, sliding2_N, spin_Nm, rolling1_Nm, rolling2_Nm (truncated by dimension)",
        minimum_total_normal_force_N=float(x[normal_indices].sum()),
        wrench_residual_world=residual.tolist(),
    )

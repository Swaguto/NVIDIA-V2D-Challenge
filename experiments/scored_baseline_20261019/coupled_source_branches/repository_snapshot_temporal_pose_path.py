"""Choose a motion-bounded path through already source-qualified pose pools.

Second-order dynamic programming enforces linear and angular speed and
acceleration at the actual source timestamps. Missing pools stay failures.
This selector does not validate source fit, native contact or grasp stability.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class MotionLimits:
    linear_speed_m_s: float
    angular_speed_rad_s: float
    linear_acceleration_m_s2: float
    angular_acceleration_rad_s2: float

    def validate(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid nonnegative motion limit: {name}")


def motion_summary(poses, timestamps):
    poses = np.asarray(poses, dtype=float)
    t = np.asarray(timestamps, dtype=float)
    if poses.shape != (len(t), 4, 4) or not np.isfinite(poses).all():
        raise ValueError("Need finite world poses at every source timestamp")
    dt = np.diff(t)
    if not np.isfinite(t).all() or np.any(dt <= 0):
        raise ValueError("Source timestamps must be finite and strictly increasing")
    velocity = np.diff(poses[:, :3, 3], axis=0)/dt[:, None]
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    angular = (rotations[1:]*rotations[:-1].inv()).as_rotvec()/dt[:, None]
    mid_dt = (dt[:-1]+dt[1:])/2
    acceleration = np.diff(velocity, axis=0)/mid_dt[:, None]
    angular_acceleration = np.diff(angular, axis=0)/mid_dt[:, None]

    def maximum(a):
        return float(np.linalg.norm(a, axis=-1).max(initial=0))

    return dict(
        maximum_linear_speed_m_s=maximum(velocity),
        maximum_angular_speed_rad_s=maximum(angular),
        maximum_linear_acceleration_m_s2=maximum(acceleration),
        maximum_angular_acceleration_rad_s2=maximum(angular_acceleration),
        p95_linear_speed_m_s=float(np.quantile(np.linalg.norm(velocity, axis=-1), .95)) if len(dt) else 0.,
        p95_angular_speed_rad_s=float(np.quantile(np.linalg.norm(angular, axis=-1), .95)) if len(dt) else 0.)


def select_path(pose_pools, source_costs, timestamps, limits, *, weights=None):
    """Return selected pool indices or an explicit infeasible-path report.

    Every pool uses the same fixed world/object gauge. The caller must
    independently verify source evidence and native collisions for the chosen
    path. No interpolation or substitution is performed here.
    """
    limits.validate()
    t = np.asarray(timestamps, dtype=float)
    count = len(pose_pools)
    if count == 0 or t.shape != (count,) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Need a complete ordered source clock")
    if len(source_costs) != count:
        raise ValueError("Source costs must match all frame pools")
    w = dict(source=1., linear_speed=.1, angular_speed=.1,
             linear_acceleration=1., angular_acceleration=1.)
    if weights is not None:
        if set(weights)-set(w):
            raise ValueError("Unknown temporal objective weight")
        w.update(weights)
    if any(not np.isfinite(v) or v < 0 for v in w.values()):
        raise ValueError("Temporal weights must be finite and nonnegative")
    report = dict(schema="v2d.source-qualified-temporal-pose-path.v1",
        limits=asdict(limits), weights=w, numerical_tolerance=1e-9,
        source_fit_validation_performed=False, native_collision_validation_performed=False,
        missing_observations_filled=False, physical_task_accepted=False,
        feasible=False, stages=[])
    pools, costs = [], []
    for frame, (pool, unary) in enumerate(zip(pose_pools, source_costs, strict=True)):
        p, c = np.asarray(pool, dtype=float), np.asarray(unary, dtype=float)
        if len(p) == 0:
            report.update(failed_frame=frame, reason="empty source-qualified candidate pool")
            return None, report
        if p.shape != (len(p), 4, 4) or c.shape != (len(p),) or not np.isfinite(p).all() or not np.isfinite(c).all() or np.any(c < 0):
            raise ValueError("Need finite poses and nonnegative matching source costs")
        rotations = p[:, :3, :3]
        if not (np.allclose(p[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                and np.allclose(np.einsum('nji,njk->nik', rotations, rotations), np.eye(3), atol=1e-6, rtol=0)
                and np.allclose(np.linalg.det(rotations), 1., atol=1e-6, rtol=0)):
            raise ValueError("Pose pools must contain proper rigid transforms")
        pools.append(p)
        costs.append(w['source']*c)
    if count == 1:
        index = int(np.argmin(costs[0]))
        report.update(feasible=True, objective=float(costs[0][index]),
                      selected_motion=motion_summary(pools[0][index:index+1], t))
        return [index], report

    def edge(first, second, dt):
        a, b = pools[first], pools[second]
        linear = (b[None, :, :3, 3]-a[:, None, :3, 3])/dt
        ar = Rotation.from_matrix(np.repeat(a[:, :3, :3], len(b), axis=0))
        br = Rotation.from_matrix(np.tile(b[:, :3, :3], (len(a), 1, 1)))
        angular = (br*ar.inv()).as_rotvec().reshape(len(a), len(b), 3)/dt
        ln, an = np.linalg.norm(linear, axis=-1), np.linalg.norm(angular, axis=-1)
        valid = (ln <= limits.linear_speed_m_s+1e-9) & (an <= limits.angular_speed_rad_s+1e-9)
        cost = w['linear_speed']*(ln/max(limits.linear_speed_m_s, 1e-12))**2
        cost += w['angular_speed']*(an/max(limits.angular_speed_rad_s, 1e-12))**2
        return linear, angular, valid, cost

    dt = np.diff(t)
    previous_linear, previous_angular, valid, e = edge(0, 1, dt[0])
    value = costs[0][:, None]+costs[1][None, :]+e
    value[~valid] = np.inf
    report['stages'].append(dict(frame=1, feasible_pair_states=int(np.isfinite(value).sum())))
    if not np.isfinite(value).any():
        report.update(failed_frame=1, reason="no path satisfies speed limits")
        return None, report
    parents = {}
    for frame in range(2, count):
        linear, angular, valid, e = edge(frame-1, frame, dt[frame-1])
        mean_dt = (dt[frame-2]+dt[frame-1])/2
        acc = np.linalg.norm((linear[None, :, :, :]-previous_linear[:, :, None, :])/mean_dt, axis=-1)
        ang_acc = np.linalg.norm((angular[None, :, :, :]-previous_angular[:, :, None, :])/mean_dt, axis=-1)
        permitted = valid[None, :, :] & (acc <= limits.linear_acceleration_m_s2+1e-9) & (ang_acc <= limits.angular_acceleration_rad_s2+1e-9)
        transitions = value[:, :, None]+costs[frame][None, None, :]+e[None, :, :]
        transitions += w['linear_acceleration']*(acc/max(limits.linear_acceleration_m_s2, 1e-12))**2
        transitions += w['angular_acceleration']*(ang_acc/max(limits.angular_acceleration_rad_s2, 1e-12))**2
        transitions[~permitted] = np.inf
        parents[frame] = np.argmin(transitions, axis=0)
        value = np.min(transitions, axis=0)
        report['stages'].append(dict(frame=frame, feasible_pair_states=int(np.isfinite(value).sum())))
        if not np.isfinite(value).any():
            report.update(failed_frame=frame, reason="no path satisfies speed and acceleration limits")
            return None, report
        previous_linear, previous_angular = linear, angular
    selected = np.zeros(count, dtype=int)
    selected[-2:] = np.unravel_index(np.argmin(value), value.shape)
    for frame in range(count-1, 1, -1):
        selected[frame-2] = parents[frame][selected[frame-1], selected[frame]]
    chosen = np.stack([p[i] for p, i in zip(pools, selected, strict=True)])
    report.update(feasible=True, objective=float(value[selected[-2], selected[-1]]),
                  selected_motion=motion_summary(chosen, t))
    return selected.tolist(), report

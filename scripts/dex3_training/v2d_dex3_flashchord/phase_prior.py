"""Bounded phase-policy parameters, separate from source reference trajectories."""

import numpy as np

BOUNDS = np.array([0.1] * 3 + [0.35] * 3 + [0.65] * 7)


def smooth(x):
    u = np.clip(x, 0, 1)
    return u * u * (3 - 2 * u)


def project(values, lower=None, upper=None):
    result = np.clip(
        values, -BOUNDS if lower is None else lower, BOUNDS if upper is None else upper
    )
    for begin, end, radius in ((0, 3, 0.1), (3, 6, 0.35)):
        vectors = result[..., begin:end]
        result[..., begin:end] /= np.maximum(
            1.0, np.linalg.norm(vectors, axis=-1, keepdims=True) / radius
        )
    return result


def phase_envelope(phase, times):
    release = 1 - smooth((times - phase["release_start_s"]) / 0.25)
    wrist = smooth((times - phase["approach_start_s"]) / 0.5) * release
    finger = smooth((times - phase["close_s"] + 0.25) / 0.125) * release
    return np.stack([wrist] * 6 + [finger] * 7, axis=1)


def schedule(spec, plan, sides, finger_names, times):
    if plan.get("schema") != "v2d.native-phase-policy-prior.v1":
        raise ValueError("Unsupported phase-policy plan")
    phases = {
        p["sample"]: (side, p) for side, rows in spec["phases"].items() for p in rows
    }
    out = np.zeros((len(times), len(sides) * 13), np.float32)
    for name, raw in plan["phase_actions"].items():
        if name not in phases:
            raise ValueError("Unknown fitted phase")
        raw = np.asarray(raw, float)
        if (
            raw.shape != (13,)
            or not np.isfinite(raw).all()
            or not np.all(abs(project(raw) - raw) <= 1e-8)
        ):
            raise ValueError("Fitted phase action exceeds physical residual bounds")
        side, phase = phases[name]
        names = plan["finger_joint_names"][side]
        if len(names) != 7 or set(names) != set(finger_names[side]):
            raise ValueError("Phase prior finger schema differs")
        order = [names.index(n) for n in finger_names[side]]
        arranged = np.r_[raw[:6], raw[6:][order]]
        start = sides.index(side) * 13
        out[:, start : start + 13] += phase_envelope(phase, times) * arranged
    return out

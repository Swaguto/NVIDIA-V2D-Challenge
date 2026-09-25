"""Development-only trajectory perturbations; originals are never changed."""
import numpy as np


def perturb(relative_positions, valid, development_sequence_id, permitted_ids,
            mode, seed=17, magnitude=.01, shift_frames=1):
    if development_sequence_id not in permitted_ids:
        raise ValueError("perturbations are restricted to the declared development split")
    positions = np.array(relative_positions, dtype=float, copy=True)
    mask = np.array(valid, dtype=bool, copy=True)
    if positions.shape[:-1] != mask.shape or positions.shape[-1] != 3:
        raise ValueError("positions[T,H,B,3] and validity[T,H,B] must align")
    if magnitude < 0 or not np.isfinite(magnitude):
        raise ValueError("nonnegative finite magnitude required")
    rng = np.random.default_rng(seed)
    if mode == "dropout":
        if magnitude > 1: raise ValueError("dropout probability must be at most one")
        mask &= rng.random(mask.shape) >= magnitude
    elif mode == "pose_jitter":
        positions += rng.normal(0, magnitude, positions.shape)
    elif mode == "timing":
        if not isinstance(shift_frames, int) or not 0 < shift_frames < len(positions):
            raise ValueError("timing shift must be a positive integer shorter than the sequence")
        positions[shift_frames:] = positions[:-shift_frames].copy()
        mask[shift_frames:] = mask[:-shift_frames].copy()
        mask[:shift_frames] = False  # no circular wraparound
    else:
        raise ValueError("unknown corruption mode")
    return positions, mask


def corrupt_contacts(points, normals, active, episode_id, development_ids,
                     mode, magnitude, seed=17):
    """Separate contact corruption from object/hand trajectory perturbation.

    Points/normals use [...,3], contact mask [...]. Rotation magnitude is radians;
    positional jitter is meters; dropout magnitude is a probability.
    """
    from .track3 import PROXY_EPISODES
    if episode_id in PROXY_EPISODES or episode_id not in development_ids:
        raise ValueError("contact corruption is development-only")
    p, n, a = np.array(points, float, copy=True), np.array(normals, float, copy=True), np.asarray(active)
    if p.shape != n.shape or p.shape[-1:] != (3,) or a.shape != p.shape[:-1] or a.dtype != bool:
        raise ValueError("aligned points/normals and boolean active mask required")
    if not np.isfinite(p).all() or not np.isfinite(n).all() or not np.isfinite(magnitude) or magnitude < 0:
        raise ValueError("finite contacts and nonnegative magnitude required")
    if not np.allclose(np.linalg.norm(n[a], axis=-1), 1, atol=1e-4, rtol=0):
        raise ValueError("active contact normals must be unit length")
    a = a.copy(); rng = np.random.default_rng(seed)
    if mode == "contact_dropout":
        if magnitude > 1: raise ValueError("dropout probability exceeds one")
        a &= rng.random(a.shape) >= magnitude
    elif mode == "contact_position_jitter":
        p[a] += rng.normal(0, magnitude, p[a].shape)
    elif mode == "contact_normal_rotation":
        if magnitude > np.pi: raise ValueError("normal rotation exceeds pi radians")
        axis = rng.normal(size=n[a].shape)
        axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
        angle = rng.uniform(-magnitude, magnitude, size=(a.sum(), 1))
        v = n[a]
        n[a] = v * np.cos(angle) + np.cross(axis, v) * np.sin(angle) + axis * np.sum(axis*v, axis=-1, keepdims=True) * (1-np.cos(angle))
    else:
        raise ValueError("unknown contact corruption")
    return p, n, a

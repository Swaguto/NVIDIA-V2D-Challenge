"""NEW symbolic joint collision construction; no episode kinematics/evaluation."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np

S = Path('/Users/rohithiruku/v2d_scored_baseline_20261017')
C = S / 'compute'
F = S / 'e037-thumb-ray-independent-validation-v1/frozen'
M = S / 'e037-coupled-angle-model-v1'
G = S / 'e037-object-full-pose-factors-v1'
O = S / 'e037-joint-collision-maps-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(value):
    if isinstance(value, np.ndarray):
        return normalize(value.tolist())
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {'nonfinite': repr(value)}
    return value


def save(path, value):
    path = Path(path)
    with path.open('x') as file:
        json.dump(normalize(value), file, indent=2, sort_keys=True, allow_nan=False)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())


def save_npz(path, **arrays):
    with Path(path).open('xb') as file:
        np.savez(file, **arrays)
        file.flush()
        os.fsync(file.fileno())


def ancestors(parents, body):
    result = []
    while body:
        if body in result:
            raise ValueError('cyclic native tree')
        result.append(body)
        body = int(parents[body])
    return list(reversed(result))


def skew(a):
    x, y, z = a
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def rod(axis, q):
    """Only used on NEW synthetic examples by the contracts."""
    k = skew(np.asarray(axis))
    return np.eye(3) + np.sin(q) * k + (1 - np.cos(q)) * (k @ k)


def hinge(origin, axis, q):
    rotation = rod(axis, q)
    transform = np.eye(4, dtype=np.result_type(q, float))
    transform[:3, :3] = rotation
    transform[:3, 3] = origin - rotation @ origin
    return transform


def whole_support(vertices, direction):
    values = vertices @ direction
    maximum = np.max(values)
    ties = np.flatnonzero(values == maximum)
    return maximum, ties


def pair_category(a, b):
    if a == b:
        return a + '_internal'
    return '_'.join(sorted([a, b]))


def geometry_group(body_name):
    if body_name.startswith('dex3_left_'):
        return 'left'
    if body_name.startswith('dex3_right_'):
        return 'right'
    if body_name == 'wooden_spoon_object':
        return 'object'
    if body_name == 'support':
        return 'support'
    raise ValueError('unknown geometry body ' + body_name)

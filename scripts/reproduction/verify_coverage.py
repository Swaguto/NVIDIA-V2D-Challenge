"""Fail on any frame-level coverage change in a frozen reconstruction replay.

This verifies repeatability, not pose accuracy or fresh raw-video inference.
The reference JSON must come from a separately hash-verified release archive.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_files(root, entries):
    for name, digest in entries.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe manifest path')
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Manifest path escapes root')
        if sha(path) != digest:
            raise ValueError('Changed input: ' + name)


def check_mask(actual, expected, label):
    if actual.dtype != np.dtype(bool) or actual.shape != expected.shape:
        raise ValueError(label + ': invalid mask schema')
    if not np.array_equal(actual, expected):
        changed = np.argwhere(actual != expected).tolist()
        raise ValueError(f'{label}: frame/side coverage differs at {changed[:20]}')


def verify(reference, inputs, raw, bundle, retarget):
    contract = json.loads(reference.read_text())
    if contract['schema'] != 'v2d.coverage-reference.v1':
        raise ValueError('Unsupported reference schema')
    frames = np.asarray(contract['frames'], dtype=np.int64)
    if frames.ndim != 1 or len(set(frames.tolist())) != len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError('Reference frame IDs must be unique and increasing')
    expected = np.asarray([[int(f) in contract['hand_frames'][s] for s in ('left', 'right')]
                           for f in frames], dtype=bool)
    lock = inputs / 'input-files.json'
    if sha(lock) != contract['input_lock_sha256']:
        raise ValueError('Input lock differs from the reviewed reconstruction')
    verify_files(inputs, json.loads(lock.read_text()))
    verify_files(raw, contract['raw_files'])
    metadata = json.loads((inputs / 'inputs.json').read_text())
    if metadata['episode'] != contract['episode'] or metadata['frames'] != frames.tolist():
        raise ValueError('Input episode/timeline differs')
    for side in ('left', 'right'):
        actual_frames = sorted(int(p.stem) for p in (inputs / 'hands' / side / 'poses').glob('*.json'))
        if actual_frames != contract['hand_frames'][side] or metadata['hand_frames'][side] != actual_frames:
            raise ValueError(side + ': input pose frame IDs differ')
    with np.load(bundle, allow_pickle=False) as data:
        if not np.array_equal(data['frame_indices'], frames):
            raise ValueError('Bundle frame IDs differ')
        check_mask(data['hand_valid'], expected, 'Bundle')
        check_mask(data['hand_directly_observed'], expected, 'Observation provenance')
        joints = data['hand_joints_world']
        if joints.shape != (len(frames), 2, 21, 3):
            raise ValueError('Hand joint schema differs')
        if not np.isfinite(joints[expected]).all() or not np.isnan(joints[~expected]).all():
            raise ValueError('Hand joint finite/unknown states disagree with coverage')
        timestamps = data['timestamps_s'].copy()
    with np.load(retarget, allow_pickle=False) as data:
        if not np.array_equal(data['frame_indices'], frames) or not np.array_equal(data['timestamps_s'], timestamps):
            raise ValueError('Robot timeline differs')
        check_mask(data['source_hand_valid'], expected, 'Robot source')
        check_mask(data['robot_valid'], expected, 'Robot output')
        qpos = data['robot_qpos']
        if qpos.shape != (len(frames), 2, 14):
            raise ValueError('Robot joint schema differs')
        if not np.isfinite(qpos[expected]).all() or not np.isnan(qpos[~expected]).all():
            raise ValueError('Robot finite/unknown states disagree with coverage')
    return {'status': 'coverage_matches', 'episode': contract['episode'], 'frames': len(frames),
            'hand_frames': contract['hand_frames'],
            'counts': {s: len(contract['hand_frames'][s]) for s in ('left', 'right')},
            'reference_sha256': sha(reference), 'input_lock_sha256': sha(lock),
            'bundle_sha256': sha(bundle), 'retarget_sha256': sha(retarget),
            'fresh_inference_verified': False, 'pose_accuracy_accepted': False,
            'scope': 'raw file identity, frozen inputs and exact frame/side coverage; not geometry equivalence'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'inputs', 'raw', 'bundle', 'retarget', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    try:
        report = verify(args.reference, args.inputs, args.raw, args.bundle, args.retarget)
    except Exception as exc:
        args.out.write_text(json.dumps({'status': 'failed', 'error': str(exc)}, indent=2) + '\n')
        raise
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'episode', 'frames', 'counts')}))


if __name__ == '__main__':
    main()

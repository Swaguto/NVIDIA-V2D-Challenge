"""Presentation-only Dex3 holds. Never promotes a missing observation to valid."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def hold_rows(valid):
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2 or valid.shape[1] != 2:
        raise ValueError('Expected full timeline x two hands')
    rows = np.empty(valid.shape, dtype=np.int64)
    for side in range(2):
        supported = np.flatnonzero(valid[:, side])
        if not len(supported):
            raise ValueError('Cannot display a hand with no supported pose')
        previous = int(supported[0])
        for row in range(len(valid)):
            if valid[row, side]:
                previous = row
            rows[row, side] = previous
    return rows


def build(retarget: Path, out: Path):
    if out.exists():
        raise FileExistsError(out)
    contract = json.loads((retarget / 'manifest.json').read_text())
    if digest(retarget / 'retarget.npz') != contract['payload']['sha256']:
        raise ValueError('Retarget payload differs from its manifest')
    with np.load(retarget / 'retarget.npz', allow_pickle=False) as source:
        valid = source['robot_valid']
        rows = hold_rows(valid)
        frames = source['frame_indices']
        qpos = np.stack([source['robot_qpos'][rows[:, side], side] for side in range(2)], axis=1)
        arrays = dict(frame_indices=frames, source_robot_valid=valid,
                      display_valid=np.ones_like(valid), held=~valid,
                      pose_source_rows=rows, pose_source_frames=frames[rows], robot_qpos=qpos)
        for side_id, side in enumerate(('left', 'right')):
            arrays[f'robot_{side}_frame_poses'] = source[f'robot_{side}_frame_poses'][rows[:, side_id]]
            if not np.array_equal(qpos[valid[:, side_id], side_id], source['robot_qpos'][valid[:, side_id], side_id]):
                raise ValueError('Observed targets changed')
            bounds = contract['joints'][side]
            if np.any(qpos[:, side_id, 7:] < np.array(bounds['lower_radians']) - 1e-6) or np.any(qpos[:, side_id, 7:] > np.array(bounds['upper_radians']) + 1e-6):
                raise ValueError('Finger joint limit violation')
        for name, array in arrays.items():
            if not np.isfinite(array).all():
                raise ValueError('Nonfinite presentation array: ' + name)
        if not np.allclose(np.linalg.norm(qpos[:, :, 3:7], axis=-1), 1, atol=1e-6):
            raise ValueError('Nonunit root quaternion')
        out.mkdir(parents=True)
        np.savez_compressed(out / 'presentation.npz', **arrays)
    manifest = dict(schema='v2d.dex3.presentation.hold.v1',
                    retarget_manifest_sha256=digest(retarget / 'manifest.json'),
                    retarget_payload_sha256=digest(retarget / 'retarget.npz'),
                    payload_sha256=digest(out / 'presentation.npz'),
                    builder_sha256=digest(Path(__file__)),
                    source_frames=len(frames), observed_counts=valid.sum(axis=0).tolist(),
                    held_counts=(~valid).sum(axis=0).tolist(),
                    policy='Hold previous supported pose; leading gaps use first supported pose, explicitly labelled',
                    observed_targets_unchanged=True, pose_accuracy_accepted=False,
                    contact_or_training_targets=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify(retarget, out)
    return manifest


def verify(retarget: Path, presentation: Path):
    manifest = json.loads((presentation / 'manifest.json').read_text())
    if manifest['schema'] != 'v2d.dex3.presentation.hold.v1':
        raise ValueError('Unknown presentation schema')
    if manifest.get('contact_or_training_targets') is not False or manifest.get('pose_accuracy_accepted') is not False or manifest.get('observed_targets_unchanged') is not True:
        raise ValueError('Presentation holds cannot claim pose accuracy, contact or training acceptance')
    if manifest['retarget_manifest_sha256'] != digest(retarget / 'manifest.json') or manifest['retarget_payload_sha256'] != digest(retarget / 'retarget.npz') or manifest['payload_sha256'] != digest(presentation / 'presentation.npz'):
        raise ValueError('Presentation provenance mismatch')
    with np.load(retarget / 'retarget.npz', allow_pickle=False) as source, np.load(presentation / 'presentation.npz', allow_pickle=False) as display:
        valid = source['robot_valid']; rows = hold_rows(valid)
        if manifest.get('source_frames') != len(valid) or manifest.get('observed_counts') != valid.sum(axis=0).tolist() or manifest.get('held_counts') != (~valid).sum(axis=0).tolist():
            raise ValueError('Presentation support counts differ from source observations')
        expected = dict(frame_indices=source['frame_indices'], source_robot_valid=valid,
                        display_valid=np.ones_like(valid), held=~valid, pose_source_rows=rows,
                        pose_source_frames=source['frame_indices'][rows],
                        robot_qpos=np.stack([source['robot_qpos'][rows[:, s], s] for s in range(2)], axis=1))
        for s, side in enumerate(('left', 'right')):
            expected[f'robot_{side}_frame_poses'] = source[f'robot_{side}_frame_poses'][rows[:, s]]
        if set(display.files) != set(expected):
            raise ValueError('Unexpected presentation keys')
        for key, value in expected.items():
            if not np.array_equal(display[key], value) or not np.isfinite(value).all():
                raise ValueError('Presentation changed: ' + key)
        return {key: display[key].copy() for key in display.files}, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retarget', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.retarget, args.out), indent=2))

"""Private preview adapter: repaired MANO geometry -> hash-bound Dex3 IK inputs."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import argparse
parser=argparse.ArgumentParser(description=__doc__)
for key in ['run','checkpoint','mano','out']:
    parser.add_argument('--'+key,type=Path,required=True)
parser.add_argument('--episode',type=int,required=True)
args=parser.parse_args()
CODE = Path(__file__).resolve().parents[2]
ROBOT = CODE / 'robotic_grounding'
sys.path[:0] = [str(CODE / 'reconstruction/modules/v2d_issue6'), str(ROBOT / 'source/robotic_grounding')]
from issue6.backend import dynhamr_frame_axis
from issue6.geometry import hand_geometry, transform
from issue6.manifest import load, camera_from_object
from issue6.pipeline import load_mesh
from robotic_grounding.retarget.track3_bundle import SCHEMA, SIDES, JOINT_NAMES, digest, validate_bundle, wrist_rotation

torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location('render_hands', CODE / 'reconstruction/modules/v2d_hamer/lib/render_hands_aligned_video.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
def export(ep):
    label = f'e{ep:03d}'
    out = args.out.resolve()
    if out.exists():raise FileExistsError('Use a fresh output directory: '+str(out))
    out.mkdir(parents=True)
    run = args.run.resolve()
    m = load(run / 'manifest.json')
    frames = m['frames']
    n = len(frames)
    ids = np.array([f['id'] for f in frames])
    times = np.array([f['timestamp_s'] for f in frames])
    fps = float(1 / np.median(np.diff(times)))
    np.testing.assert_allclose(times, ids / fps, rtol=1e-5, atol=1e-6)
    c = m['color_camera']
    checkpoint = args.checkpoint.resolve()
    axis = dynhamr_frame_axis(checkpoint, frames)
    index = {int(f): i for i, f in enumerate(ids)}
    w2c = np.tile(np.eye(4), (n, 1, 1))
    camera_known = np.zeros(n, bool)
    with np.load(checkpoint) as fit:
        rotations, translations = fit['cam_R'], fit['cam_t']
        if len(rotations) > 1:
            np.testing.assert_allclose(rotations[0], rotations[1], atol=1e-5)
            np.testing.assert_allclose(translations[0], translations[1], atol=1e-5)
        for native, frame in enumerate(axis):
            i = index[frame['id']]
            if np.isfinite(rotations[0, native]).all() and np.isfinite(translations[0, native]).all():
                w2c[i, :3, :3] = rotations[0, native]
                w2c[i, :3, 3] = translations[0, native]
                camera_known[i] = True
    c2w = np.linalg.inv(w2c)
    arrays = dict(frame_indices=ids, timestamps_s=times, camera_c2w=c2w,
                  camera_native_valid=camera_known,
                  intrinsics=np.array([[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0, 0, 1]]),
                  image_size=np.array([c['width'], c['height']]),
                  hand_valid=np.zeros((n, 2), bool), hand_directly_observed=np.zeros((n, 2), bool),
                  hand_metric_valid=np.zeros((n, 2), bool),
                  hand_joints_world=np.full((n, 2, 21, 3), np.nan),
                  hand_vertices_world=np.full((n, 2, 778, 3), np.nan),
                  hand_wrist_world=np.full((n, 2, 4, 4), np.nan),
                  contact_known=np.zeros((n, 2, 5), bool), contact_probability=np.full((n, 2, 5), np.nan),
                  wrench_known=np.zeros((n, 2), bool), wrench_world_N_Nm=np.full((n, 2, 6), np.nan))
    frozen = {'manifest.json': run / 'manifest.json', 'continuity-receipt.json': run / 'continuity-receipt.json',
              'native-camera-checkpoint.npz': checkpoint}
    mano = renderer._mano_layer(str(args.mano.resolve()))
    for si, side in enumerate(SIDES):
        for p in sorted((run / 'align/records' / side).glob('*.json')):
            rec = json.loads(p.read_text())
            assert rec['is_right'] == (side == 'right') and rec['frame_idx'] == int(p.stem)
            frozen[f'hands/{side}/{p.name}'] = p
            if not rec.get('visual_valid', False):
                continue
            i = index[rec['frame_idx']]
            if not camera_known[i]:
                raise ValueError('Visible hand cannot use an unknown native camera')
            aa = torch.tensor([rec['mano']['global_orient'] + rec['mano']['hand_pose']], dtype=torch.float32)
            beta = torch.tensor([rec['mano']['betas']], dtype=torch.float32)
            with torch.no_grad():
                prediction = mano(aa, beta)
            v, j = hand_geometry(prediction.verts[0].numpy(), prediction.joints[0].numpy(),
                                 rec['is_right'], rec['hand_scale'], rec['cam_t'])
            arrays['hand_vertices_world'][i, si] = transform(v, c2w[i])
            arrays['hand_joints_world'][i, si] = transform(j, c2w[i])
            wrist = np.eye(4)
            wrist[:3, :3] = c2w[i, :3, :3] @ wrist_rotation(rec, side)
            wrist[:3, 3] = arrays['hand_joints_world'][i, si, 0]
            arrays['hand_wrist_world'][i, si] = wrist
            arrays['hand_valid'][i, si] = True
            arrays['hand_directly_observed'][i, si] = bool(rec.get('directly_observed'))
            arrays['hand_metric_valid'][i, si] = bool(rec['diagnostics']['metric_valid'])
            # Exact renderer parity prevents a second MANO coordinate convention.
            vv, faces = renderer._build_hand_mesh(rec, mano)
            np.testing.assert_allclose(v, vv, atol=1e-6)
        arrays[f'hand_{side}_faces'] = mano.th_faces.numpy()[:, [0, 2, 1]] if si == 0 else mano.th_faces.numpy()
    objects = []
    for oi, obj in enumerate(m['objects']):
        mesh = load_mesh(obj['mesh'])
        arrays[f'object_{oi}_vertices'] = mesh.vertices.copy()
        arrays[f'object_{oi}_faces'] = mesh.faces.copy()
        poses = np.tile(np.eye(4), (n, 1, 1))
        known = np.zeros(n, bool)
        for i, frame in enumerate(frames):
            pose = camera_from_object(frame, obj)
            if pose is not None:
                poses[i] = c2w[i] @ pose
                known[i] = True
                path = frame['objects'][obj['id']].get('pose_path')
                if path:
                    frozen[f'objects/{obj["id"]}/{ids[i]:06d}.json'] = Path(path)
        arrays[f'object_{oi}_world'] = poses
        arrays[f'object_{oi}_valid'] = known
        arrays[f'object_{oi}_scale'] = np.ones(n)
        frozen[f'meshes/{obj["id"]}.glb'] = Path(obj['mesh'])
        objects.append(dict(index=oi, name=obj['id'], pose_source='unchanged saved CAD estimates', geometry_provenance='provided development CAD'))
    sources = []
    for name, original in frozen.items():
        dest = out / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        before = digest(original)
        shutil.copyfile(original, dest)
        assert digest(dest) == before == digest(original)
        sources.append(dict(path='source/' + name, sha256=before))
    np.savez_compressed(out / 'episode.npz', **arrays)
    metadata = dict(schema=SCHEMA, episode=ep, units='metres', fps=fps, frames=ids.tolist(),
                    hand_sides=list(SIDES), mano_joint_names=JOINT_NAMES, objects=objects,
                    payload=dict(path='episode.npz', sha256=digest(out / 'episode.npz')), sources=sources,
                    source_repair_commit='9f93a50', retarget_code_commit='3db10766bf09bca10fe1b452afe5c78e1649c870',
                    exporter_sha256=digest(Path(__file__)),
                    scope='VISUAL REVIEW INPUTS: hand_valid means visual geometry available, including labeled estimates; hand_metric_valid remains separate',
                    unknown_policy='NaN hand geometry and false hand_valid; original metric validity and all unknown contacts preserved',
                    camera_unknown_policy='Identity gauge only for leading unknown-camera frames with no visible hand; source CAD remains camera relative',
                    robot_motion_inferred=True, physical_task_accepted=False)
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    report = validate_bundle(out)
    report['source_metric_hand_frames'] = arrays['hand_metric_valid'].sum(axis=0).tolist()
    (out / 'validation.json').write_text(json.dumps(report, indent=2))
    print('BUNDLE_READY', label, n, arrays['hand_valid'].sum(axis=0).tolist(), flush=True)


if __name__ == '__main__':
    export(args.episode)

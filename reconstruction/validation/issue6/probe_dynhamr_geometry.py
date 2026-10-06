"""Check camera conversion against actual upstream MANO vertices, without fitting substitutes."""
import importlib.util
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
sys.path.insert(0, '/workspace/Dyn-HaMR/dyn-hamr')
from body_model import MANO
from body_model.utils import run_mano
from issue6.backend import final_dynhamr_result
from v2d.hand_alignment.lib.dynhamr_to_hamer_tracks import _axis_angle_to_rotmat, _rotmat_to_axis_angle, _camera_translation

torch.set_num_threads(4)
parser = argparse.ArgumentParser()
parser.add_argument('--episode', type=int, default=2)
parser.add_argument('--checkpoint', type=Path)
parser.add_argument('--receipt-path', type=Path)
parser.add_argument('--all-frames', action='store_true')
parser.add_argument('--converted-records', type=Path)
args = parser.parse_args()
root = Path(f'/workspace/temporal-hands/e{args.episode:03d}')
path = args.checkpoint or final_dynhamr_result(root / 'fit')
with np.load(path, allow_pickle=False) as data:
    ids = list(range(data['trans'].shape[1])) if args.all_frames else [0, data['trans'].shape[1]//2, data['trans'].shape[1]-1]
    b, t = data['trans'].shape[0], len(ids)
    model = MANO('/workspace/hamer-weights/_DATA/data/mano', batch_size=b*t, pose2rot=True,
                 num_hand_joints=15, create_body_pose=False)
    tensors = {key: torch.tensor(data[key][:,ids], dtype=torch.float32)
               for key in ('trans','root_orient','pose_body','is_right')}
    tensors['betas'] = torch.tensor(data['betas'], dtype=torch.float32)
    with torch.no_grad():
        vertices = run_mano(model,tensors['trans'],tensors['root_orient'],
                             tensors['pose_body'].reshape(b,t,45),tensors['is_right'],tensors['betas'])['vertices'].numpy()
    spec = importlib.util.spec_from_file_location('align_probe', '/workspace/pr32/reconstruction/modules/v2d_hamer/lib/align_hands.py')
    align = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(align)
    worker = align._make_mano('/workspace/mano')
    rows = []
    for h in range(b):
        right = bool(data['is_right'][h].mean()>.5)
        mirror = np.eye(3) if right else np.diag([-1.,1.,1.])
        for n, frame in enumerate(ids):
            cam_r, cam_t = data['cam_R'][h,frame], data['cam_t'][h,frame]
            aa = _rotmat_to_axis_angle(mirror @ cam_r @ mirror @ _axis_angle_to_rotmat(data['root_orient'][h,frame]))
            pose = np.concatenate([aa,data['pose_body'][h,frame].reshape(45)]).astype(np.float32)
            record = None
            if args.converted_records:
                record = json.loads((args.converted_records / str(3 if right else 2) / f'{frame:06d}.json').read_text())
                pose = np.array(record['mano']['global_orient'] + record['mano']['hand_pose'], dtype=np.float32)
            with torch.no_grad():
                hand = worker(torch.tensor(pose)[None],tensors['betas'][h:h+1])
                local = hand.verts[0].numpy()
            translation = _camera_translation(data['trans'][h,frame],cam_r,cam_t,mirror,hand.joints[0,0].numpy())
            if record:
                translation = np.array(record['camera']['pred_cam_t_full'])
            converted = local @ mirror.T + translation
            native = vertices[h,n] @ cam_r.T + cam_t
            errors = np.linalg.norm(converted-native,axis=1)
            rows.append({'hand':'right' if right else 'left','cropped_frame':frame,
                         'max_vertex_error_m':float(errors.max()),'mean_vertex_error_m':float(errors.mean())})
receipt = {'episode': args.episode, 'result_used':str(path),
           'purpose':'Final checkpoint camera conversion agrees with actual upstream MANO vertices; this checks geometry conventions, not ground-truth accuracy',
           'converted_records': str(args.converted_records) if args.converted_records else None,
           'checks':rows}
(args.receipt_path or root / 'dynhamr_camera_conversion_probe.json').write_text(json.dumps(receipt,indent=2)+'\n')
failures = [row for row in rows if row['max_vertex_error_m'] >= 1e-5]
print(json.dumps({'episode': args.episode, 'checks': len(rows), 'max_error_m': max(r['max_vertex_error_m'] for r in rows), 'failures': failures}, indent=2),flush=True)
assert not failures, 'Camera conversion exceeded unchanged 1e-5 m tolerance'

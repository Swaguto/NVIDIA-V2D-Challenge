"""Render consecutive frames around measured motion extremes for visual review."""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--episode', type=int, required=True)
args = parser.parse_args()
root = Path(f'/workspace/temporal-hands/e{args.episode:03d}')
run = Path(os.environ.get('V2D_ISSUE6_RUN_ROOT','/workspace/issue6-full-runs/calibrated-temporal')) / f'e{args.episode:03d}'
out = Path(os.environ.get('V2D_ISSUE6_REVIEW_ROOT','/workspace/pr32-evidence/review')) / f'e{args.episode:03d}'
hands = json.loads((run / 'align/hands.json').read_text())
with np.load(root / 'cameras/vipe_c2w.npz', allow_pickle=False) as data:
    cameras = data['data']
peaks = []
for right in (False, True):
    rows = sorted((r for r in hands if bool(r['is_right']) == right), key=lambda r:r['frame_id'])
    frames = np.array([r['frame_id'] for r in rows])
    times = np.array([r['timestamp_s'] for r in rows])
    tracks = np.array([r['track_id'] for r in rows])
    joints = np.array([r['joints_camera_m'] for r in rows])
    world = np.einsum('nij,nkj->nki', cameras[frames,:3,:3], joints)+cameras[frames,None,:3,3]
    velocity = np.diff(world,axis=0)/np.diff(times)[:,None,None]
    acceleration = np.linalg.norm(np.diff(velocity,axis=0)/((times[2:]-times[:-2])/2)[:,None,None],axis=-1)
    score = np.median(acceleration, axis=1)
    valid = (np.diff(frames)[:-1]==1)&(np.diff(frames)[1:]==1)&(tracks[:-2]==tracks[1:-1])&(tracks[1:-1]==tracks[2:])
    supported = np.array([r['metric_valid'] for r in rows], bool)
    metric = supported[:-2]&supported[1:-1]&supported[2:]
    for label, mask in [('all directly observed',valid),('metric supported',valid&metric)]:
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        index = int(indices[np.argmax(score[indices])])
        peaks.append({'hand':'right' if right else 'left', 'subset':label,
                      'frame_ids':frames[index:index+3].tolist(),
                      'median_joint_acceleration_m_s2':float(score[index]),
                      'metric_valid':supported[index:index+3].tolist()})
cap = cv2.VideoCapture(str(out/'hand-camera-overlay.mp4'))
canvas = Image.new('RGB',(1920,428*len(peaks)),(12,17,24))
draw = ImageDraw.Draw(canvas)
for row, peak in enumerate(peaks):
    for column, frame_id in enumerate(peak['frame_ids']):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
        okay, frame = cap.read()
        assert okay
        tile = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((640,400))
        x,y=640*column,428*row
        canvas.paste(tile,(x,y))
        draw.text((x+6,y+403),f"{peak['hand']} | {peak['subset']} | frame {frame_id} | {peak['median_joint_acceleration_m_s2']:.1f} m/s2",fill='white')
cap.release()
canvas.save(out/'motion-extremes.jpg',quality=94)
(out/'motion-extremes.json').write_text(json.dumps({'episode':args.episode,'selection':'Maximum median joint acceleration on consecutive same-track triples, in estimated ViPE world coordinates; not ground-truth accuracy','peaks':peaks},indent=2)+'\n')
print(json.dumps(peaks),flush=True)

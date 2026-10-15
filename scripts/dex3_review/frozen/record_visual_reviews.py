import json
from pathlib import Path

R=Path('/Users/rohithiruku/v2d_episode_previews/recovered_20261014/continuity_fix/dex3_isaac')
proof=json.loads((R/'playback-verification.json').read_text())
for ep,p in proof.items():
    if not (p['ended'] and p['rate']==1 and p.get('error') is None):raise ValueError('Playback is incomplete: '+ep)
    folder=R/ep
    n=json.load(open(folder/'isaac-receipt.json'))['source_frames']
    rows=json.load(open(folder/'bundle-validation.json'))
    notes=['Sampled hand side identity and gross motion follow the supplied retargeting.',
           'Finger/object fit remains approximate; physical grasp success is not accepted.']
    if sum(rows['hand_unknown_frames'].values()):notes.append('Unknown input geometry remains absent in the robot display.')
    if ep=='e002':notes.append('The oblique view explicitly marks five frames without a native world-camera pose unavailable.')
    if ep=='e021':notes.append('The long right-hand input gap is retained; object motion is kinematic playback.')
    if ep=='e042':notes.append('Source finger articulation remains approximate after the reviewed mask recovery.')
    (folder/'visual-review.json').write_text(json.dumps({'episode':ep,'summary':'Full playback and sampled input/robot correspondence reviewed; grasp physics unvalidated.','sampled_source_frames':[round(i*(n-1)/7) for i in range(8)],'checks':['Eight frame-aligned input/source/oblique samples inspected','Full normal-rate comparison playback reached its end without a media error','Frame counts and strictly increasing presentation timestamps verified','Simulator root/joint/link/camera and asset locks checked'],'playback':p,'findings':notes,'user_grasp_acceptance':False},indent=2))
print('VISUAL_REVIEWS_RECORDED',len(proof))

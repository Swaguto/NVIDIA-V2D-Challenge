"""Prepare one private eight-image source recovery probe; no inference or billing."""
from pathlib import Path
import hashlib,json,sys
import cv2,numpy as np
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path[:0]=[str(R/'scripts/track3'),str(R/'reconstruction/modules/v2d_stereo_depth/lib')]
from recover_source_hand import digest,validate_request,SCHEMA
from stereo_depth import load_rig
O=S/'e037-fresh-source-hand-request-v1';O.mkdir();(O/'images').mkdir()
E=S/'e037-inferred-hand-image-evidence-v1';identity=json.load(open(S/'e037-source-hand-image-motion-v3/report.json'));evidence=json.load(open(E/'report.json'));B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:direct=z['hand_directly_observed'].copy();times=z['timestamps_s'].copy();K=z['intrinsics'].copy()
with np.load(E/'image-observations.npz') as z:points=z['landmarks_px'].copy()
rig=load_rig(str(S/'dataset/track_3/public/meta/camera_calibration.json'));V=[S/f'dataset/track_3/evaluation/videos/chunk-000/observation.images.{v}/episode_000037.mp4' for v in [rig.left,rig.right]];caps=[cv2.VideoCapture(str(p)) for p in V];selected=[25,35,40,45,50,55,67,99];rows=[]
for f in range(100):
 images=[]
 for cap in caps:
  ok,im=cap.read();assert ok;images.append(im)
 if f not in selected:continue
 image,_=rig.rectify_pair(*images);path=O/'images'/f'{f:06d}.png';cv2.imwrite(str(path),image);active=np.isfinite(points[f]).all(-1);assert active.sum()>=4;kp=np.zeros((21,3));kp[active,:2]=points[f,active];kp[active,2]=.5;p=points[f,active];centre=(p.min(0)+p.max(0))/2;size=max(256.,3.*float(np.ptp(p,axis=0).max()));prior=json.load(open(B.parent/f'source/hands/right/{f:06d}.json'))
 rows.append(dict(frame_idx=f,timestamp_s=float(times[f]),track_id=3,is_right=True,original_direct_observation=bool(direct[f,1]),evidence_kind=evidence['rows'][f]['kind'],identity_anchor_frames=identity['identity_anchor_frames'],keypoints=kp.tolist(),crop=dict(cx=float(centre[0]),cy=float(centre[1]),size=size),fixed_betas=prior['mano']['betas'],fixed_hand_scale=prior['hand_scale'],image=str(path),image_sha256=digest(path)))
for cap in caps:cap.release()
request=dict(schema=SCHEMA,source_hand='right',camera=dict(width=1280,height=800,fx=float(K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2])),evidence_receipt=str(E/'report.json'),evidence_sha256=digest(E/'report.json'),evidence_arrays=str(E/'image-observations.npz'),evidence_arrays_sha256=digest(E/'image-observations.npz'),canonical_bundle=str(B),canonical_bundle_sha256=digest(B),identity_receipt_sha256=digest(S/'e037-source-hand-image-motion-v3/report.json'),source_videos_sha256={str(v):digest(v) for v in V},frames=rows)
p=O/'request.json';p.write_text(json.dumps(request,indent=2)+'\n');validate_request(p)
report=dict(schema='v2d.prepared-fresh-hand-probe.v1',producer_sha256=digest(Path(__file__)),recovery_code_sha256=digest(R/'scripts/track3/recover_source_hand.py'),request_sha256=digest(p),frames=selected,requests_validated=True,private_source_images=True,new_gpu_work=False,new_training_steps=0,sam2_private_weights_sha256=digest(S/'models/sam2.1_hiera_large.pt'),hamer_private_checkpoint_sha256=digest(S/'models/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt'),queued=False,blocked='Brev shutdown control must be repaired and independently tested before any START; source priors must pass unchanged 1mm native contact plus source/motion gates before PPO.')
(O/'preparation-receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

"""Score fixed geometry hypotheses on other source frames, before more GPU work."""
from pathlib import Path
import json,os,sys,time,hashlib
import cv2,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');OUT=S/'metric-scale-sequence-check-v2';OUT.mkdir();sys.path[:0]=[str(R/'scripts/track3'),str(R/'reconstruction/modules/v2d_stereo_depth/lib')]
from stereo_depth import load_rig,stereo_depth
from select_pose_hypothesis import observed_fit
from foundationpose_track import pose_rejection
from camera_prior import transport
rig=load_rig(str(S/'dataset/track_3/public/meta/camera_calibration.json'));K=rig.P1[:3,:3];rows=[];started=time.monotonic()
for ep,cid in [(33,13),(35,13),(37,14)]:
 root=S/f'scored-cohort-{cid:02d}-results/reconstruction-e{ep:03d}';bundle=root/f'bundles/e{ep:03d}';geo=root/f'geometry/e{ep:03d}/wooden_spoon';candidate=S/f'metric-scale-source-probes-v1/e{ep:03d}';report=json.loads((candidate/'candidate-report.json').read_text());raw=json.loads((geo/'manifest.json').read_text());old=trimesh.load(geo/'mesh.glb',force='mesh',process=False);new=trimesh.load(candidate/'mesh.glb',force='mesh',process=False)
 with np.load(bundle/'episode.npz',allow_pickle=False) as b:camera=b['camera_c2w'].copy();axis=b['frame_indices'].copy();np.testing.assert_allclose(K,b['intrinsics'],atol=1e-6)
 correction=np.eye(4);correction[:3,3]=np.asarray(raw['seed_camera_from_object'])[:3,:3].T@np.asarray(report['anchor_translation_delta_m'])
 pose_root=root/f'source/e{ep:03d}/poses/e{ep:03d}/ego_object_poses/e{ep:03d}/wooden_spoon/object_to_cam/ego_cam_c'
 def pose(frame):
  p=pose_root/f'{frame:06d}.json'
  if not p.is_file():return None
  j=json.loads(p.read_text());T=np.eye(4);T[:3,:3]=Rotation.from_quat(np.asarray(j['rotation'])[[1,2,3,0]]).as_matrix();T[:3,3]=j['translation'];return T
 caps=[cv2.VideoCapture(str(S/f'dataset/track_3/evaluation/videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4')) for cam in [rig.left,rig.right]]
 tree_old=cKDTree(old.vertices);tree_new=cKDTree(new.vertices)
 for frame in np.linspace(0,len(axis)-1,7,dtype=int).tolist():
  if time.monotonic()-started>240:raise TimeoutError('CPU source comparison deadline')
  T=pose(frame)
  if T is None:rows.append(dict(episode=ep,frame=frame,source_pose_missing=True));continue
  images=[]
  for cap in caps:cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();assert ok;images.append(image)
  depth,valid=stereo_depth(rig,*images);mask=cv2.imread(str(root/f'source/e{ep:03d}/masks/wooden_spoon/{frame:06d}.png'),0);assert mask is not None
  previous=pose(frame-1) if frame else None
  prior=transport(previous,np.linalg.inv(camera[frame-1]),np.linalg.inv(camera[frame])) if previous is not None else None
  fits={}
  for label,mesh,tree,pose_now,pose_prior in [('control',old,tree_old,T,prior),('scale-only',new,tree_new,T,prior),('scale-and-fixed-body-offset',new,tree_new,T@correction,None if prior is None else prior@correction)]:
   fit=observed_fit(np.asarray(mesh.vertices),np.asarray(mesh.faces),tree,pose_now,depth,valid,mask>127,K);rejection,support=pose_rejection(pose_now,depth,valid,mask,K,mesh.bounds,True,pose_prior);fits[label]=dict(fit=fit,original_depth_and_jump_rejection=rejection)
  rows.append(dict(episode=ep,frame=frame,seed_frame=raw['seed_frame'],seed_frame_reused=frame==raw['seed_frame'],fits=fits));print(ep,frame,{k:round(v['fit']['quality'],4) for k,v in fits.items()},flush=True)
 for cap in caps:cap.release()
report=dict(schema='v2d.metric-scale-other-source-frame-check.v1',rows=rows,fixed_seed_fit_parameters=True,frame_sample_rule='Seven uniform full-sequence source indices, same for every case',baseline_geometry_or_poses_changed=False,scored_reserved_episodes_tuned=False,original_depth_or_jump_gates_relaxed=False,physics_stepped=False,trajectory_promoted=False,new_gpu_lease_started=False,physical_task_accepted=False,wall_seconds=time.monotonic()-started,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')

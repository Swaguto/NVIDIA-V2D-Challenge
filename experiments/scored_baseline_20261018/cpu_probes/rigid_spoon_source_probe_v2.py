"""CPU check for the diagnosed rest/return failures; all controls unchanged."""
from pathlib import Path
import hashlib,json,sys,time
import cv2,numpy as np,trimesh
from scipy.spatial.transform import Rotation
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');ROOT=S/'scored-cohort-14-results/reconstruction-e037';OUT=S/'rigid-spoon-source-probe-v2';OUT.mkdir();sys.path[:0]=[str(R/'scripts/track3'),str(R/'reconstruction/modules/v2d_stereo_depth/lib')]
from stereo_depth import load_rig,stereo_depth
from measured_rigid_pose_probe_v2 import refine
from measured_pose_refinement import finite_support_overlap
from foundationpose_track import pose_rejection
from camera_prior import transport
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rig=load_rig(str(S/'dataset/track_3/public/meta/camera_calibration.json'));K=rig.P1[:3,:3]
with np.load(ROOT/'bundles/e037/episode.npz',allow_pickle=False) as f:camera=f['camera_c2w'].copy();np.testing.assert_allclose(K,f['intrinsics'],atol=1e-6)
world=np.asarray(json.loads((ROOT/'support/e037/report.json').read_text())['plane_world'])
with np.load(ROOT/'support/e037/inferred-support-sheet.npz',allow_pickle=False) as f:sheet=f['vertices_world_m'].copy()
path=ROOT/'geometry/e037/wooden_spoon/mesh.glb';mesh=trimesh.load(path,force='mesh',process=False);base_vertices=np.asarray(mesh.vertices);source_hash=sha(path)
poses=ROOT/'source/e037/poses/e037/ego_object_poses/e037/wooden_spoon/object_to_cam/ego_cam_c'
def load(frame):
 j=json.loads((poses/f'{frame:06d}.json').read_text());T=np.eye(4);T[:3,:3]=Rotation.from_quat(np.asarray(j['rotation'])[[1,2,3,0]]).as_matrix();T[:3,3]=j['translation'];return T
caps=[cv2.VideoCapture(str(S/f'dataset/track_3/evaluation/videos/chunk-000/observation.images.{cam}/episode_000037.mp4')) for cam in [rig.left,rig.right]];r=dict(schema='v2d.measured-rigid-pose-source-probe.v1',episode=37,rows=[],source_mesh_sha256=source_hash,source_inputs_modified=False,original_acceptance_gates_relaxed=False,new_gpu_lease_started=False,physics_stepped=False,trajectory_promoted=False,physical_task_accepted=False,producer_sha256=sha(Path(__file__)),refiner_sha256=sha(Path(__file__).with_name('measured_rigid_pose_probe_v2.py')));started=time.monotonic()
for frame in [0,100,125]:
 if time.monotonic()-started>900:raise TimeoutError('CPU 6DoF probe deadline')
 images=[]
 for cap in caps:cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();assert ok;images.append(image)
 depth,valid=stereo_depth(rig,*images);mask=cv2.imread(str(ROOT/f'source/e037/masks/wooden_spoon/{frame:06d}.png'),0)>127;P=camera[frame];plane=np.r_[P[:3,:3].T.dot(world[:3]),world[3]+np.dot(world[:3],P[:3,3])];footprint=np.einsum('ni,ij->nj',sheet-P[:3,3],P[:3,:3],optimize=False);T=load(frame);prior=transport(load(frame-1),np.linalg.inv(camera[frame-1]),np.linalg.inv(P)) if frame else None
 for label,scale in [('control-shape',1.),('uniform-1.3-shape',1.3)]:
  vertices=base_vertices*scale;candidate,fit=refine(vertices,np.asarray(mesh.faces),T,depth,valid,mask,K,plane,footprint);rejection,support=pose_rejection(candidate,depth,valid,(mask*255).astype(np.uint8),K,np.stack([vertices.min(0),vertices.max(0)]),True,prior);row=dict(frame=frame,arm=label,fit=fit,original_gate_rejection=rejection,finite_overlap_before=finite_support_overlap(vertices,np.asarray(mesh.faces),T,plane,footprint),finite_overlap_after=finite_support_overlap(vertices,np.asarray(mesh.faces),candidate,plane,footprint));r['rows'].append(row);(OUT/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(frame=frame,arm=label,fit_accepted=fit['accepted'],original_gate_passed=rejection is None,height_before_m=row['finite_overlap_before']['minimum_height_m'] if row['finite_overlap_before'] else None,height_after_m=row['finite_overlap_after']['minimum_height_m'] if row['finite_overlap_after'] else None)),flush=True)
for cap in caps:cap.release()
assert sha(path)==source_hash;r['wall_seconds']=time.monotonic()-started;(OUT/'report.json').write_text(json.dumps(r,indent=2)+'\n')

"""Source-bound finite-table pose fitting probe; raw depth/masks/poses unchanged."""
from pathlib import Path
import hashlib,json,os,sys,time
import cv2,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
S=Path('/Users/rohithiruku/v2d_scored_baseline_20261017');R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');ROOT=S/'scored-cohort-14-results/reconstruction-e037';OUT=S/'spoon-support-fit-probe-v1';OUT.mkdir();os.environ['V2D_CALIB']=str(S/'dataset/track_3/public/meta/camera_calibration.json');sys.path.insert(0,str(R/'scripts/track3'));sys.path.insert(0,str(R/'reconstruction/modules/v2d_stereo_depth/lib'))
from stereo_depth import load_rig,stereo_depth
from measured_pose_refinement import refine_translation,finite_support_overlap
from select_pose_hypothesis import observed_fit
from foundationpose_track import pose_rejection
from camera_prior import load_sequence,transport
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rig=load_rig();K=np.asarray(rig.P1[:3,:3]);ep=37;obj='wooden_spoon';bundle=ROOT/'bundles/e037';geo=ROOT/'geometry/e037'/obj;mesh=trimesh.load(S/'metric-scale-source-probes-v1/e037/mesh.glb',force='mesh',process=False);vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces);tree=cKDTree(vertices)
with np.load(bundle/'episode.npz',allow_pickle=False) as f:c2w=f['camera_c2w'];np.testing.assert_allclose(K,f['intrinsics'],atol=1e-6,rtol=0)
cameras,camera_sha=load_sequence(ROOT/'source/e037/camera/ego_cam_c',37,len(c2w));np.testing.assert_allclose(np.linalg.inv(cameras),c2w,atol=1e-6,rtol=0)
plane_report=ROOT/'support/e037/report.json';world_plane=np.asarray(json.loads(plane_report.read_text())['plane_world'])
with np.load(ROOT/'support/e037/inferred-support-sheet.npz',allow_pickle=False) as f:world_sheet=f['vertices_world_m']
pose_root=ROOT/f'source/e037/poses/e037/ego_object_poses/e037/{obj}/object_to_cam/ego_cam_c';mask_root=ROOT/f'source/e037/masks/{obj}'
def read_pose(frame):
 p=pose_root/f'{frame:06d}.json';j=json.loads(p.read_text());q=np.asarray(j['rotation']);T=np.eye(4);T[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();T[:3,3]=j['translation'];return T
caps=[cv2.VideoCapture(str(S/f'dataset/track_3/evaluation/videos/chunk-000/observation.images.{camera}/episode_000037.mp4')) for camera in [rig.left,rig.right]];r=dict(schema='v2d.finite-support-object-pose-cpu-probe.v1',episode=37,object=obj,source_mesh_sha256=sha(geo/'mesh.glb'),bundle_payload_sha256=sha(bundle/'episode.npz'),support_report_sha256=sha(plane_report),camera_sequence_sha256=camera_sha,source_inputs_changed=False,shape_rotation_and_scale_changed=True,uniform_spoon_scale=1.3,geometry_candidate_not_promoted=True,plane_depth_exclusion_band_m=.007,original_acceptance_gates_relaxed=False,depth_recomputed_from_identical_local_source_videos=True,physical_task_accepted=False,trajectory_promoted=False,rows=[],producer_sha256=sha(Path(__file__)))
start=time.monotonic()
for frame in [0,25,50,75,100,125,150]:
 images=[]
 for cap in caps:cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();assert ok;images.append(image)
 depth,valid=stereo_depth(rig,*images);mask=cv2.imread(str(mask_root/f'{frame:06d}.png'),0);assert mask is not None;pose=read_pose(frame);P=c2w[frame];n=np.einsum('ji,j->i',P[:3,:3],world_plane[:3],optimize=False);plane=np.r_[n,world_plane[3]+np.dot(world_plane[:3],P[:3,3])];footprint=np.einsum('ni,ij->nj',world_sheet-P[:3,3],P[:3,:3],optimize=False)
 y,x=np.indices(depth.shape);points=np.stack([(x-K[0,2])*depth/K[0,0],(y-K[1,2])*depth/K[1,1],depth],axis=-1);heights=np.einsum('nmi,i->nm',points,n,optimize=False)+plane[3];filtered=valid&(heights>.007)
 old_height=float(np.min(np.einsum('ni,i->n',np.einsum('ni,ji->nj',vertices,pose[:3,:3],optimize=False)+pose[:3,3],n,optimize=False)+plane[3]));prior=transport(read_pose(frame-1),cameras[frame-1],cameras[frame]) if frame and (pose_root/f'{frame-1:06d}.json').is_file() else None
 for label,fit_valid in [('full-measured-foreground',valid),('above-measured-support',filtered)]:
  candidate,fit=refine_translation(vertices,faces,pose,depth,fit_valid,mask>127,K,plane,footprint,partial_support=True)
  rejection,support=pose_rejection(candidate,depth,valid,mask,K,mesh.bounds,True,prior)
  before=observed_fit(vertices,faces,tree,pose,depth,valid,mask>127,K);after=observed_fit(vertices,faces,tree,candidate,depth,valid,mask>127,K)
  height=float(np.min(np.einsum('ni,i->n',np.einsum('ni,ji->nj',vertices,candidate[:3,:3],optimize=False)+candidate[:3,3],n,optimize=False)+plane[3]));row=dict(frame=frame,arm=label,fit=fit,original_full_depth_and_jump_gate_rejection=rejection,original_depth_support=support,full_source_fit_before=before,full_source_fit_after=after,minimum_support_height_before_m=old_height,minimum_support_height_after_m=height,fit_foreground_depth_pixels=int(((mask>127)&fit_valid).sum()),original_foreground_depth_pixels=int(((mask>127)&valid).sum()),candidate_camera_from_object=candidate.tolist())
  row['finite_overlap_before']=finite_support_overlap(vertices,faces,pose,plane,footprint);row['finite_overlap_after']=finite_support_overlap(vertices,faces,candidate,plane,footprint);r['rows'].append(row);(OUT/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(frame=frame,arm=label,fit_accepted=fit['accepted'],finite_support_applied=fit['support_constraint_applied'],height_before_m=old_height,height_after_m=height,original_gate_passed=rejection is None,full_iou_before=before.get('silhouette_iou'),full_iou_after=after.get('silhouette_iou'))),flush=True)
for cap in caps:cap.release()
r['wall_seconds']=time.monotonic()-start;(OUT/'report.json').write_text(json.dumps(r,indent=2)+'\n')

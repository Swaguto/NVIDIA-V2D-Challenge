"""Register one inferred hand with the frozen own-object proxy; no physics or contact labels."""
from pathlib import Path
import datetime,hashlib,json,sys
S=Path(__file__).resolve().parents[2];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path[:0]=[str(S/'mediapipe-cpu-deps-v1'),str(R/'scripts/track3'),str(R/'scripts/dex3_training')]
import cv2,mujoco,numpy as np,trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from source_hand_depth import front_surface_depth
from audit_hand_reference_contacts import canonical_id
cv2.setNumThreads(1);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();O=S/'e037-source-hand-object-registration-v1';O.mkdir(exist_ok=True)
P=S/'plane-aligned-native-reference-v1/frozen';manifest=json.loads((P/'manifest.json').read_text())
for name,expected in manifest['files'].items():assert sha(P/name)==expected,name
latest=S/'plane-aligned-articulated-native-validation-v2/frozen';latest_manifest=json.loads((latest/'manifest.json').read_text());assert sha(latest/'targets.npz')==latest_manifest['files']['targets.npz']
with np.load(P/'targets.npz') as z:original_object=z['object_poses'].copy()
with np.load(latest/'targets.npz') as z:object_poses=z['object_poses'].copy()
assert np.array_equal(object_poses,original_object)
with np.load(P/'aligned-source-reference.npz') as z:camera=z['camera_c2w'][50].copy()
stage=np.asarray(json.loads((P/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:K=z['intrinsics'].copy()
mp=S/'plane-aligned-native-reference-v1/native-model.mjb';model=mujoco.MjModel.from_binary_path(str(mp));data=mujoco.MjData(model);bodies=[model.body(i).name for i in range(model.nbody)];bid=canonical_id(bodies,'wooden_spoon_object');adr=int(model.jnt_qposadr[model.body_jntadr[bid]]);pose=object_poses[50,0];data.qpos[:]=model.qpos0;data.qpos[adr:adr+3]=pose[:3];data.qpos[adr+3:adr+7]=pose[[6,3,4,5]];mujoco.mj_fwdPosition(model,data)
assert np.max(abs(data.xpos[bid]-pose[:3]))<1e-12;assert np.max(abs(data.xmat[bid].reshape(3,3)-Rotation.from_quat(pose[3:]).as_matrix()))<1e-12
def camera_points(world):
 y_world=np.einsum('ni,ij->nj',world,stage)
 return np.einsum('ni,ij->nj',y_world-camera[:3,3],camera[:3,:3])
def native_points(cam):
 y_world=np.einsum('ni,ji->nj',cam,camera[:3,:3])+camera[:3,3]
 return np.einsum('ni,ji->nj',y_world,stage)
parts=[];all_vertices=[];all_faces=[];offset=0;max_closure=0.
for g in np.flatnonzero(model.geom_bodyid==bid):
 assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
 mid=int(model.geom_dataid[g]);va=int(model.mesh_vertadr[mid]);nv=int(model.mesh_vertnum[mid]);fa=int(model.mesh_faceadr[mid]);nf=int(model.mesh_facenum[mid]);local=model.mesh_vert[va:va+nv].astype(float);faces=model.mesh_face[fa:fa+nf].astype(int)
 world=np.einsum('ni,ji->nj',local,data.geom_xmat[g].reshape(3,3))+data.geom_xpos[g];verts=camera_points(world);max_closure=max(max_closure,float(abs(native_points(verts)-world).max()));parts.append(verts);all_vertices.append(verts);all_faces.append(faces+offset);offset+=len(verts)
vertices=np.vstack(all_vertices);faces=np.vstack(all_faces);assert max_closure<1e-10 and (vertices[:,2]>0).all()
fit=S/'e037-native-RGB-hard-skin-coverage-fit-v2/source-hand-candidate.npz'
with np.load(fit) as z:
 with np.load(B) as b:old_camera=b['camera_c2w'][50].copy()
 hand_v=np.einsum('ni,ij->nj',z['vertices_world'][0]-old_camera[:3,3],old_camera[:3,:3]);hand_j=np.einsum('ni,ij->nj',z['joints_world'][0]-old_camera[:3,3],old_camera[:3,:3]);px=z['heldout_pixels'].copy();observed_depth=z['heldout_estimated_source_depth_m'].copy();hand_front=z['heldout_front_depth_m'].copy()
object_front=front_surface_depth(vertices,faces,px,K);cloud_z=observed_depth;points=np.c_[(px[:,0]-K[0,2])*cloud_z/K[0,0],(px[:,1]-K[1,2])*cloud_z/K[1,1],cloud_z]
object_mesh=trimesh.Trimesh(vertices,faces,process=False);selected=np.vstack([hand_j[[0,4,8,12,16,20]],points]);closest,dist,_=trimesh.proximity.closest_point_naive(object_mesh,selected)
inside=np.zeros(len(selected),bool)
for verts in parts:
 hull=ConvexHull(verts);inside|=(np.einsum('ni,ki->nk',selected,hull.equations[:,:3])+hull.equations[:,3]<=1e-10).all(1)
tip_rows=[dict(joint_id=int(joint),point_is_estimated=True,nearest_object_surface_distance_m=float(dist[i]),inside_estimated_native_object_proxy=bool(inside[i]),closest_object_point_C_m=closest[i].tolist()) for i,joint in enumerate([0,4,8,12,16,20])]
missing=np.flatnonzero(~np.isfinite(hand_front));rows=[]
for n in missing:
 rows.append(dict(heldout_index=int(n),source_C_pixel=px[n].tolist(),estimated_source_skin_depth_m=float(observed_depth[n]),object_proxy_first_depth_m=float(object_front[n]) if np.isfinite(object_front[n]) else None,object_proxy_minus_estimated_skin_depth_m=float(object_front[n]-observed_depth[n]) if np.isfinite(object_front[n]) else None,point_to_object_proxy_distance_m=float(dist[6+n]),inside_estimated_object_proxy=bool(inside[6+n])))
uv=np.c_[K[0,0]*vertices[:,0]/vertices[:,2]+K[0,2],K[1,1]*vertices[:,1]/vertices[:,2]+K[1,2]];image=cv2.imread(str(S/'e037-fresh-source-hand-request-v1/images/000050.png'));silhouette=np.zeros(image.shape[:2],np.uint8)
for tri in faces:cv2.fillConvexPoly(silhouette,np.rint(uv[tri]).astype(int),255)
contours,_=cv2.findContours(silhouette,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(image,contours,-1,(0,180,255),1)
for n in missing:cv2.circle(image,tuple(map(int,px[n])),4,(0,0,255),1)
cv2.imwrite(str(O/'source-with-native-object-proxy-outline.png'),image)
mask_path=S/'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/000050.png';source_mask=cv2.imread(str(mask_path),0)>0;overlap=int((source_mask&(silhouette>0)).sum())
report=dict(schema='v2d.source-hand-own-object-registration-audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),source_hashes={str(p.relative_to(S)):sha(p) for p in [P/'manifest.json',P/'aligned-source-reference.npz',latest/'manifest.json',mp,fit,B,mask_path]},frame=50,own_reconstructed_object_proxy=True,native_mesh_hulls=len(parts),source_to_native_roundtrip_max_error_m=max_closure,reference_object_pose_matches_frozen_native_body=True,object_reference_targets_not_modified=True,proxy_outline_covers_inferred_object_mask_fraction=overlap/int(source_mask.sum()),source_object_mask_pixels=int(source_mask.sum()),occluded_object_proxy_silhouette_is_not_visible_segmentation=True,inferred_hand_points=tip_rows,missing_heldout_skin_rays=rows,physical_contact_or_grasp_not_established=True,source_hand_validity_flags_not_upgraded=True,geometry_distances_are_not_measured_contacts=True,physics_stepped=False,new_GPU_inference=False,new_PPO_steps=0,required_native_reference_contact_gate_m=.001,full_source_reference_qualified=False,physical_task_accepted=False)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(O/'registered-proxy-and-source-samples.npz',object_vertices_C=vertices,object_faces=faces,hand_vertices_C=hand_v,hand_joints_C=hand_j,heldout_pixels=px,estimated_skin_depth_m=observed_depth,object_front_depth_m=object_front);print(json.dumps(report,indent=2))

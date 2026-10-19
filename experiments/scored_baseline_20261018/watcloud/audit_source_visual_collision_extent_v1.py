"""Separate a source-shape/pose problem from collision truncation using frozen own assets."""
from pathlib import Path
import datetime,hashlib,json,sys
S=Path(__file__).resolve().parents[2];sys.path.insert(0,str(S/'mediapipe-cpu-deps-v1'))
import cv2,numpy as np,trimesh
from scipy.spatial.transform import Rotation
cv2.setNumThreads(1);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();O=S/'e037-source-visual-collision-extent-v1';O.mkdir()
P=S/'plane-aligned-native-reference-v1/frozen';m=json.loads((P/'manifest.json').read_text());obj=m['objects'][0];mesh_path=P/obj['mesh'];assert sha(mesh_path)==m['files'][obj['mesh']]
mesh=trimesh.load(mesh_path,process=False,force='mesh');assert isinstance(mesh,trimesh.Trimesh)
with np.load(P/'targets.npz') as z:pose=z['object_poses'][50,0].copy()
with np.load(P/'aligned-source-reference.npz') as z:camera=z['camera_c2w'][50].copy()
stage=np.asarray(json.loads((P/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:K=z['intrinsics'].copy()
world=np.einsum('ni,ji->nj',mesh.vertices,Rotation.from_quat(pose[3:]).as_matrix())+pose[:3];v=np.einsum('ni,ij->nj',np.einsum('ni,ij->nj',world,stage)-camera[:3,3],camera[:3,:3]);assert (v[:,2]>0).all()
proxy_path=S/'e037-source-hand-object-registration-v1/registered-proxy-and-source-samples.npz'
with np.load(proxy_path) as z:proxy=z['object_vertices_C'].copy();proxy_faces=z['object_faces'].copy()
def project(points):return np.c_[K[0,0]*points[:,0]/points[:,2]+K[0,2],K[1,1]*points[:,1]/points[:,2]+K[1,2]]
source_image=S/'e037-fresh-source-hand-request-v1/images/000050.png';image=cv2.imread(str(source_image));masks=[]
for verts,faces,colour in [(v,mesh.faces,(255,0,255)),(proxy,proxy_faces,(0,180,255))]:
 uv=project(verts);mask=np.zeros(image.shape[:2],np.uint8)
 for tri in faces:cv2.fillConvexPoly(mask,np.rint(uv[tri]).astype(int),255)
 contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(image,contours,-1,colour,1);masks.append(mask>0)
cv2.imwrite(str(O/'source-visual-magenta-collision-yellow.png'),image)
source_mask_path=S/'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/000050.png';source_mask=cv2.imread(str(source_mask_path),0)>0
visual_uv,proxy_uv=project(v),project(proxy);rows=[]
for name,uv,mask in [('reconstructed_eval_mesh',visual_uv,masks[0]),('native_collision_proxy',proxy_uv,masks[1])]:rows.append(dict(asset=name,projected_vertex_bounds_px=[uv.min(0).tolist(),uv.max(0).tolist()],inferred_object_mask_covered_fraction=float((source_mask&mask).sum()/source_mask.sum())))
report=dict(schema='v2d.own-visual-vs-native-collision-source-extent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),source_hashes={str(p.relative_to(S)):sha(p) for p in [P/'manifest.json',mesh_path,proxy_path,B,source_image,source_mask_path]},frame=50,source_object_reference_pose_unchanged=True,visual_collision_projection_IoU=float((masks[0]&masks[1]).sum()/(masks[0]|masks[1]).sum()),projected_visual_collision_bounds_max_delta_px=float(abs(np.array([visual_uv.min(0),visual_uv.max(0)])-np.array([proxy_uv.min(0),proxy_uv.max(0)])).max()),assets=rows,provided_or_hidden_mesh_read=False,reconstruction_or_collision_assets_modified=False,source_mask_is_inferred_and_includes_uncertain_regions=True,source_geometry_or_pose_error_not_causally_separated=True,physical_task_accepted=False,new_GPU_work=False,new_PPO_steps=0,required_native_reference_contact_gate_m=.001)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

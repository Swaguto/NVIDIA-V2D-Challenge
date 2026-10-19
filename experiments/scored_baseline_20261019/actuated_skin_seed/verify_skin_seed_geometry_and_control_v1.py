"""Independent tensor closure for new geometry plus cached positive-control eligibility.

No site generation, old source fit, native forward or load solve is repeated.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(S/'compute'))
import mujoco,numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from numpy_mano_source_geometry_v2 import NumpyMano
O=S/'e037-skin-seed-independent-closure-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=S/'e037-motion-constrained-wrist-native-v1';mp=P/'native-model.mjb';model=mujoco.MjModel.from_binary_path(str(mp));names=[model.body(i).name for i in range(model.nbody)];bid=next(i for i,n in enumerate(names) if n.endswith('wooden_spoon_object'));report_path=S/'e037-skin-associated-wrench-seed-v1/report.json';source=json.loads(report_path.read_text());gp=S/'e037-skin-associated-wrench-seed-v1/seed-geometry.npz';plan=json.loads((S/'compute/skin-associated-motor-seed-CPU-plan-v1.json').read_text())
with np.load(P/'frozen/targets.npz') as z:obj=z['object_poses'][50,0].copy()
objR=Rotation.from_quat(obj[3:]).as_matrix();tri=[];planes=[]
for g in np.flatnonzero(model.geom_bodyid==bid):
 mid=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);v=np.asarray(model.mesh_vert[a:a+n],float);G=Rotation.from_quat(model.geom_quat[g][[1,2,3,0]]).as_matrix();body=np.einsum('ni,ji->nj',v,G,optimize=False)+model.geom_pos[g];world=np.einsum('ni,ji->nj',body,objR,optimize=False)+obj[:3];assert np.isfinite(world).all();h=ConvexHull(world);tri.extend(world[h.simplices]);planes.append(h.equations)
with np.load(gp) as z:
 saved=z['object_triangles_world'].copy();assert np.isfinite(saved).all()
 # Triangle order may change under roundoff; compare the entire object vertex set.
 from scipy.spatial import cKDTree
 error=max(cKDTree(np.asarray(tri).reshape(-1,3)).query(saved.reshape(-1,3))[0].max(),cKDTree(saved.reshape(-1,3)).query(np.asarray(tri).reshape(-1,3))[0].max());assert error<1e-10,error
orientation=S/'e037-source-wrist-convention-audit-v1/saved-palm-orientation-diagnostics.npz'
with np.load(orientation) as z:NC=z['native_from_C'].copy()
with np.load(P/'frozen/aligned-source-reference.npz') as z:camera=z['camera_c2w'][50].copy()
stage=np.asarray(json.loads((P/'frozen/command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal']);origin=np.einsum('ij,j->i',stage,camera[:3,3],optimize=False)
bundle=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(bundle) as z:oldcam=z['camera_c2w'][50].copy()
mano_path=S/'models/mano/models/MANO_RIGHT.pkl';mano=NumpyMano(mano_path);faces=np.asarray(mano.faces,int);dominant=np.asarray(mano.weights).argmax(1);groups={'thumb':[13,14,15],'index':[1,2,3],'middle':[4,5,6]}
control_path=S/'e037-source-patch-wrench-sites-v1/report.json';control=json.loads(control_path.read_text());audit_path=S/'e037-wrench-site-source-skin-v1/report.json';audit=json.loads(audit_path.read_text());paths=[S/'e037-native-RGB-hand-source-audit-v1/unfitted-source-geometry.npz',S/'e037-native-RGB-multiview-source-fit-v1/source-hand-candidate.npz',S/'e037-native-RGB-hard-skin-coverage-fit-v2/source-hand-candidate.npz'];rows=[];centroid_error=0.
for i,(case,path) in enumerate(zip(source['rows'],paths,strict=True)):
 with np.load(path) as z:C=z['vertices_camera_C'].copy() if i==0 else np.einsum('ni,ij->nj',z['vertices_world'][0]-oldcam[:3,3],oldcam[:3,:3],optimize=False)
 world=np.einsum('ni,ji->nj',C,NC,optimize=False)+origin;skin=world[faces];centers=skin.mean(1);sn=np.cross(skin[:,1]-skin[:,0],skin[:,2]-skin[:,0]);sn/=np.linalg.norm(sn,axis=1)[:,None]
 for pool in case['sampled_skin_sites'].values():
  for p in pool:
   e=float(np.abs(centers[p['source_skin_face_id']]-p['source_skin_centroid']).max());centroid_error=max(centroid_error,e);assert e<1e-10
 # Bind the retained load-balanced point to its nearest actual finger-associated
 # skin triangle, independently of the previous nearest-joint pool selection.
 seed=control['rows'][i]['proposed_site_seed'];checks=[]
 for digit,segments in groups.items():
  point=np.asarray(seed['sites'][digit]['point']);normal=np.asarray(seed['sites'][digit]['normal']);ids=np.flatnonzero(np.isin(dominant[faces],segments).all(1)&(dominant[faces]==segments[-1]).any(1))
  from trimesh.triangles import closest_point
  closest=closest_point(skin[ids],np.broadcast_to(point,(len(ids),3)));distance=np.linalg.norm(closest-point,axis=1);k=int(ids[int(distance.argmin())]);gap=float(np.linalg.norm(point-centers[k]));penetration=max([max(0,-float((np.einsum('ij,j->i',eq[:,:3],centers[k],optimize=False)+eq[:,3]).max())) for eq in planes]);dot=float(np.dot(sn[k],normal));eligible=bool(gap<=plan['maximum_skin_to_site_distance_m'] and penetration<=plan['maximum_source_skin_model_penetration_m'] and dot<=plan['maximum_skin_object_normal_dot'])
  checks.append(dict(digit=digit,source_face_id=k,source_face_centroid_world_m=centers[k].tolist(),source_face_outward_world=sn[k].tolist(),site_to_source_face_centroid_m=gap,source_skin_model_penetration_m=penetration,source_skin_object_normal_dot=dot,eligible_under_unchanged_new_skin_plan=eligible))
 rows.append(dict(source_label=case['source_label'],all_three_cached_sites_source_skin_eligible=all(c['eligible_under_unchanged_new_skin_plan'] for c in checks),checks=checks,no_source_pose_or_contact_truth_qualification=True))
r=dict(schema='v2d.skin-seed-independent-closure-and-control.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),inputs_sha256={str(p):sha(p) for p in [report_path,gp,control_path,audit_path,mp,mano_path]+paths},object_vertex_set_explicit_tensor_closure_max_error_m=float(error),selected_source_centroid_explicit_tensor_closure_max_error_m=centroid_error,initial_numpy_matmul_warnings_retained_and_cause_unproven=True,all_saved_geometry_finite=True,rows=rows,no_source_fit_or_generation_or_native_forward_or_load_solver_repeated=True,source_flags_and_heldout_failures_unchanged=True,new_GPU_work=False,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)

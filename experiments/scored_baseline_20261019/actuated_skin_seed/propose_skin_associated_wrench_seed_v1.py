"""NEW skin-associated object sites and wrench seed, not a source/contact certificate.

No old projection-family replay: sites originate at finger-associated MANO skin
triangles within a declared distance, with opposed surface normals. Raw source
separation cannot be erased by projection. Native geometry is read only.
"""
from pathlib import Path
import datetime,hashlib,itertools,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path[:0]=[str(S/'compute'),str(R/'scripts/dex3_training')]
import mujoco,numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from trimesh.triangles import closest_point
from numpy_mano_source_geometry_v2 import NumpyMano
from v2d_dex3_flashchord.contact_load_feasibility import ContactGeometry,outer_contact_load_feasibility
O=S/'e037-skin-associated-wrench-seed-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();plan_path=S/'compute/skin-associated-motor-seed-CPU-plan-v1.json';plan=json.loads(plan_path.read_text())
P=S/'e037-motion-constrained-wrist-native-v1';f=P/'frozen';m=json.loads((f/'manifest.json').read_text())
for name,h in m['files'].items():assert sha(f/name)==h
meta_path=S/'compute/palm-wrench-library-compatibility-inputs-v1.json';meta=json.loads(meta_path.read_text());load_path=Path(meta['source_report']);assert sha(load_path)==meta['source_report_sha256'];load_report=json.loads(load_path.read_text())
mp=P/'native-model.mjb';assert sha(mp)==meta['native_model_sha256'];model=mujoco.MjModel.from_binary_path(str(mp));names=[model.body(i).name for i in range(model.nbody)];bid=next(i for i,n in enumerate(names) if n.endswith('wooden_spoon_object'))
with np.load(f/'targets.npz') as z:obj=z['object_poses'][50,0].copy();old=z['robot_qpos'][50,1].copy()
objR=Rotation.from_quat(obj[3:]).as_matrix();tri=[];norm=[];planes=[];geom={}
def vertices(g):
 mid=int(model.geom_dataid[g]);assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH;a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);v=np.asarray(model.mesh_vert[a:a+n],float);return v@Rotation.from_quat(model.geom_quat[g][[1,2,3,0]]).as_matrix().T+model.geom_pos[g]
for g in np.flatnonzero(model.geom_bodyid==bid):
 v=vertices(g)@objR.T+obj[:3];h=ConvexHull(v);tri.extend(v[h.simplices]);norm.extend(h.equations[:,:3]);planes.append(h.equations)
tri=np.asarray(tri);norm=np.asarray(norm);geom['object_triangles_world']=tri;geom['object_outward_normals_world']=norm
for digit,last in [('thumb',2),('index',1),('middle',1)]:
 b=next(i for i,n in enumerate(names) if n.endswith(f'right_hand_{digit}_{last}_link'));ts=[];ns=[]
 for g in np.flatnonzero(model.geom_bodyid==b):v=vertices(g);h=ConvexHull(v);ts.extend(v[h.simplices]);ns.extend(h.equations[:,:3])
 geom[f'robot_{digit}_triangles_body']=np.asarray(ts);geom[f'robot_{digit}_outward_normals_body']=np.asarray(ns)
source_report_path=S/'e037-source-wrist-convention-audit-v1/report.json';source_report=json.loads(source_report_path.read_text());orientation_path=S/'e037-source-wrist-convention-audit-v1/saved-palm-orientation-diagnostics.npz'
with np.load(orientation_path) as z:NC=z['native_from_C'].copy();rootR=z['source_robot_palm_rotations_native'].copy()
with np.load(f/'aligned-source-reference.npz') as z:cam=z['camera_c2w'][50].copy();flags={k:bool(z[k][50,1]) for k in ['hand_valid','hand_directly_observed','hand_metric_valid']}
assert not any(flags.values())
stage=np.asarray(json.loads((f/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal']);origin=stage@cam[:3,3]
bundle=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(bundle) as z:oldcam=z['camera_c2w'][50].copy()
mano_path=S/'models/mano/models/MANO_RIGHT.pkl';mano=NumpyMano(mano_path);faces=np.asarray(mano.faces,int);dominant=np.asarray(mano.weights).argmax(1);groups={'thumb':[13,14,15],'index':[1,2,3],'middle':[4,5,6]}
friction=np.asarray(next(c for r in load_report['rows'] for c in r['contacts'])['friction']);loads={k:load_report['rows'][0][field] for k,field in [('gravity','gravity_load_wrench_world'),('estimated_motion','estimated_reference_motion_load_wrench_world')]}
paths=[S/'e037-native-RGB-hand-source-audit-v1/unfitted-source-geometry.npz',S/'e037-native-RGB-multiview-source-fit-v1/source-hand-candidate.npz',S/'e037-native-RGB-hard-skin-coverage-fit-v2/source-hand-candidate.npz'];labels=['raw_RGB','multiview_fit_unqualified','hard_skin_fit_failed'];rows=[]
def inward_frame(n):
 n=-np.asarray(n);t=np.cross(n,np.eye(3)[np.argmin(abs(n))]);t/=np.linalg.norm(t);return np.stack([n,t,np.cross(n,t)])
for case,(path,label) in enumerate(zip(paths,labels,strict=True)):
 with np.load(path) as z:vC=z['vertices_camera_C'].copy() if case==0 else (z['vertices_world'][0]-oldcam[:3,3])@oldcam[:3,:3]
 world=vC@NC.T+origin;skin=world[faces];centers=skin.mean(1);sn=np.cross(skin[:,1]-skin[:,0],skin[:,2]-skin[:,0]);sn/=np.linalg.norm(sn,axis=1)[:,None];pools={};counts={}
 for digit,segments in groups.items():
  ids=np.flatnonzero(np.isin(dominant[faces],segments).all(1)&(dominant[faces]==segments[-1]).any(1));pairs=[];near=0;facing=0;outside=0;min_gap=np.inf
  for face in ids:
   center=centers[face];penetration=max([max(0,-float((eq[:,:3]@center+eq[:,3]).max())) for eq in planes]);p=closest_point(tri,np.broadcast_to(center,(len(tri),3)));distance=np.linalg.norm(p-center,axis=1);min_gap=min(min_gap,float(distance.min()))
   if penetration>plan['maximum_source_skin_model_penetration_m']:continue
   external=np.ones(len(p),bool)
   for eq in planes:external &= ~((p@eq[:,:3].T+eq[:,3]).max(1)<-1e-7)
   eligible=np.flatnonzero(external&(distance<=plan['maximum_skin_to_site_distance_m']));near+=len(eligible);eligible=eligible[(norm[eligible]@sn[face])<=plan['maximum_skin_object_normal_dot']];facing+=len(eligible)
   for k in eligible:pairs.append(dict(point=p[k].tolist(),normal=norm[k].tolist(),source_skin_face_id=int(face),source_skin_centroid=centers[face].tolist(),source_skin_outward_normal=sn[face].tolist(),source_gap_m=float(distance[k]),source_skin_model_penetration_m=penetration,source_skin_object_normal_dot=float(norm[k]@sn[face]),object_triangle_id=int(k)))
  pairs.sort(key=lambda x:(x['source_gap_m'],x['object_triangle_id'],x['source_skin_face_id']));chosen=[]
  for p in pairs:
   if any(np.linalg.norm(np.asarray(p['point'])-c['point'])<.0005 and np.dot(p['normal'],c['normal'])>.99 for c in chosen):continue
   chosen.append(p)
   if len(chosen)==plan['maximum_skin_sites_per_digit']:break
  pools[digit]=chosen;counts[digit]=dict(associated_skin_faces=len(ids),minimum_face_centroid_object_gap_m=float(min_gap),within_distance_pairs=near,facing_pairs=facing,selected_sites=len(chosen))
 trials=[];feasible=[]
 for triple in itertools.product(*(pools[d] for d in groups)):
  if time.monotonic()-start>220:raise TimeoutError('New skin-site seed cap')
  dot=min(float(np.dot(triple[0]['normal'],a['normal'])) for a in triple[1:]);separation=min(np.linalg.norm(np.asarray(a['point'])-b['point']) for a,b in itertools.combinations(triple,2))
  if dot>plan['maximum_thumb_other_outward_normal_dot'] or separation<plan['minimum_site_pair_separation_m']:continue
  contacts=[ContactGeometry(a['point'],inward_frame(a['normal']),3,friction) for a in triple];fits={k:outer_contact_load_feasibility(contacts,meta['object_COM_native_world_m'],load) for k,load in loads.items()};ok=all(x['feasible_in_outer_model'] for x in fits.values())
  trials.append(dict(sites=list(triple),required_loads= fits,both_loads_outer_feasible=ok))
  if ok:feasible.append((max(x['minimum_total_normal_force_N'] for x in fits.values()),sum(x['source_gap_m']**2 for x in triple),len(trials)-1))
 root=old[:7].copy();root[:3]+=source_report['orientation_rows'][case]['estimated_human_wrist_minus_robot_root_m'];root[3:]=Rotation.from_matrix(rootR[case]).as_quat();selected=min(feasible) if feasible else None
 rows.append(dict(source_label=label,source_index=case,source_flags=flags,source_pose_not_qualified=True,root_seed_xyzw=root.tolist(),original_fingers_rad=old[7:].tolist(),skin_site_pool_counts=counts,sampled_skin_sites=pools,all_evaluated_load_trials=trials,feasible_load_trials=len(feasible),selected_trial_index=selected[2] if selected else None,selected_maximum_total_normal_force_N=selected[0] if selected else None,selection='Minimum worst-load total normal force within this finite source-skin-associated pool; no source-pose winner',status='potential_seed_only' if selected else 'rejected_no_source_skin_associated_wrench_seed'))
np.savez_compressed(O/'seed-geometry.npz',**geom)
report=dict(schema='v2d.source-skin-associated-wrench-seed.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),inputs_sha256={str(p):sha(p) for p in [mp,f/'manifest.json',f/'targets.npz',orientation_path,source_report_path,bundle,mano_path,meta_path,load_path]+paths},seed_geometry_sha256=sha(O/'seed-geometry.npz'),native_model_sha256=sha(mp),COM_world_m=meta['object_COM_native_world_m'],friction=friction.tolist(),loads=loads,rows=rows,source_hand_flags_and_heldout_failures_unchanged=True,source_skin_distances_and_normals_are_inferred_not_contact_truth=True,root_bounds_not_enlarged=True,no_source_pose_winner=True,new_native_compile_forward_or_physics_steps=False,new_source_fit_or_inference=False,old_648_projection_site_family_not_repeated=True,required_full_native_reference_contact_gate_m=.001,training_approved=False,physical_task_accepted=False,new_GPU_work=False,new_PPO_steps=0,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(rows=[{k:r[k] for k in ['source_label','skin_site_pool_counts','feasible_load_trials','selected_maximum_total_normal_force_N','status']} for r in rows],wall_seconds=report['wall_seconds']),indent=2),flush=True)

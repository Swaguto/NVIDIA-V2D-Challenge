"""Independent two-pose source/URDF/native check; no optimizer or physics step."""
from pathlib import Path
import datetime, hashlib, json, sys, time, xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1]
R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(R/'scripts/dex3_training'),str(R/'scripts/track3')]
import numpy as np, mujoco
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.kinematics import URDFKinematics
from audit_hand_reference_contacts import canonical_id
from temporal_pose_path import motion_summary
start=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
O=S/'e037-source-coupled-isolated-independent-v1'
F=S/'e037-thumb-ray-independent-validation-v1/frozen'
MP=S/'e037-motion-constrained-wrist-native-v1/native-model.mjb'
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037'
P=S/'e037-source-orientation-coupled-fingers-v1/report.json'
A=S/'e037-wrist-source-authority-v1/report.json'
assert sha(MP)=='755fc51729d4c4fc03d04108d39bec2b249074351863f096f9aa498c9ac28566'
rep=json.loads(P.read_text());authority=json.loads(A.read_text())
assert sha(F/'manifest.json')==rep['input_manifest_sha256']
mf=json.loads((F/'manifest.json').read_text())
for n,h in mf['files'].items():assert sha(F/n)==h
assert all(r['solver_success'] and r['one_knot_source_self_bounds_prerequisites_passed'] for r in rep['rows'])
with np.load(F/'targets.npz') as z:control=z['robot_qpos'].copy();objects=z['object_poses'].copy();ts=z['timestamps_s'].copy()
with np.load(F/'aligned-source-reference.npz') as z:aligned=z['robot_qpos'].copy();cams=z['camera_c2w'].copy();human=z['hand_joints_world'].copy()
with np.load(F/'source-confidence.npz') as z:flags={k:z[k].copy() for k in z.files}
with np.load(B/'episode.npz') as z:K=z['intrinsics'].copy()
sources={row['path']:row['sha256'] for row in json.loads((B/'manifest.json').read_text())['sources']}
stage=np.array(json.loads((F/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
kin=URDFKinematics(F/'robot/urdfs/g1/dex3_right.urdf')
limits={j.attrib['name']:j.find('limit').attrib for j in ET.parse(F/'robot/urdfs/g1/dex3_right.urdf').getroot().findall('joint') if j.attrib['type']!='fixed'}
lo=np.array([float(limits[n]['lower']) for n in mf['joint_names']['right']]);hi=np.array([float(limits[n]['upper']) for n in mf['joint_names']['right']]);vel=np.array([float(limits[n]['velocity']) for n in mf['joint_names']['right']])
model=mujoco.MjModel.from_binary_path(str(MP));data=mujoco.MjData(model)
bn=[model.body(i).name for i in range(model.nbody)];jn=[model.joint(i).name for i in range(model.njnt)]
wrists={s:[canonical_id(jn,f'{s}_wrist_{a}') for a in ['x','y','z','rotvec']] for s in ['left','right']}
fingers={s:[canonical_id(jn,n) for n in mf['joint_names'][s]] for s in ['left','right']}
names=['right_hand_palm_link','right_thumb_tip','right_index_tip','right_middle_tip']
native_ids=[canonical_id(bn,n) for n in names]
def native(frame,cmd):
 data.qpos[:]=model.qpos0;data.qvel[:]=0
 for oi,obj in enumerate(mf['objects']):
  b=canonical_id(bn,obj['name']+'_object');a=int(model.jnt_qposadr[model.body_jntadr[b]]);v=objects[frame,oi];data.qpos[a:a+3]=v[:3];data.qpos[a+3:a+7]=v[[6,3,4,5]]
 for si,s in enumerate(['left','right']):
  v=control[frame,si] if si==0 else cmd;w=wrists[s];data.qpos[model.jnt_qposadr[w[:3]]]=v[:3];a=int(model.jnt_qposadr[w[3]]);data.qpos[a:a+4]=v[[6,3,4,5]];data.qpos[model.jnt_qposadr[fingers[s]]]=v[7:]
 mujoco.mj_fwdPosition(model,data)
 contacts=[];target=[]
 for c in data.contact[:data.ncon]:
  a,b=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2]);ra=bn[a].startswith('dex3_right_');rb=bn[b].startswith('dex3_right_')
  row=dict(bodies=[bn[a],bn[b]],signed_gap_m=float(c.dist))
  if ra and rb:contacts.append(row)
  elif ra!=rb and not (bn[b] if ra else bn[a]).startswith('dex3_'):target.append(row)
 return data.xpos[native_ids].copy(),contacts,target
def projection_errors(frame,points):
 world=np.einsum('ni,ij->nj',points,stage,optimize=False)
 c=cams[frame];robot=np.einsum('ni,ij->nj',world-c[:3,3],c[:3,:3],optimize=False)
 observed=np.einsum('ni,ij->nj',human[frame,1,[0,4,8,12]]-c[:3,3],c[:3,:3],optimize=False)
 mask=np.isfinite(observed).all(1)&(observed[:,2]>.01)&bool(flags['hand_valid'][frame,1]);assert mask.all() and (robot[:,2]>.01).all()
 def uv(x):return np.c_[K[0,0]*x[:,0]/x[:,2]+K[0,2],K[1,1]*x[:,1]/x[:,2]+K[1,2]]
 return np.linalg.norm(uv(robot)-uv(observed),axis=1)
q=control.copy();rows=[]
for seed in rep['candidate_commands']:
 f=int(seed['frame']);cmd=np.array(seed['command']);p=B/f'source/hands/right/{f:06d}.json';assert sha(p)==sources[str(p.relative_to(B))];record=json.loads(p.read_text());assert record['directly_observed'] and record['visual_valid']
 sourceR=stage@cams[f,:3,:3]@Rotation.from_rotvec(record['mano']['global_orient']).as_matrix()@np.diag([-1.,-1.,1.])
 errors=[];closures=[];contacts=None;target=None
 for v in [control[f,1],cmd]:
  poses=kin.poses(v[:7],dict(zip(mf['joint_names']['right'],v[7:],strict=True)));points=np.stack([poses[n][:3,3] for n in names]);native_points,contacts,target=native(f,v);closures.append(float(abs(points-native_points).max()));errors.append(projection_errors(f,points))
 ori=[float(Rotation.from_matrix(sourceR@Rotation.from_quat(v[3:7]).as_matrix().T).magnitude()) for v in [control[f,1],cmd]]
 residual=Rotation.from_matrix(Rotation.from_quat(cmd[3:7]).as_matrix()@Rotation.from_quat(aligned[f,1,3:7]).as_matrix().T).as_rotvec()
 component=bool((cmd[7:]>=lo-1e-8).all() and (cmd[7:]<=hi+1e-8).all() and abs(cmd[7:]-aligned[f,1,7:]).max()<=.25+1e-8 and abs(residual).max()<=.08+1e-8)
 np.testing.assert_array_equal(cmd[:3],control[f,1,:3]);depth=max([max(0.,-r['signed_gap_m']) for r in contacts],default=0.)
 source_ok=bool((errors[1]<=errors[0]+1e-6).all() and ori[1]<=ori[0]+1e-9)
 row=dict(frame=f,source_record_sha256=sha(p),source_model_orientation_not_measured=True,source_direct_frame_flag=bool(flags['hand_directly_observed'][f,1]),source_metric_frame_flag=bool(flags['hand_metric_valid'][f,1]),URDF_native_point_closure_max_m=max(closures),source_image_before_after_px=[x.tolist() for x in errors],source_orientation_before_after_deg=np.degrees(ori).tolist(),original_component_and_position_bounds_passed=component,source_image_and_orientation_nonregression=source_ok,self_contacts=contacts,self_maximum_penetration_m=depth,actual_hand_target_contacts=target,isolated_source_self_bounds_prerequisites_passed=bool(max(closures)<1e-8 and component and source_ok and depth<=.001))
 rows.append(row);q[f,1]=cmd
def transforms(a):
 out=np.tile(np.eye(4),(len(a),1,1));out[:,:3,:3]=Rotation.from_quat(a[:,3:7]).as_matrix();out[:,:3,3]=a[:,:3];return out
motion=motion_summary(transforms(q[:,1]),ts);root_fails={k:dict(value=motion[k],cap=v) for k,v in authority['original_root_motion_caps'].items() if motion[k]>v+1e-7}
delta=q[:,1,7:]-aligned[:,1,7:];velocity=abs(np.diff(q[:,1,7:],axis=0)/np.diff(ts)[:,None]);finger_failures=dict(max_adjacent_added_rad=float(abs(np.diff(delta,axis=0)).max()),max_second_added_rad=float(abs(np.diff(delta,n=2,axis=0)).max()),adjacent_cap_rad=.05,second_cap_rad=.05,actual_velocity_over_cap=bool((velocity>vel+1e-7).any()),max_velocity_rad_s=velocity.max(0).tolist())
finger_motion_ok=bool(finger_failures['max_adjacent_added_rad']<=.05+1e-7 and finger_failures['max_second_added_rad']<=.05+1e-7 and not finger_failures['actual_velocity_over_cap'])
assert all(r['isolated_source_self_bounds_prerequisites_passed'] for r in rows)
assert root_fails or not finger_motion_ok
for n,h in mf['files'].items():assert sha(F/n)==h
report=dict(schema='v2d.source-coupled-isolated-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),seed_report_sha256=sha(P),authority_report_sha256=sha(A),input_manifest_sha256=sha(F/'manifest.json'),native_model_sha256=sha(MP),rows=rows,all_two_isolated_source_self_bounds_prerequisites_passed=True,sparse_seed_insertion_root_motion_failures=root_fails,sparse_seed_insertion_finger_motion=finger_failures,sparse_seed_insertion_rejected_as_full_trajectory=True,current_verified_reference_byte_unchanged=True,all_three_verified_components_preserved=True,no_optimization_source_fit_native_compile_or_physics_step=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_grasp_or_physical_task_qualified=False,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

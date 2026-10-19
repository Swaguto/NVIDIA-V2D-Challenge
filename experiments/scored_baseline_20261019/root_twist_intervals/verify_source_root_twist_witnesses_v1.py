"""Independent NEW interval witnesses/native/source/FK checks; no old audits."""
from pathlib import Path
import datetime,hashlib,json,sys,time,xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');O=S/'e037-root-twist-independent-verification-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();sys.path[:0]=[str(R/'scripts/dex3_training'),str(R/'scripts/track3')]
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from audit_hand_reference_contacts import canonical_id
from v2d_dex3_flashchord.kinematics import URDFKinematics
from temporal_pose_path import motion_summary
F=S/'e037-thumb-ray-independent-validation-v1/frozen';MP=S/'e037-motion-constrained-wrist-native-v1/native-model.mjb';assert sha(F/'manifest.json')=='f438adf3da846ec1c76c31c7b6d6c12b46bea76004aa7745cfada504378da606';assert sha(MP)=='755fc51729d4c4fc03d04108d39bec2b249074351863f096f9aa498c9ac28566';mf=json.loads((F/'manifest.json').read_text())
for n,h in mf['files'].items():assert sha(F/n)==h
with np.load(F/'targets.npz') as z:control=z['robot_qpos'].copy();objects=z['object_poses'].copy();ts=z['timestamps_s'].copy()
with np.load(F/'aligned-source-reference.npz') as z:aligned=z['robot_qpos'].copy();cameras=z['camera_c2w'].copy();human=z['hand_joints_world'].copy()
with np.load(F/'source-confidence.npz') as z:flags={k:z[k].copy() for k in z.files}
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037'
with np.load(B/'episode.npz') as z:K=z['intrinsics'].copy()
stage=np.array(json.loads((F/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal']);kin=URDFKinematics(F/'robot/urdfs/g1/dex3_right.urdf');studies=['e037-root-twist-interval-study-v1','e037-root-twist-source-disk-study-v2','e037-root-twist-model-ray-study-v3'];witnesses=[];interval_checks=[]
for name in studies:
 folder=S/name;rep=json.loads((folder/'report.json').read_text());assert sha(folder/'all-interval-partitions.json')==rep['all_intervals_sha256'];assert sha(folder/'all-proposals.jsonl')==rep['all_proposals_sha256'];assert sha(folder/'accepted-finite-witnesses.json')==rep['accepted_witnesses_sha256'];witnesses.extend(json.loads((folder/'accepted-finite-witnesses.json').read_text()))
 for case in json.loads((folder/'all-interval-partitions.json').read_text()):
  base=np.array(case['anchor_rotation']);axis=np.array(case['axis_world']);source=np.array(case['source_rotation']);align=np.array(case['aligned_rotation']);local=np.array(case['local_thumb_point']);part=case['partition'];nodes={n['id']:n for n in part['nodes']};leaves=[nodes[i] for i in part['leaf_ids']];assert len(leaves)==len(set(part['leaf_ids']));length=sum(n['interval'][1]-n['interval'][0] for n in leaves);expected=sum(hi-lo for lo,hi in part['orientation_intervals']);assert abs(length-expected)<1e-12;max_point=0.;max_enclosure_excess=0.;sample_count=0
  for node in part['nodes']:
   if node['classification']=='split':
    left,right=[nodes[i] for i in node['children']];assert left['interval'][0]==node['interval'][0] and left['interval'][1]==right['interval'][0] and right['interval'][1]==node['interval'][1]
  for leaf in leaves:
   for theta in np.linspace(*leaf['interval'],9):
    rot=Rotation.from_rotvec(theta*axis).as_matrix()@base;value=Rotation.from_matrix(rot@align.T).as_rotvec();bounds=np.array(leaf['enclosure']['component_bounds']);max_enclosure_excess=max(max_enclosure_excess,float(max((bounds[:,0]-value).max(),(value-bounds[:,1]).max())));assert (value>=bounds[:,0]-1e-12).all() and (value<=bounds[:,1]+1e-12).all();assert Rotation.from_matrix(source@rot.T).magnitude()<=case['source_orientation_cap_rad']+1e-10;max_point=max(max_point,float(np.linalg.norm(rot@local-base@local)));sample_count+=1
  # Independent matrix-trace sinusoid checks the full twist's minimum error.
  values=[float(np.trace(source@(Rotation.from_rotvec(t*axis).as_matrix()@base).T)) for t in [0.,np.pi,np.pi/2]];constant=(values[0]+values[1])/2;cosine=(values[0]-values[1])/2;sine=values[2]-constant
  maximum=constant+float(np.hypot(cosine,sine));min_error=float(np.arccos(np.clip((maximum-1)/2,-1,1)))
  if not part['orientation_intervals']:assert min_error>=case['source_orientation_cap_rad']-1e-10
  interval_checks.append(dict(study=name,frame=case['frame'],anchor=case['anchor_index'],all_nodes_saved=len(part['nodes']),all_leaves_saved=len(leaves),interval_cover_length_error_rad=abs(length-expected),sampled_leaf_states=sample_count,point_preservation_max_m=max_point,enclosure_sample_excess_rad=max_enclosure_excess,independent_trace_minimum_orientation_error_rad=min_error,empty_domain_not_a_full_source_cone_or_task_exclusion=True))
selected=[]
for frame in sorted({w['frame'] for w in witnesses}):
 candidates=[w for w in witnesses if w['frame']==frame]
 def cost(w):
  cmd=np.array(w['command']);rotation=(Rotation.from_quat(cmd[3:7])*Rotation.from_quat(control[frame,1,3:7]).inv()).as_rotvec();return float(np.sum((rotation/.08)**2)+np.sum(((cmd[7:]-control[frame,1,7:])/.25)**2))
 selected.append(min(candidates,key=cost))
model=mujoco.MjModel.from_binary_path(str(MP));data=mujoco.MjData(model);bn=[model.body(i).name for i in range(model.nbody)];jn=[model.joint(i).name for i in range(model.njnt)];wrists={s:[canonical_id(jn,f'{s}_wrist_{a}') for a in ['x','y','z','rotvec']] for s in ['left','right']};fingers={s:[canonical_id(jn,n) for n in mf['joint_names'][s]] for s in ['left','right']};tipnames=['right_hand_palm_link','right_thumb_tip','right_index_tip','right_middle_tip'];tipids=[canonical_id(bn,n) for n in tipnames];limits={j.attrib['name']:j.find('limit').attrib for j in ET.parse(F/'robot/urdfs/g1/dex3_right.urdf').getroot().findall('joint') if j.attrib['type']!='fixed'};lo=np.array([float(limits[n]['lower']) for n in mf['joint_names']['right']]);hi=np.array([float(limits[n]['upper']) for n in mf['joint_names']['right']]);vc=np.array([float(limits[n]['velocity']) for n in mf['joint_names']['right']]);commands=control.copy();rows=[]
def image_errors(frame,cmd):
 poses=kin.poses(cmd[:7],dict(zip(mf['joint_names']['right'],cmd[7:],strict=True)));points=np.stack([poses[n][:3,3] for n in tipnames]);world=points@stage;camera=cameras[frame];C=(world-camera[:3,3])@camera[:3,:3];observed=(human[frame,1,[0,4,8,12]]-camera[:3,3])@camera[:3,:3];assert np.isfinite(observed).all() and (C[:,2]>.01).all() and (observed[:,2]>.01).all()
 def uv(v):return np.c_[K[0,0]*v[:,0]/v[:,2]+K[0,2],K[1,1]*v[:,1]/v[:,2]+K[1,2]]
 return points,np.linalg.norm(uv(C)-uv(observed),axis=1)
for w in selected:
 frame=w['frame'];cmd=np.array(w['command']);np.testing.assert_array_equal(cmd[:3],control[frame,1,:3]);beforepoints,before=image_errors(frame,control[frame,1]);points,after=image_errors(frame,cmd);data.qpos[:]=model.qpos0;data.qvel[:]=0
 for oi,obj in enumerate(mf['objects']):
  bid=canonical_id(bn,obj['name']+'_object');a=int(model.jnt_qposadr[model.body_jntadr[bid]]);v=objects[frame,oi];data.qpos[a:a+3]=v[:3];data.qpos[a+3:a+7]=v[[6,3,4,5]]
 for si,s in enumerate(['left','right']):
  v=control[frame,si] if s=='left' else cmd;ids=wrists[s];data.qpos[model.jnt_qposadr[ids[:3]]]=v[:3];a=int(model.jnt_qposadr[ids[3]]);data.qpos[a:a+4]=v[[6,3,4,5]];data.qpos[model.jnt_qposadr[fingers[s]]]=v[7:]
 mujoco.mj_fwdPosition(model,data);closure=float(abs(points-data.xpos[tipids]).max());self_contacts=[];external=[];crosshand=[]
 for ct in data.contact[:data.ncon]:
  a,b=int(model.geom_bodyid[ct.geom1]),int(model.geom_bodyid[ct.geom2]);ra=bn[a].startswith('dex3_right_');rb=bn[b].startswith('dex3_right_');row=dict(bodies=[bn[a],bn[b]],signed_gap_m=float(ct.dist))
  if ra and rb:self_contacts.append(row)
  elif ra!=rb:
   (crosshand if (bn[b] if ra else bn[a]).startswith('dex3_left_') else external).append(row)
 depth=lambda seq:max([max(0.,-r['signed_gap_m']) for r in seq],default=0.);self_depth=depth(self_contacts);target_depth=depth(external);cross_depth=depth(crosshand);camera=cameras[frame];source=json.loads((B/f'source/hands/right/{frame:06d}.json').read_text());sourceR=stage@camera[:3,:3]@Rotation.from_rotvec(source['mano']['global_orient']).as_matrix()@np.diag([-1.,-1.,1.]);oldR=Rotation.from_quat(control[frame,1,3:7]).as_matrix();newR=Rotation.from_quat(cmd[3:7]).as_matrix();ori=[float(Rotation.from_matrix(sourceR@rot.T).magnitude()) for rot in [oldR,newR]];residual=Rotation.from_matrix(newR@Rotation.from_quat(aligned[frame,1,3:7]).as_matrix().T).as_rotvec();position=bool((cmd[7:]>=lo-1e-8).all() and (cmd[7:]<=hi+1e-8).all() and abs(cmd[7:]-aligned[frame,1,7:]).max()<=.25+1e-7 and abs(residual).max()<=.08+1e-9);source_ok=bool((after<=before+1e-6).all() and ori[1]<=ori[0]+1e-9);passed=bool(closure<1e-8 and position and source_ok and max(self_depth,target_depth,cross_depth)<=.001);rows.append(dict(frame=frame,command=cmd.tolist(),source_direct_frame_flag=bool(flags['hand_directly_observed'][frame,1]),source_metric_frame_flag=bool(flags['hand_metric_valid'][frame,1]),URDF_native_point_closure_max_m=closure,source_image_before_after_px=[before.tolist(),after.tolist()],source_model_orientation_before_after_rad=ori,source_image_and_model_orientation_nonregression=source_ok,original_component_and_actual_URDF_position_passed=position,native_self_contacts=self_contacts,native_external_contacts=external,native_interhand_contacts=crosshand,maximum_self_penetration_m=self_depth,maximum_external_penetration_m=target_depth,maximum_interhand_penetration_m=cross_depth,isolated_source_native_position_checks_passed=passed));commands[frame,1]=cmd
assert all(r['isolated_source_native_position_checks_passed'] for r in rows)
delta=commands[:,1,7:]-aligned[:,1,7:];vel=abs(np.diff(commands[:,1,7:],axis=0)/np.diff(ts)[:,None]);motion=dict(maximum_added_adjacent_rad=float(abs(np.diff(delta,axis=0)).max()),maximum_added_second_rad=float(abs(np.diff(delta,n=2,axis=0)).max()),actual_URDF_velocity_passed=bool((vel<=vc+1e-7).all()),velocity_caps_rad_s=vc.tolist(),max_velocity_rad_s=vel.max(0).tolist());motion['all_original_finger_motion_passed']=bool(motion['maximum_added_adjacent_rad']<=.05+1e-7 and motion['maximum_added_second_rad']<=.05+1e-7 and motion['actual_URDF_velocity_passed'])
def transforms(a):
 out=np.tile(np.eye(4),(len(a),1,1));out[:,:3,:3]=Rotation.from_quat(a[:,3:7]).as_matrix();out[:,:3,3]=a[:,:3];return out
rootcaps=json.loads((S/'e037-wrist-source-authority-v1/report.json').read_text())['original_root_motion_caps'];roots=motion_summary(transforms(commands[:,1]),ts);root_ok=all(roots[k]<=v+1e-7 for k,v in rootcaps.items());assert not motion['all_original_finger_motion_passed'] or not root_ok
for n,h in mf['files'].items():assert sha(F/n)==h
selected_path=O/'selected-isolated-pose-witnesses.json';selected_path.write_text(json.dumps(rows,indent=2)+'\n');report=dict(schema='v2d.root-twist-independent-verification.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),input_manifest_sha256=sha(F/'manifest.json'),native_model_sha256=sha(MP),interval_real_case_checks=interval_checks,selected_witnesses_sha256=sha(selected_path),selected_frames=[r['frame'] for r in rows],all_selected_isolated_source_native_actual_URDF_checks_passed=True,maximum_URDF_native_closure_m=max(r['URDF_native_point_closure_max_m'] for r in rows),sparse_insertion_original_finger_motion=motion,sparse_insertion_original_root_motion=roots,original_root_motion_caps=rootcaps,sparse_insertion_root_motion_passed=root_ok,sparse_insertion_rejected_as_complete_reference=True,legacy_whole_path_still_rejected=True,no_current_verified_reference_change=True,no_full_clock_native_or_swept_motor_grasp_certificate=True,no_source_inference_or_old_native_optimization_job_repeated=True,no_native_compile_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_task_or_policy_qualified=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(selected_frames=report['selected_frames'],maximum_closure=report['maximum_URDF_native_closure_m'],all_isolated_passed=True,finger_motion=motion,root_motion_passed=root_ok,complete_reference_qualified=False),indent=2))

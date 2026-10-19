"""Independent freeze/FK/source/self/motion/hand-target checks on NEW commands."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,sys,time,xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();start=time.monotonic();O=S/'e037-left-thumb-independent-validation-v1';plan_path=S/'compute/left-thumb-independent-validation-CPU-plan-v1.json';plan=json.loads(plan_path.read_text())
import numpy as np,mujoco
from audit_hand_self_contacts import audit as self_audit
from audit_hand_reference_contacts import canonical_id,summarize as target_summary
from v2d_dex3_flashchord.kinematics import URDFKinematics
C=S/'e037-bound-aware-left-thumb-v2';producer_report=json.loads((C/'report.json').read_text());assert producer_report['left_trajectory_source_self_motion_prerequisites_passed'];command_path=C/'left-thumb-reference-commands.npz';assert sha(command_path)==producer_report['candidate_commands_sha256'];P=S/'e037-motion-constrained-wrist-native-v1';f=P/'frozen';mp=P/'native-model.mjb';assert sha(mp)==plan['native_model_sha256'];manifest=json.loads((f/'manifest.json').read_text())
for n,h in manifest['files'].items():assert sha(f/n)==h
with np.load(command_path) as z:q=z['robot_qpos'].copy();control=z['original_robot_qpos'].copy();ts=z['timestamps_s'].copy()
with np.load(f/'targets.npz') as z:targets={k:z[k].copy() for k in z.files}
np.testing.assert_array_equal(control,targets['robot_qpos']);assert np.array_equal(q[:,:,0:7],control[:,:,0:7]) and np.array_equal(q[:,1],control[:,1]) and np.array_equal(q[:,0,7:11],control[:,0,7:11]);assert len(q)==151;kin={s:URDFKinematics(f/f'robot/urdfs/g1/dex3_{s}.urdf') for s in ['left','right']};targets['robot_qpos']=q
for si,side in enumerate(['left','right']):
 new=[]
 for frame in range(len(q)):
  before=kin[side].poses(control[frame,si,:7],dict(zip(manifest['joint_names'][side],control[frame,si,7:],strict=True)));np.testing.assert_allclose(np.stack([before[n][:3,3] for n in manifest['frame_names'][side]]),targets[side+'_frames'][frame],atol=1e-9,rtol=0);after=kin[side].poses(q[frame,si,:7],dict(zip(manifest['joint_names'][side],q[frame,si,7:],strict=True)));new.append(np.stack([after[n][:3,3] for n in manifest['frame_names'][side]]))
 targets[side+'_frames']=np.array(new)
newf=O/'frozen';shutil.copytree(f,newf,copy_function=os.link);(newf/'targets.npz').unlink();np.savez_compressed(newf/'targets.npz',**targets);(newf/'manifest.json').unlink();new_manifest=json.loads(json.dumps(manifest));new_manifest['files']['targets.npz']=sha(newf/'targets.npz');new_manifest['left_thumb_self_contact_candidate']=dict(parent_manifest_sha256=sha(f/'manifest.json'),command_file_sha256=sha(command_path),source_human_data_and_authority_unchanged=True,root_commands_and_right_hand_unchanged=True,physical_task_accepted=False,training_approved=False);(newf/'manifest.json').write_text(json.dumps(new_manifest,indent=2)+'\n')
self_result=self_audit(newf,mp,plan['native_model_sha256']);(O/'fresh-candidate-self-contacts.json').write_text(json.dumps(self_result,indent=2)+'\n');assert self_result['summary_by_hand']['left']['frames_over_tolerance']==0;assert self_result['summary_by_hand']['right']['frames_over_tolerance']==46
with np.load(f/'aligned-source-reference.npz') as z:cameras=z['camera_c2w'].copy();human=z['hand_joints_world'].copy();aligned=z['robot_qpos'].copy()
with np.load(f/'source-confidence.npz') as z:source_flags={k:z[k].copy() for k in z.files}
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz') as z:K=z['intrinsics'].copy()
stage=np.asarray(json.loads((f/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal']);source_rows=[];before_all=[];after_all=[]
def image_points(points,frame):
 camera=cameras[frame];C=np.einsum('ni,ij->nj',points-camera[:3,3],camera[:3,:3],optimize=False);assert (C[:,2]>.01).all();return np.c_[K[0,0]*C[:,0]/C[:,2]+K[0,2],K[1,1]*C[:,1]/C[:,2]+K[1,2]]
for frame in range(len(q)):
 errors=[]
 for cmds in [control,q]:
  poses=kin['left'].poses(cmds[frame,0,:7],dict(zip(manifest['joint_names']['left'],cmds[frame,0,7:],strict=True)));points=np.stack([poses[n][:3,3] for n in ['left_hand_palm_link','left_thumb_tip','left_index_tip','left_middle_tip']]);world=np.einsum('ni,ij->nj',points,stage,optimize=False);errors.append(np.linalg.norm(image_points(world,frame)-image_points(human[frame,0,[0,4,8,12]],frame),axis=1))
 source_rows.append(dict(frame=frame,before_after_errors_px=[a.tolist() for a in errors],every_point_nonregressing=bool((errors[1]<=errors[0]+1e-6).all())));before_all.extend(errors[0]);after_all.extend(errors[1])
assert all(r['every_point_nonregressing'] for r in source_rows)
motion=[]
for si,side in enumerate(['left','right']):
 joints={j.attrib['name']:j.find('limit').attrib for j in ET.parse(f/f'robot/urdfs/g1/dex3_{side}.urdf').getroot().findall('joint') if j.attrib['type']!='fixed'};names=manifest['joint_names'][side];lo=np.array([float(joints[n]['lower']) for n in names]);hi=np.array([float(joints[n]['upper']) for n in names]);vmax=np.array([float(joints[n]['velocity']) for n in names]);speed=abs(np.diff(q[:,si,7:],axis=0)/np.diff(ts)[:,None]);delta=q[:,si,7:]-aligned[:,si,7:];ok=bool(((q[:,si,7:]>=lo-1e-8)&(q[:,si,7:]<=hi+1e-8)).all() and (speed<=vmax+1e-7).all() and abs(delta).max()<=.25+1e-7 and abs(np.diff(delta,axis=0)).max()<=.05+1e-7 and abs(np.diff(delta,n=2,axis=0)).max()<=.05+1e-7);assert ok;motion.append(dict(hand=side,original_URDF_position_velocity_component_and_added_motion_caps_passed=ok))
model=mujoco.MjModel.from_binary_path(str(mp));data=mujoco.MjData(model);bn=[model.body(i).name for i in range(model.nbody)];jn=[model.joint(i).name for i in range(model.njnt)];wrists={s:[canonical_id(jn,f'{s}_wrist_{a}') for a in ['x','y','z','rotvec']] for s in ['left','right']};fingers={s:[canonical_id(jn,n) for n in manifest['joint_names'][s]] for s in ['left','right']};target_bodies={canonical_id(bn,obj['name']+'_object'):obj['name'] for obj in manifest['objects']};target_bodies[canonical_id(bn,'support')]='support'
def set_frame(frame,fraction=None):
 data.qpos[:]=model.qpos0;data.qvel[:]=0
 for oi,obj in enumerate(manifest['objects']):
  bid=canonical_id(bn,obj['name']+'_object');a=int(model.jnt_qposadr[model.body_jntadr[bid]]);v=targets['object_poses'][frame,oi];data.qpos[a:a+3]=v[:3];data.qpos[a+3:a+7]=v[[6,3,4,5]]
 for si,side in enumerate(['left','right']):
  v=q[frame,si].copy()
  if fraction is not None:v[7:]=(1-fraction)*q[frame,si,7:]+fraction*q[frame+1,si,7:]
  w=wrists[side];data.qpos[model.jnt_qposadr[w[:3]]]=v[:3];a=int(model.jnt_qposadr[w[3]]);data.qpos[a:a+4]=v[[6,3,4,5]];data.qpos[model.jnt_qposadr[fingers[side]]]=v[7:]
 mujoco.mj_fwdPosition(model,data)
hand_target_rows=[]
for frame in range(len(q)):
 set_frame(frame);contacts=[]
 for c in data.contact[:data.ncon]:
  a,b=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
  for side in ['left','right']:
   if bn[a].startswith('dex3_'+side+'_') and b in target_bodies:hand,target=a,b
   elif bn[b].startswith('dex3_'+side+'_') and a in target_bodies:hand,target=b,a
   else:continue
   contacts.append(dict(hand=side,target=target_bodies[target],native_bodies=[bn[hand],bn[target]],signed_gap_m=float(c.dist)))
 hand_target_rows.append(dict(reference_frame=frame,contacts=contacts))
interpolation=[]
for frame in range(len(q)-1):
 for fraction in [.25,.5,.75]:
  set_frame(frame,fraction);depth=max([max(0.,-float(c.dist)) for c in data.contact[:data.ncon] if bn[int(model.geom_bodyid[c.geom1])].startswith('dex3_left_') and bn[int(model.geom_bodyid[c.geom2])].startswith('dex3_left_')],default=0.);interpolation.append(dict(frame=frame,fraction=fraction,maximum_left_self_penetration_m=depth))
source_file_sha_unchanged=sha(newf/'source-confidence.npz')==sha(f/'source-confidence.npz') and sha(newf/'aligned-source-reference.npz')==sha(f/'aligned-source-reference.npz');assert source_file_sha_unchanged;summary=target_summary(hand_target_rows);sample_max=max(r['maximum_left_self_penetration_m'] for r in interpolation);assert sample_max<=.001
report=dict(schema='v2d.independent-left-thumb-reference-validation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),inputs_sha256={str(p):sha(p) for p in [C/'report.json',command_path,mp,f/'manifest.json',f/'source-confidence.npz',f/'aligned-source-reference.npz']},new_frozen_manifest_sha256=sha(newf/'manifest.json'),all_frozen_source_closure_and_hand_FK_verified=True,maximum_native_hand_FK_error=self_result['maximum_native_FK_error'],self_summary_by_hand=self_result['summary_by_hand'],source_anchor_points_assessed=len(before_all),source_anchor_rows=source_rows,source_left_all604_image_points_nonregressing=True,source_before_after_median_px=[float(np.median(before_all)),float(np.median(after_all))],source_before_after_p95_px=[float(np.quantile(before_all,.95)),float(np.quantile(after_all,.95))],source_flags_and_human_reference_files_byte_identical=source_file_sha_unchanged,source_metric_flags_not_promoted_to_per_landmark_measured_depth=True,all_original_joint_motion_caps_passed=motion,root_arrays_exactly_unchanged=True,fresh_hand_object_support_summary=summary,fresh_hand_object_support_rows=hand_target_rows,interpolated_left_self_states=len(interpolation),maximum_interpolated_left_self_penetration_m=sample_max,interpolation_rows=interpolation,sampled_self_interpolation_not_continuous_or_full_task_swept_certificate=True,left_component_source_self_motion_checks_passed=True,right_self_and_hand_object_failures_still_present=True,full_task_1mm_gate_passed=False,no_native_model_or_filter_change=True,new_reference_not_old_job_rerun=True,no_native_compile_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['self_summary_by_hand','source_before_after_median_px','source_before_after_p95_px','fresh_hand_object_support_summary','interpolated_left_self_states','maximum_interpolated_left_self_penetration_m','left_component_source_self_motion_checks_passed','full_task_1mm_gate_passed','wall_seconds']},indent=2),flush=True)

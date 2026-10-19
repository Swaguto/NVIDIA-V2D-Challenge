"""Independent NEW combined reference: full-clock FK/source/self/target/motion and sampled paths."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,sys,time,xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();O=S/'e037-thumb-ray-independent-validation-v1';F=S/'e037-right-late-independent-validation-v1/frozen';MP=S/'e037-motion-constrained-wrist-native-v1/native-model.mjb';C=S/'e037-thumb-ray-temporal-path-v2';rep=json.loads((C/'report.json').read_text());command_path=C/'reference-commands.npz';assert sha(command_path)==rep['commands_sha256'];assert rep['input_manifest_sha256']==sha(F/'manifest.json');expected_model='755fc51729d4c4fc03d04108d39bec2b249074351863f096f9aa498c9ac28566';assert sha(MP)==expected_model
import numpy as np,mujoco
from scipy.spatial.transform import Rotation,Slerp
from audit_hand_self_contacts import audit as self_audit
from audit_hand_reference_contacts import canonical_id,summarize as target_summary
from v2d_dex3_flashchord.kinematics import URDFKinematics
manifest=json.loads((F/'manifest.json').read_text())
for n,h in manifest['files'].items():assert sha(F/n)==h
with np.load(command_path) as z:q=z['robot_qpos'].copy();control=z['original_robot_qpos'].copy();ts=z['timestamps_s'].copy()
with np.load(F/'targets.npz') as z:targets={k:z[k].copy() for k in z.files}
np.testing.assert_array_equal(control,targets['robot_qpos']);np.testing.assert_array_equal(q[:,0],control[:,0]);np.testing.assert_array_equal(q[:,:,:7],control[:,:,:7]);np.testing.assert_array_equal(q[:,1,7:11],control[:,1,7:11]);np.testing.assert_array_equal(q[5:,1],control[5:,1]);kin={s:URDFKinematics(F/f'robot/urdfs/g1/dex3_{s}.urdf') for s in ['left','right']};targets['robot_qpos']=q
for si,side in enumerate(['left','right']):
 frames=[]
 for frame in range(len(q)):
  before=kin[side].poses(control[frame,si,:7],dict(zip(manifest['joint_names'][side],control[frame,si,7:],strict=True)));np.testing.assert_allclose(np.stack([before[n][:3,3] for n in manifest['frame_names'][side]]),targets[side+'_frames'][frame],atol=1e-9,rtol=0);after=kin[side].poses(q[frame,si,:7],dict(zip(manifest['joint_names'][side],q[frame,si,7:],strict=True)));frames.append(np.stack([after[n][:3,3] for n in manifest['frame_names'][side]]))
 targets[side+'_frames']=np.array(frames)
newf=O/'frozen';shutil.copytree(F,newf,copy_function=os.link);(newf/'targets.npz').unlink();np.savez_compressed(newf/'targets.npz',**targets);(newf/'manifest.json').unlink();new_manifest=json.loads(json.dumps(manifest));new_manifest['files']['targets.npz']=sha(newf/'targets.npz');new_manifest['right_early_thumb_ray_component']=dict(parent_verified_combined_manifest_sha256=sha(F/'manifest.json'),command_file_sha256=sha(command_path),source_human_authority_unchanged=True,left_and_root_commands_unchanged=True,partial_component_not_whole_task=True,training_approved=False,physical_task_accepted=False);(newf/'manifest.json').write_text(json.dumps(new_manifest,indent=2)+'\n');assert (newf/'targets.npz').stat().st_ino!=(F/'targets.npz').stat().st_ino
self_result=self_audit(newf,MP,expected_model);(O/'fresh-candidate-self-contacts.json').write_text(json.dumps(self_result,indent=2)+'\n');old_self=json.loads((S/'e037-right-late-independent-validation-v1/fresh-candidate-self-contacts.json').read_text());assert self_result['summary_by_hand']['left']['frames_over_tolerance']==0
with np.load(F/'aligned-source-reference.npz') as z:cameras=z['camera_c2w'].copy();human=z['hand_joints_world'].copy();aligned=z['robot_qpos'].copy()
with np.load(F/'source-confidence.npz') as z:flags={k:z[k].copy() for k in z.files}
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz') as z:K=z['intrinsics'].copy()
stage=np.asarray(json.loads((F/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal']);source_rows=[];before_all=[];after_all=[];missing=[]
def camera_points(world,frame):
 camera=cameras[frame];return np.einsum('ni,ij->nj',world-camera[:3,3],camera[:3,:3],optimize=False)
def pixels(C):
 assert np.isfinite(C).all() and (C[:,2]>.01).all()
 return np.c_[K[0,0]*C[:,0]/C[:,2]+K[0,2],K[1,1]*C[:,1]/C[:,2]+K[1,2]]
for si,side in enumerate(['left','right']):
 for frame in range(len(q)):
  observed=camera_points(human[frame,si,[0,4,8,12]],frame);mask=np.isfinite(observed).all(1)&(observed[:,2]>.01)&bool(flags['hand_valid'][frame,si]);errors=[]
  if not mask.all():missing.append(dict(hand=side,frame=frame,missing_anchor_indices=np.flatnonzero(~mask).tolist()));np.testing.assert_array_equal(q[frame,si],control[frame,si])
  for cmds in [control,q]:
   poses=kin[side].poses(cmds[frame,si,:7],dict(zip(manifest['joint_names'][side],cmds[frame,si,7:],strict=True)));points=np.stack([poses[n][:3,3] for n in [f'{side}_hand_palm_link',f'{side}_thumb_tip',f'{side}_index_tip',f'{side}_middle_tip']]);world=np.einsum('ni,ij->nj',points,stage,optimize=False);err=np.full(4,np.nan);err[mask]=np.linalg.norm(pixels(camera_points(world,frame)[mask])-pixels(observed[mask]),axis=1);errors.append(err)
  ok=bool((errors[1][mask]<=errors[0][mask]+1e-6).all());source_rows.append(dict(hand=side,frame=frame,source_direct=bool(flags['hand_directly_observed'][frame,si]),source_metric_flag=bool(flags['hand_metric_valid'][frame,si]),image_anchor_available=mask.tolist(),source_before_after_px=[[float(x) if np.isfinite(x) else None for x in err] for err in errors],nonregressing=ok));before_all.extend(errors[0][mask]);after_all.extend(errors[1][mask])
motion=[]
for si,side in enumerate(['left','right']):
 lj={j.attrib['name']:j.find('limit').attrib for j in ET.parse(F/f'robot/urdfs/g1/dex3_{side}.urdf').getroot().findall('joint') if j.attrib['type']!='fixed'};names=manifest['joint_names'][side];lo=np.array([float(lj[n]['lower']) for n in names]);hi=np.array([float(lj[n]['upper']) for n in names]);vc=np.array([float(lj[n]['velocity']) for n in names]);v=abs(np.diff(q[:,si,7:],axis=0)/np.diff(ts)[:,None]);d=q[:,si,7:]-aligned[:,si,7:];ok=bool(((q[:,si,7:]>=lo-1e-8)&(q[:,si,7:]<=hi+1e-8)).all() and (v<=vc+1e-7).all() and abs(d).max()<=.25+1e-7 and abs(np.diff(d,axis=0)).max()<=.05+1e-7 and abs(np.diff(d,n=2,axis=0)).max()<=.05+1e-7);motion.append(dict(hand=side,passed=ok,component_rad=float(abs(d).max()),adjacent_added_rad=float(abs(np.diff(d,axis=0)).max()),second_added_rad=float(abs(np.diff(d,n=2,axis=0)).max()),max_velocity_rad_s=v.max(0).tolist(),velocity_caps_rad_s=vc.tolist()))
model=mujoco.MjModel.from_binary_path(str(MP));data=mujoco.MjData(model);bn=[model.body(i).name for i in range(model.nbody)];jn=[model.joint(i).name for i in range(model.njnt)];wrists={s:[canonical_id(jn,f'{s}_wrist_{a}') for a in ['x','y','z','rotvec']] for s in ['left','right']};fingers={s:[canonical_id(jn,n) for n in manifest['joint_names'][s]] for s in ['left','right']};target_bodies={canonical_id(bn,obj['name']+'_object'):obj['name'] for obj in manifest['objects']};target_bodies[canonical_id(bn,'support')]='support'
def blend(v,w,t):
 out=(1-t)*v+t*w;out[3:7]=Slerp([0,1],Rotation.from_quat(np.stack([v[3:7],w[3:7]])))(t).as_quat();return out

def set_frame(frame,t=None):
 data.qpos[:]=model.qpos0;data.qvel[:]=0
 for oi,obj in enumerate(manifest['objects']):
  bid=canonical_id(bn,obj['name']+'_object');a=int(model.jnt_qposadr[model.body_jntadr[bid]]);v=targets['object_poses'][frame,oi].copy()
  if t is not None:v=blend(v,targets['object_poses'][frame+1,oi],t)
  data.qpos[a:a+3]=v[:3];data.qpos[a+3:a+7]=v[[6,3,4,5]]
 for si,side in enumerate(['left','right']):
  v=q[frame,si].copy()
  if t is not None:v=blend(v,q[frame+1,si],t)
  w=wrists[side];data.qpos[model.jnt_qposadr[w[:3]]]=v[:3];a=int(model.jnt_qposadr[w[3]]);data.qpos[a:a+4]=v[[6,3,4,5]];data.qpos[model.jnt_qposadr[fingers[side]]]=v[7:]
 mujoco.mj_fwdPosition(model,data)
def current_contacts():
 target=[];self_depth=dict(left=0.,right=0.)
 for c in data.contact[:data.ncon]:
  a,b=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
  for side in ['left','right']:
   if bn[a].startswith('dex3_'+side+'_') and bn[b].startswith('dex3_'+side+'_'):self_depth[side]=max(self_depth[side],max(0.,-float(c.dist)))
   if bn[a].startswith('dex3_'+side+'_') and b in target_bodies:hand,ob=a,b
   elif bn[b].startswith('dex3_'+side+'_') and a in target_bodies:hand,ob=b,a
   else:continue
   target.append(dict(hand=side,target=target_bodies[ob],native_bodies=[bn[hand],bn[ob]],signed_gap_m=float(c.dist)))
 return target,self_depth
hand_target_rows=[]
for frame in range(len(q)):
 set_frame(frame);contacts,_=current_contacts();hand_target_rows.append(dict(reference_frame=frame,contacts=contacts))
interpolation=[]
for frame in range(len(q)-1):
 for fraction in [.25,.5,.75]:
  set_frame(frame,fraction);contacts,depth=current_contacts();interpolation.append(dict(frame=frame,fraction=fraction,self_penetration_m=depth,hand_object_support_contacts=contacts))
source_unchanged=all(sha(newf/n)==sha(F/n) for n in manifest['files'] if n!='targets.npz');assert source_unchanged;summary=target_summary(hand_target_rows);old_depth=np.array([max([max(0.,-c['signed_gap_m']) for c in row['contacts'] if c['hand']=='right'],default=0.) for row in old_self['frames']]);new_depth=np.array([max([max(0.,-c['signed_gap_m']) for c in row['contacts'] if c['hand']=='right'],default=0.) for row in self_result['frames']]);self_nonregress=bool((new_depth<=old_depth+1e-8).all());source_ok=all(r['nonregressing'] for r in source_rows);component_ok=bool(source_ok and all(m['passed'] for m in motion) and self_nonregress and self_result['summary_by_hand']['left']['frames_over_tolerance']==0 and self_result['summary_by_hand']['right']['frames_over_tolerance']<13)
report=dict(schema='v2d.independent-right-early-ray-reference-validation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),inputs_sha256={str(p):sha(p) for p in [C/'report.json',command_path,MP,F/'manifest.json',F/'source-confidence.npz',F/'aligned-source-reference.npz']},new_frozen_manifest_sha256=sha(newf/'manifest.json'),full151_native_FK_source_closure_verified=True,maximum_native_FK_error=self_result['maximum_native_FK_error'],self_summary_by_hand=self_result['summary_by_hand'],every_right_self_knot_nonregressing=self_nonregress,right_remaining_bad_self_frames=np.flatnonzero(new_depth>.001).tolist(),source_anchor_points_assessed=len(before_all),source_rows=source_rows,all_available_source_image_anchors_nonregressing=source_ok,source_missing_anchor_records=missing,source_missing_right_commands_unchanged=True,source_before_after_median_px=[float(np.median(before_all)),float(np.median(after_all))],source_before_after_p95_px=[float(np.quantile(before_all,.95)),float(np.quantile(after_all,.95))],all_non_target_files_byte_identical=source_unchanged,left_and_late_right_components_roots_nonthumb_exact_unchanged=True,source_metric_flags_not_promoted_to_per_landmark_measured_depth=True,original_motion=motion,fresh_hand_object_support_summary=summary,fresh_hand_object_support_rows=hand_target_rows,sampled_full_rigid_root_object_and_finger_path_states=len(interpolation),interpolation_rows=interpolation,sampled_self_max_penetration_m={s:max(r['self_penetration_m'][s] for r in interpolation) for s in ['left','right']},sampled_target_max_penetration_m=max([max(0.,-c['signed_gap_m']) for r in interpolation for c in r['hand_object_support_contacts']],default=0.),samples_not_continuous_or_full_task_swept_certificate=True,right_early_ray_component_source_self_motion_checks_passed=component_ok,full_task_1mm_gate_passed=False,whole_reference_physical_task_or_policy_qualified=False,all_previous_failures_retained=True,no_native_model_filter_source_change_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,training_approved=False,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['self_summary_by_hand','right_remaining_bad_self_frames','source_anchor_points_assessed','all_available_source_image_anchors_nonregressing','source_before_after_median_px','source_before_after_p95_px','original_motion','fresh_hand_object_support_summary','sampled_self_max_penetration_m','sampled_target_max_penetration_m','right_early_ray_component_source_self_motion_checks_passed','wall_seconds']},indent=2))

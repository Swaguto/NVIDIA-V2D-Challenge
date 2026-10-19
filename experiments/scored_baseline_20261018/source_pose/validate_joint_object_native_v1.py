"""Fresh full native audit of new object poses with unchanged robot commands."""
from pathlib import Path
import datetime,hashlib,json,shutil,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path[:0]=[str(S/'cpu-native-runtime-v1/src'),str(R/'scripts/dex3_training'),str(R/'scripts/track3')]
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from check_native_scene import compile_scene
import audit_hand_reference_contacts as hands
import audit_reference_contacts as objects
from reference_contact_gate import assess
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();O=S/'e037-joint-object-native-validation-v1';O.mkdir(exist_ok=True);start=time.monotonic()
P=S/'plane-aligned-articulated-native-validation-v2/frozen';base_manifest=json.loads((P/'manifest.json').read_text())
for name,h in base_manifest['files'].items():assert sha(P/name)==h,name
proposal_path=S/'e037-joint-pose-translation-repair-v1/unqualified-combined-object-proposal.npz';proposal_report_path=S/'e037-joint-pose-translation-repair-v1/report.json';proof=json.loads(proposal_report_path.read_text());assert proof['all19changed_source_frames_passed'] and not proof['over_original_motion_caps']
with np.load(proposal_path) as z:new_world=z['world_from_object'].copy();times=z['timestamps_s'].copy();axis=z['frame_indices'].copy()
old_track_path=S/'plane-aligned-pose-temporal-path-v4/inferred-object-trajectory.npz'
with np.load(old_track_path) as z:old_world=z['candidate_world_from_object'].copy()
with np.load(P/'targets.npz') as z:targets={k:z[k].copy() for k in z.files}
with np.load(P/'aligned-source-reference.npz') as z:flags={k:z[k].copy() for k in ['hand_valid','hand_directly_observed','hand_metric_valid']}
stage=np.asarray(json.loads((P/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal']);T=np.eye(4);T[:3,:3]=stage
old_native=np.einsum('ij,tjk->tik',T,old_world,optimize=False);new_native=np.einsum('ij,tjk->tik',T,new_world,optimize=False)
old_pose=targets['object_poses'][:,0];old_reference=np.tile(np.eye(4),(len(old_pose),1,1));old_reference[:,:3,:3]=Rotation.from_quat(old_pose[:,3:]).as_matrix();old_reference[:,:3,3]=old_pose[:,:3];closure=float(abs(old_native-old_reference).max());assert closure<1e-12
new={k:v.copy() for k,v in targets.items()};new['object_poses'][:,0,:3]=new_native[:,:3,3];new['object_poses'][:,0,3:]=Rotation.from_matrix(new_native[:,:3,:3]).as_quat()
np.testing.assert_array_equal(times,targets['timestamps_s']);np.testing.assert_array_equal(axis,targets['frame_indices'])
inputs=O/'frozen';shutil.copytree(P,inputs)
np.savez_compressed(inputs/'targets.npz',**new);manifest=json.loads((inputs/'manifest.json').read_text());manifest['files']['targets.npz']=sha(inputs/'targets.npz');manifest['new_object_source_pose_proposal']=dict(producer_sha256=sha(Path(__file__)),original_manifest_sha256=sha(P/'manifest.json'),proposal_sha256=sha(proposal_path),source_proof_sha256=sha(proposal_report_path),source_flags_unchanged=True,robot_targets_unchanged=True,mesh_URDF_or_support_unchanged=True,physical_task_accepted=False)
(inputs/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
report=dict(schema='v2d.joint-source-object-full-native-validation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),input_manifest_sha256=sha(inputs/'manifest.json'),original_manifest_sha256=sha(P/'manifest.json'),proposal_sha256=sha(proposal_path),source_native_old_object_pose_closure_error_m=closure,all151reference_knots=True,robot_commands_source_flags_geometry_URDF_and_support_unchanged=True,source_object_window_nonregression_and_motion_passed=True,physics_stepped=False,new_training_steps=0,new_GPU_work=False,physical_task_accepted=False,full_reference_promoted=False)
save=lambda:(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');save()
try:
 compiled=compile_scene(inputs)
 hands.compile_scene=lambda candidate_path:compiled if candidate_path.resolve()==inputs.resolve() else (_ for _ in ()).throw(ValueError('Unexpected candidate input'))
 objects.compile_scene=hands.compile_scene
 hand=hands.audit(inputs.resolve());obj=objects.audit(inputs.resolve())
 for name,value in [('hand-reference-contacts.json',hand),('object-reference-contacts.json',obj)]: (O/name).write_text(json.dumps(value,indent=2)+'\n')
 gate=assess(inputs/'manifest.json',manifest,obj,hand);(O/'complete-reference-contact-gate.json').write_text(json.dumps(gate,indent=2)+'\n');report['native_contact_gate']=gate
 import mujoco
 model=compiled[2].mj_model;mujoco.mj_saveModel(model,str(O/'native-model.mjb'),None)
 control=mujoco.MjModel.from_binary_path(str(S/'plane-aligned-native-reference-v1/native-model.mjb'));checks=[]
 for key in ['geom_type','geom_bodyid','geom_dataid','geom_pos','geom_quat','geom_size','mesh_vert','mesh_pos','mesh_quat']:
  a,b=np.asarray(getattr(control,key)),np.asarray(getattr(model,key));error=float(abs(a-b).max(initial=0));checks.append(dict(array=key,same_shape=a.shape==b.shape,maximum_absolute_error=error,passed=a.shape==b.shape and error<1e-10))
 assert all(c['passed'] for c in checks);report['fresh_native_geometry_matches_unchanged_control']=checks
 bodies=[model.body(i).name for i in range(model.nbody)];joints=[model.joint(i).name for i in range(model.njnt)];data=mujoco.MjData(model);bid=hands.canonical_id(bodies,'wooden_spoon_object');oa=int(model.jnt_qposadr[model.body_jntadr[bid]]);object_geoms=list(np.flatnonzero(model.geom_bodyid==bid));assert len(object_geoms)==9
 binding={};groups={}
 for si,side in enumerate(['left','right']):
  binding[side]=([int(model.jnt_qposadr[hands.canonical_id(joints,f'{side}_wrist_{a}')]) for a in ['x','y','z','rotvec']],[int(model.jnt_qposadr[hands.canonical_id(joints,n)]) for n in manifest['joint_names'][side]])
  for digit in ['thumb','index','middle']:groups[side+'/'+digit]=[i for i,b in enumerate(model.geom_bodyid) if bodies[b].startswith('dex3_'+side+'_') and ('hand_'+digit+'_') in bodies[b]]
 assert all(groups.values());local={}
 for g in set(object_geoms+[i for values in groups.values() for i in values]):
  mid=int(model.geom_dataid[g]);assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH;first=int(model.mesh_vertadr[mid]);count=int(model.mesh_vertnum[mid]);verts=np.asarray(model.mesh_vert[first:first+count],float);hull=ConvexHull(verts);normals=hull.equations[:,:3];_,ids=np.unique(np.round(normals,10),axis=0,return_index=True);local[g]=(verts[hull.vertices],normals[ids])
 def mm(a,b):return np.einsum('ij,jk->ik',a,b,optimize=False)
 def bound(a,an,b,bn):
  axes=np.r_[an,bn];axes/=np.linalg.norm(axes,axis=1)[:,None];ap=mm(a,axes.T);bp=mm(b,axes.T);lower=float(np.maximum(np.maximum(ap.min(0)-bp.max(0),bp.min(0)-ap.max(0)),0).max());delta=a[:,None]-b[None];upper=float(np.sqrt(np.einsum('ijk,ijk->ij',delta,delta,optimize=False).min()));assert 0<=lower<=upper+1e-7;return lower,upper
 box=np.array([[a,b,c] for a in [-.5,.5] for b in [-.5,.5] for c in [-.5,.5]]);lo,hi=bound(box,np.eye(3),box+[2.,0,0],np.eye(3));assert abs(lo-1)<1e-12 and abs(hi-1)<1e-12
 grasp=[]
 for frame in [41,47,49,50,59]:
  data.qpos[:]=model.qpos0;data.qpos[oa:oa+3]=new['object_poses'][frame,0,:3];data.qpos[oa+3:oa+7]=new['object_poses'][frame,0,[6,3,4,5]]
  for si,side in enumerate(['left','right']):
   wrists,fingers=binding[side];command=new['robot_qpos'][frame,si];data.qpos[wrists[:3]]=command[:3];adr=wrists[3];data.qpos[adr:adr+4]=command[[6,3,4,5]];data.qpos[fingers]=command[7:]
  mujoco.mj_fwdPosition(model,data);world_parts={g:(mm(v,data.geom_xmat[g].reshape(3,3).T)+data.geom_xpos[g],mm(n,data.geom_xmat[g].reshape(3,3).T)) for g,(v,n) in local.items()};distances={}
  for group,geoms in groups.items():
   values=[bound(*world_parts[g],*world_parts[og]) for g in geoms for og in object_geoms];distances[group]=dict(lower_m=min(x[0] for x in values),upper_m=min(x[1] for x in values))
  grasp.append(dict(frame=frame,whole_digit_separation_bounds=distances,source_flags={side:{k:bool(flags[k][frame,si]) for k in flags} for si,side in enumerate(['left','right'])}))
 report['grasp_separation_samples']=grasp;report['conservative_distance_bound_controls_passed']=True;report['force_closure_or_grasp_stability_certified']=False
 report['reference_grasp_candidate_qualified']=False;report['swept_checks_not_run_because_grasp_reference_still_unqualified']=True
 for k in targets:
  if k!='object_poses':np.testing.assert_array_equal(new[k],targets[k])
 for name,h in manifest['files'].items():assert sha(inputs/name)==h
 for name,h in base_manifest['files'].items():assert sha(P/name)==h
 report['new_and_original_frozen_closures_verified_after']=True
except BaseException as error:report['error']=repr(error);raise
finally:
 report['wall_seconds']=time.monotonic()-start;save();print(json.dumps(report,indent=2))

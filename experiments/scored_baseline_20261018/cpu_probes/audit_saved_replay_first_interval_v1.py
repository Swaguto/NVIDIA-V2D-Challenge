"""Inspect saved replay controls/body poses only; never rerun an episode.

Every recorded sample is AFTER a complete control interval. Differences in
reactive effort therefore do not establish different initial actions or resets.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.spatial.transform import Rotation
S=Path(__file__).resolve().parents[1];O=S/'saved-replay-first-interval-audit-v1';O.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();catalog=json.load(open(S/'complete-submission-catalog-v1.json'));rows=[];inputs={};started=time.monotonic()
for ep in [3,15,38]:
 source=next(x for x in catalog['episodes'] if x['episode']==ep);left=Path(source['policy_root'])/'readback';right=S/f'scored-replay-rehearsal-results-v1/runs/e{ep:03d}/readback';paths=[left/'native-state-readback.npz',right/'native-state-readback.npz',left/'body-labels.json',right/'body-labels.json',left/'report.json',right/'report.json']
 before={str(p.relative_to(S)):sha(p) for p in paths};inputs.update(before);labels=json.load(open(left/'body-labels.json'));assert labels==json.load(open(right/'body-labels.json'));a_report=json.load(open(left/'report.json'));b_report=json.load(open(right/'report.json'));assert a_report['checkpoint_sha256']==b_report['checkpoint_sha256'] and a_report['source_manifest_sha256']==b_report['source_manifest_sha256'];n=a_report['frame_count'];assert n==b_report['frame_count'];frames=[]
 with np.load(left/'native-state-readback.npz') as a,np.load(right/'native-state-readback.npz') as b:
  controls={}
  for key in ['motor_target','applied_motor_position_target','motor_velocity_target','direct_joint_effort','native_actuator_force','native_combined_joint_motor_force']:
   av,bv=a[key],b[key];assert av.shape==bv.shape and av.shape[0]==n
   first=np.abs(av[0].astype(np.float64)-bv[0].astype(np.float64));controls[key]=dict(first_saved_sample_bit_exact=av[0].tobytes()==bv[0].tobytes(),first_saved_sample_maximum_absolute_difference=float(first.max()),finite_first_samples=bool(np.isfinite(av[0]).all() and np.isfinite(bv[0]).all()))
  av,bv=a['body_poses'],b['body_poses'];assert av.shape==bv.shape and av.shape[1]==len(labels) and len(labels)%8==0
  perworld=len(labels)//8;assert all(labels[i*perworld:(i+1)*perworld]==labels[:perworld] for i in range(8));assert np.allclose(np.linalg.norm(av[0,:,3:],axis=-1),1,atol=1e-4) and np.allclose(np.linalg.norm(bv[0,:,3:],axis=-1),1,atol=1e-4)
  for index,label in enumerate(labels[:perworld]):
   if not(label.endswith('/object') or label.endswith('hand_palm_link')):continue
   ids=np.arange(8)*perworld+index;pos=np.linalg.norm(av[0,ids,:3]-bv[0,ids,:3],axis=-1);angle=(Rotation.from_quat(av[0,ids,3:]).inv()*Rotation.from_quat(bv[0,ids,3:])).magnitude();frames.append(dict(body=label,worlds=8,first_saved_position_maximum_difference_m=float(pos.max()),first_saved_position_median_difference_m=float(np.median(pos)),first_saved_orientation_maximum_difference_rad=float(angle.max()),worlds_position_difference_over_1um=int(np.count_nonzero(pos>1e-6))))
 rows.append(dict(episode=ep,frames=n,checkpoint_sha256=a_report['checkpoint_sha256'],source_manifest_sha256=a_report['source_manifest_sha256'],first_saved_control_buffers=controls,first_saved_object_and_palm_pose_differences=frames,all_three_motor_target_fields_first_saved_sample_exact=all(controls[k]['first_saved_sample_bit_exact'] for k in ['motor_target','applied_motor_position_target','motor_velocity_target'])))
 for p in paths:assert sha(p)==before[str(p.relative_to(S))]
report=dict(schema='v2d.saved-replay-first-control-interval-audit.v1',producer_sha256=sha(Path(__file__)),input_sha256=inputs,rows=rows,first_saved_sample_is_after_full_control_interval=True,initial_reset_state_recorded=False,first_substep_state_recorded=False,physics_rerun=False,source_inputs_changed=False,new_training_steps=0,variability_cause_proven=False,all_inputs_verified_after=True,wall_seconds=time.monotonic()-started)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(rows,indent=2))

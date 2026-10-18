"""Freeze a separate inferred-command candidate and run the unchanged full audits.

Independent URDF source projection and motion checks accompany the original
native auditors and the committed 1 mm complete-task gate. No physics or PPO.
"""
import hashlib
import json
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

S=Path(__file__).resolve().parents[1]
R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(S/'cpu-native-runtime-v1/src'),str(R/'scripts/dex3_training'),str(R/'scripts/track3')]
from v2d_dex3_flashchord.kinematics import URDFKinematics
from check_native_scene import compile_scene
import audit_hand_reference_contacts as hands
import audit_reference_contacts as objects
from reference_contact_gate import assess

P=S/'scored-cohort-14-results/policy-e037/frozen'
proposal=S/'articulated-source-contact-trajectory-v2/inferred-robot-command-candidate.npz'
solver_report=S/'articulated-source-contact-trajectory-v2/report.json'
OUT=S/'articulated-source-contact-native-validation-v1';OUT.mkdir()
inputs=OUT/'frozen'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((P/'manifest.json').read_text())
for name,h in manifest['files'].items():assert sha(P/name)==h
assert json.loads(solver_report.read_text())['hand_source_contact_motion_stage_passed']
with np.load(P/'targets.npz',allow_pickle=False) as z:original={k:z[k].copy() for k in z.files}
with np.load(proposal,allow_pickle=False) as z:
    candidate=z['robot_qpos'].copy();normalized=z['normalized_corrections'].copy()
    np.testing.assert_array_equal(z['frame_indices'],original['frame_indices'])
    np.testing.assert_array_equal(z['timestamps_s'],original['timestamps_s'])
    np.testing.assert_array_equal(z['source_confidence_valid'],original['confidence_valid'])
shutil.copytree(P,inputs)
new={k:v.copy() for k,v in original.items()};new['robot_qpos']=candidate
kin={side:URDFKinematics(inputs/f'robot/urdfs/g1/dex3_{side}.urdf') for side in ['left','right']}
for si,side in enumerate(['left','right']):
    new[f'{side}_frames']=np.stack([np.stack([poses[name][:3,3] for name in manifest['frame_names'][side]]) for command in candidate[:,si] for poses in [kin[side].poses(command[:7],dict(zip(manifest['joint_names'][side],command[7:],strict=True)))]])
np.savez_compressed(inputs/'targets.npz',**new)
manifest['files']['targets.npz']=sha(inputs/'targets.npz')
manifest['inferred_contact_command_repair']=dict(method='source-constrained conservative native convex separation with articulated temporal SQP',original_manifest_sha256=sha(P/'manifest.json'),candidate_sha256=sha(proposal),producer_sha256=sha(Path(__file__)),solver_report_sha256=sha(solver_report),source_observations_changed=False,source_confidence_changed=False,object_targets_changed=False,contact_labels_fabricated=False,physical_task_accepted=False)
(inputs/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
report=dict(schema='v2d.articulated-trajectory-independent-validation.v1',episode=37,producer_sha256=sha(Path(__file__)),original_manifest_sha256=sha(P/'manifest.json'),candidate_manifest_sha256=sha(inputs/'manifest.json'),candidate_sha256=sha(proposal),source_confidence_changed=False,object_targets_changed=False,physics_stepped=False,new_training_steps=0,physical_task_accepted=False,trajectory_promoted_to_training=False,full_task_gate_passed=False)
save=lambda:(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
started=time.monotonic();save()
try:
    # Actual fresh compiler product; reuse it only for the unchanged two audit
    # functions at this exact input path, as in the validated original control.
    compiled=compile_scene(inputs)
    hands.compile_scene=lambda candidate_path:compiled if candidate_path.resolve()==inputs.resolve() else (_ for _ in ()).throw(ValueError('Unexpected candidate input'))
    objects.compile_scene=hands.compile_scene
    hand=hands.audit(inputs.resolve());obj=objects.audit(inputs.resolve())
    for name,value in [('hand-reference-contacts.json',hand),('object-reference-contacts.json',obj)]:
        (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
    gate=assess(inputs/'manifest.json',manifest,obj,hand)
    (OUT/'complete-reference-contact-gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    report['native_contact_gate']=gate
    report['native_hand_body_fk_verified']=hand['all_native_hand_body_fk_verified']
    report['auditor_producer_sha256']={n:sha(R/'scripts/dex3_training'/n) for n in ['audit_hand_reference_contacts.py','audit_reference_contacts.py','check_native_scene.py']}
    report['contact_gate_producer_sha256']=sha(R/'scripts/track3/reference_contact_gate.py')
    report['fresh_compiler_all_geometry_buffers_match_control']=True
    import mujoco
    control=mujoco.MjModel.from_binary_path(str(S/'local-native-contact-control-v1/native-model.mjb'))
    current=compiled[2].mj_model
    geometry=[]
    for key in ['geom_type','geom_bodyid','geom_dataid','geom_pos','geom_quat','geom_size','mesh_vert','mesh_pos','mesh_quat']:
        a,b=np.asarray(getattr(control,key)),np.asarray(getattr(current,key))
        error=float(abs(a-b).max(initial=0))
        passed=a.shape==b.shape and error<1e-10;geometry.append(dict(array=key,maximum_absolute_error=error,passed=passed))
        report['fresh_compiler_all_geometry_buffers_match_control'] &= passed
    report['native_geometry_comparisons']=geometry
    bundle=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
    with np.load(bundle,allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
    stage=np.asarray(json.loads((P/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
    K=b['intrinsics'];projection=[]
    np.testing.assert_allclose(np.einsum('ij,tj->ti',stage,b['object_0_world'][:,:3,3],optimize=False),new['object_poses'][:,0,:3],atol=1e-8,rtol=0)
    def project(points,frame):
        c=b['camera_c2w'][frame]
        camera=np.einsum('ni,ij->nj',points-c[:3,3],c[:3,:3],optimize=False)
        assert np.isfinite(camera).all() and (camera[:,2]>.01).all()
        return np.c_[K[0,0]*camera[:,0]/camera[:,2]+K[0,2],K[1,1]*camera[:,1]/camera[:,2]+K[1,2]]
    for si,side in enumerate(['left','right']):
        before=[];after=[];max_point_regression=0.;observed_frames=[]
        for frame in range(len(candidate)):
            source=b['hand_joints_world'][frame,si,[0,4,8,12]]
            if not(b['hand_valid'][frame,si] and b['hand_directly_observed'][frame,si] and np.isfinite(source).all()):continue
            observed_frames.append(frame);uv=project(source,frame);errors=[]
            for commands in [original['robot_qpos'],candidate]:
                command=commands[frame,si];poses=kin[side].poses(command[:7],dict(zip(manifest['joint_names'][side],command[7:],strict=True)))
                points=np.stack([command[:3],poses[f'{side}_thumb_tip'][:3,3],poses[f'{side}_index_tip'][:3,3],poses[f'{side}_middle_tip'][:3,3]])
                errors.append(np.linalg.norm(project(np.einsum('ni,ij->nj',points,stage,optimize=False),frame)-uv,axis=1))
            before.extend(errors[0]);after.extend(errors[1]);max_point_regression=max(max_point_regression,float((errors[1]-errors[0]).max()))
        passed=bool(max_point_regression<=1e-6 and np.median(after)<=np.median(before) and np.quantile(after,.95)<=np.quantile(before,.95))
        projection.append(dict(hand=side,observed_points=len(before),source_metric_supported_frames=int(b['hand_metric_valid'][observed_frames,si].sum()),maximum_point_regression_px=max_point_regression,before_median_px=float(np.median(before)),after_median_px=float(np.median(after)),before_p95_px=float(np.quantile(before,.95)),after_p95_px=float(np.quantile(after,.95)),passed=passed))
    report['independent_source_projection']=projection
    report['source_fit_stage_passed']=all(v['passed'] for v in projection)
    velocity=[]
    for si,side in enumerate(['left','right']):
        root=ET.parse(inputs/f'robot/urdfs/g1/dex3_{side}.urdf').getroot()
        limits={j.attrib['name']:j.find('limit').attrib for j in root.findall('joint') if j.attrib['type']!='fixed'}
        for j,name in enumerate(manifest['joint_names'][side]):
            lo,hi,limit=[float(limits[name][key]) for key in ['lower','upper','velocity']]
            values=candidate[:,si,7+j];speed=float(abs(np.diff(values)/np.diff(new['timestamps_s'])).max())
            velocity.append(dict(hand=side,joint=name,minimum_rad=float(values.min()),maximum_rad=float(values.max()),maximum_velocity_rad_s=speed,urdf_velocity_limit_rad_s=limit,passed=bool(values.min()>=lo-1e-6 and values.max()<=hi+1e-6 and speed<=limit+1e-6)))
    correction=candidate[:,:,:3]-original['robot_qpos'][:,:,:3]
    rotation=[]
    for si in range(2):
        original_r=Rotation.from_quat(original['robot_qpos'][:,si,3:7]);new_r=Rotation.from_quat(candidate[:,si,3:7])
        diff=(original_r.inv()*new_r).as_rotvec();np.testing.assert_allclose(diff,normalized[:,si,3:6]*.1,atol=1e-8,rtol=0)
        rotation.append(dict(hand=['left','right'][si],maximum_correction_rad=float(np.linalg.norm(diff,axis=1).max()),maximum_original_root_velocity_m_s=float(np.linalg.norm(np.diff(original['robot_qpos'][:,si,:3],axis=0),axis=-1).max()/0.05),maximum_candidate_root_velocity_m_s=float(np.linalg.norm(np.diff(candidate[:,si,:3],axis=0),axis=-1).max()/0.05),maximum_original_wrist_velocity_rad_s=float((original_r[:-1].inv()*original_r[1:]).magnitude().max()/0.05),maximum_candidate_wrist_velocity_rad_s=float((new_r[:-1].inv()*new_r[1:]).magnitude().max()/0.05)))
    report['finger_limit_velocity_checks']=velocity;report['wrist_motion']=rotation
    report['inferred_motion_bounds']=dict(maximum_root_correction_norm_m=float(np.linalg.norm(correction,axis=-1).max()),maximum_added_root_adjacent_component_m=float(abs(np.diff(correction,axis=0)).max()),maximum_added_root_second_component_m=float(abs(np.diff(correction,n=2,axis=0)).max()),maximum_added_rotation_adjacent_component_rad=float(abs(np.diff(normalized[:,:,3:6]*.1,axis=0)).max()),maximum_added_rotation_second_component_rad=float(abs(np.diff(normalized[:,:,3:6]*.1,n=2,axis=0)).max()),maximum_added_finger_adjacent_rad=float(abs(np.diff(candidate[:,:,7:]-original['robot_qpos'][:,:,7:],axis=0)).max()),maximum_added_finger_second_rad=float(abs(np.diff(candidate[:,:,7:]-original['robot_qpos'][:,:,7:],n=2,axis=0)).max()))
    m=report['inferred_motion_bounds']
    report['motion_limits_stage_passed']=all(v['passed'] for v in velocity) and m['maximum_root_correction_norm_m']<=.025 and m['maximum_added_root_adjacent_component_m']<=.0015+1e-7 and m['maximum_added_root_second_component_m']<=.0015+1e-7 and max(r['maximum_correction_rad'] for r in rotation)<=.15+1e-8 and m['maximum_added_rotation_adjacent_component_rad']<=.02+1e-8 and m['maximum_added_rotation_second_component_rad']<=.02+1e-8 and m['maximum_added_finger_adjacent_rad']<=.05+1e-8 and m['maximum_added_finger_second_rad']<=.05+1e-8
    report['hand_source_motion_stage_passed']=bool(gate['hand_slots_over_tolerance']==0 and report['source_fit_stage_passed'] and report['motion_limits_stage_passed'] and report['native_hand_body_fk_verified'] and report['fresh_compiler_all_geometry_buffers_match_control'])
    report['full_task_gate_passed']=gate['native_penetration_gate_passed']
    for name,h in manifest['files'].items():assert sha(inputs/name)==h
    old=json.loads((P/'manifest.json').read_text())
    for name,h in old['files'].items():assert sha(P/name)==h
    for key in original.keys()-{'robot_qpos','left_frames','right_frames'}:np.testing.assert_array_equal(new[key],original[key])
    report['original_frozen_closure_and_unchanged_targets_verified_after']=True
except BaseException as error:
    report['error']=repr(error);raise
finally:
    report['wall_seconds']=time.monotonic()-started;save()
    print(json.dumps({k:v for k,v in report.items() if k not in ['finger_limit_velocity_checks','native_geometry_comparisons']},indent=2),flush=True)

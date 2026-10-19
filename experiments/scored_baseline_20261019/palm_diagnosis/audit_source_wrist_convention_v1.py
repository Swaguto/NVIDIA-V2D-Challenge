"""Passive saved-geometry audit of right MANO/Dex3 palm orientation conventions.

No inference, source fit, command mutation, simulator step or gate relaxation.
An inferred RGB palm orientation is not a measured orientation/contact label.
"""
from pathlib import Path
import ast,datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(R/'scripts/dex3_training')]
import numpy as np
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.kinematics import URDFKinematics
O=S/'e037-source-wrist-convention-audit-v1';O.mkdir(exist_ok=True)
P=S/'e037-motion-constrained-wrist-native-v1/frozen';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();start=time.monotonic()
manifest=json.loads((P/'manifest.json').read_text())
for name,h in manifest['files'].items():assert sha(P/name)==h
with np.load(P/'targets.npz') as z:q=z['robot_qpos'][50,1].copy()
with np.load(P/'aligned-source-reference.npz') as z:
    camera=z['camera_c2w'][50].copy();flags={k:bool(z[k][50,1]) for k in ['hand_valid','hand_directly_observed','hand_metric_valid']}
assert not any(flags.values())
stage=np.asarray(json.loads((P/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
native_from_C=np.einsum('ij,jk->ik',stage,camera[:3,:3],optimize=False)
native_origin=np.einsum('ij,j->i',stage,camera[:3,3],optimize=False)
code=R/'robotic_grounding/source/robotic_grounding/robotic_grounding/retarget/hand_kinematics.py'
tree=ast.parse(code.read_text());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Dex3HandKinematics')
assignment=next(n for n in klass.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_R_MANO_PALM_RIGHT' for t in n.targets))
C=np.asarray(ast.literal_eval(assignment.value.args[0]),float)
np.testing.assert_array_equal(C,np.diag([-1, -1, 1]))
assert abs(np.linalg.det(C)-1)<1e-12
record_path=S/'e037-native-RGB-hand-source-audit-v1/canonical-C-unqualified-record.json';record=json.loads(record_path.read_text())
assert record['is_right'] and record['estimated'] and not any(record[k] for k in ['directly_observed','metric_valid','visual_valid'])
raw_path=S/'e037-native-RGB-hand-source-audit-v1/unfitted-source-geometry.npz'
with np.load(raw_path) as z:raw_j=z['joints_camera_C'].copy()
raw_R=Rotation.from_rotvec(record['mano']['global_orient']).as_matrix()
canonical=np.einsum('ni,ij->nj',raw_j-raw_j[0],raw_R,optimize=False)
def basis(forward,radial):
    f=forward/np.linalg.norm(forward);r=radial-np.dot(radial,f)*f;r/=np.linalg.norm(r)
    out=np.column_stack([f,r,np.cross(f,r)])
    np.testing.assert_allclose(out.T@out,np.eye(3),atol=1e-12,rtol=0);assert np.linalg.det(out)>0
    return out
human_basis=basis(canonical[9],canonical[5]-canonical[17])
kin=URDFKinematics(P/'robot/urdfs/g1/dex3_right.urdf');rest=kin.poses([0,0,0,0,0,0,1],dict.fromkeys(kin.names,0.))
index=rest['right_hand_index_0_link'][:3,3];middle=rest['right_hand_middle_0_link'][:3,3]
robot_basis=basis((index+middle)/2,index-middle)
anatomical_C=np.einsum('ij,kj->ik',human_basis,robot_basis,optimize=False)
def degrees(a,b):return float(np.rad2deg(Rotation.from_matrix(np.einsum('ji,jk->ik',a,b,optimize=False)).magnitude()))
anatomical_difference=degrees(C,anatomical_C)
canonical_forward_robot=np.einsum('ji,j->i',C,human_basis[:,0],optimize=False)
assert canonical_forward_robot[0]>.99
current_R=Rotation.from_quat(q[3:7]).as_matrix()
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:old_camera=z['camera_c2w'][50].copy();K=z['intrinsics'].copy()
image_path=S/'e037-watcloud-mask-refinement-handoff-v1/image-observations/right/000050.json';image=json.loads(image_path.read_text());pixels=np.asarray(image['keypoints']);ids=np.array([4,8,12]);select=pixels[ids,2]>0
assert not image['metric_valid'] and not image['directly_observed']
def image_errors(command):
    poses=kin.poses(command[:7],dict(zip(manifest['joint_names']['right'],command[7:],strict=True)))
    points=np.stack([poses[f'right_{digit}_tip'][:3,3] for digit in ['thumb','index','middle']])[select]
    y=np.einsum('ni,ij->nj',points,stage,optimize=False)
    cam=np.einsum('ni,ij->nj',y-camera[:3,3],camera[:3,:3],optimize=False);assert (cam[:,2]>.01).all()
    uv=np.c_[K[0,0]*cam[:,0]/cam[:,2]+K[0,2],K[1,1]*cam[:,1]/cam[:,2]+K[1,2]]
    return np.linalg.norm(uv-pixels[ids[select],:2],axis=1)
input_paths=[P/'manifest.json',P/'targets.npz',P/'aligned-source-reference.npz',P/'command-proposal-report.json',code,R/'robotic_grounding/scripts/retarget/track3_to_dex3.py',record_path,raw_path,B,image_path]
rows=[];rotations=[];base_errors=image_errors(q)
for label,directory in [('raw_RGB',None),('multiview_fit_unqualified','e037-native-RGB-multiview-source-fit-v1'),('hard_skin_fit_failed','e037-native-RGB-hard-skin-coverage-fit-v2')]:
    if directory is None:source_R=raw_R;joints=raw_j
    else:
        path=S/directory/'source-hand-candidate.npz';input_paths.append(path)
        with np.load(path) as z:
            source_R=Rotation.from_rotvec(z['mano_pose'][0,:3]).as_matrix()
            joints=np.einsum('ni,ij->nj',z['joints_world'][0]-old_camera[:3,3],old_camera[:3,:3],optimize=False)
    local=np.einsum('ni,ij->nj',joints-joints[0],source_R,optimize=False)
    local_basis=basis(local[9],local[5]-local[17]);assert degrees(human_basis,local_basis)<1e-4
    target=np.einsum('ij,jk,kl->il',native_from_C,source_R,C,optimize=False);rotations.append(target)
    command=q.copy();command[3:7]=Rotation.from_matrix(target).as_quat()
    errors=image_errors(command)
    wrist_position=np.einsum('ij,j->i',native_from_C,joints[0],optimize=False)+native_origin
    rows.append(dict(source=label,source_was_accepted=False,model_orientation_is_inferred=True,current_to_source_robot_palm_angle_deg=degrees(current_R,target),anatomical_basis_drift_deg=degrees(human_basis,local_basis),current_tip_errors_px=base_errors.tolist(),source_orientation_only_tip_errors_px=errors.tolist(),source_orientation_tip_median_px=float(np.median(errors)),all_inferred_tip_errors_nonregressing=bool(np.all(errors<=base_errors+1e-6)),estimated_human_wrist_minus_robot_root_m=(wrist_position-q[:3]).tolist(),estimated_wrist_translation_is_not_metric_truth=True))
pairwise={rows[i]['source']+'/'+rows[j]['source']:degrees(a,b) for i,a in enumerate(rotations) for j,b in enumerate(rotations) if j>i}
np.savez_compressed(O/'saved-palm-orientation-diagnostics.npz',current_robot_root_xyzw=q[:7],source_robot_palm_rotations_native=np.stack(rotations),native_from_C=native_from_C,human_canonical_palm_basis=human_basis,robot_URDF_rest_palm_basis=robot_basis,current_right_correction=C,anatomical_basis_correction=anatomical_C)
report=dict(schema='v2d.saved-source-wrist-convention-audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),inputs_sha256={str(p):sha(p) for p in input_paths},frame=50,side='right',source_flags=flags,raw_RGB_record_is_estimated_and_unqualified=True,corrected_canonical_finger_direction_robot=canonical_forward_robot.tolist(),actual_right_correction_degrees=float(np.rad2deg(Rotation.from_matrix(C).magnitude())),actual_right_correction_is_proper_rotation=True,canonical_forward_dot_URDF_forward=float(np.dot(canonical_forward_robot,robot_basis[:,0])),anatomical_basis_vs_existing_correction_deg=anatomical_difference,anatomical_basis_difference_is_not_an_adapter_calibration=True,existing_right_adapter_180_degree_flip_error_established=False,orientation_rows=rows,saved_source_orientation_pairwise_degrees=pairwise,source_fit_failure_reports_unchanged=True,source_heldout_samples_or_authority_changed=False,command_files_or_defaults_changed=False,new_hand_model_or_LBS_inference=False,new_GPU_work=False,new_PPO_steps=0,new_native_or_motion_certificate=False,full_source_hand_qualified=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start)
for name,h in manifest['files'].items():assert sha(P/name)==h
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

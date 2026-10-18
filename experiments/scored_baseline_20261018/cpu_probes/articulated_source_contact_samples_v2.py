"""CPU feasibility probe at three previously recorded worst contact knots.

Root translation, local wrist rotation and seven finger angles are optimized
together against exact native distances and original source projections. This
does not supply a full trajectory, force closure, policy or training approval.
"""
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

S=Path(__file__).resolve().parents[1]
R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0,str(R/'scripts/dex3_training'))
from audit_hand_reference_contacts import canonical_id

P=S/'scored-cohort-14-results/policy-e037/frozen'
OUT=S/'articulated-source-contact-samples-v2'
OUT.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
native_proof=S/'local-native-contact-control-v1/report.json'
assert json.loads(native_proof.read_text())['all_full_reference_controls_passed']
manifest=json.loads((P/'manifest.json').read_text())
for name,h in manifest['files'].items():
    assert sha(P/name)==h
model_path=S/'local-native-contact-control-v1/native-model.mjb'
model=mujoco.MjModel.from_binary_path(str(model_path))
data=mujoco.MjData(model)
bodies=[model.body(i).name for i in range(model.nbody)]
joints=[model.joint(i).name for i in range(model.njnt)]
with np.load(P/'targets.npz',allow_pickle=False) as z:
    original={k:z[k].copy() for k in z.files}
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz',allow_pickle=False) as z:
    source={k:z[k].copy() for k in z.files}
stage=np.asarray(json.loads((P/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
K=source['intrinsics']
binding={}
for si,side in enumerate(['left','right']):
    wrist=[canonical_id(joints,f'{side}_wrist_{a}') for a in ['x','y','z','rotvec']]
    fingers=[canonical_id(joints,n) for n in manifest['joint_names'][side]]
    root=ET.parse(P/f'robot/urdfs/g1/dex3_{side}.urdf').getroot()
    limits={j.attrib['name']:j.find('limit').attrib for j in root.findall('joint') if j.attrib['type']!='fixed'}
    binding[side]=dict(si=si,wrist=wrist,fingers=fingers,
        dofs=np.r_[[int(model.jnt_dofadr[j]) for j in wrist[:3]],np.arange(int(model.jnt_dofadr[wrist[3]]),int(model.jnt_dofadr[wrist[3]])+3),[int(model.jnt_dofadr[j]) for j in fingers]].astype(int),
        points=[canonical_id(bodies,n) for n in [f'{side}_hand_palm_link',f'{side}_thumb_tip',f'{side}_index_tip',f'{side}_middle_tip']],
        hand_geoms=[g for g,b in enumerate(model.geom_bodyid) if bodies[b].startswith('dex3_'+side+'_')],
        finger_lower=np.array([float(limits[n]['lower']) for n in manifest['joint_names'][side]]),
        finger_upper=np.array([float(limits[n]['upper']) for n in manifest['joint_names'][side]]))
target_bodies={canonical_id(bodies,'wooden_spoon_object'):'wooden_spoon',canonical_id(bodies,'support'):'support'}
target_geoms=[g for g,b in enumerate(model.geom_bodyid) if b in target_bodies]
object_bid=next(b for b,n in target_bodies.items() if n=='wooden_spoon')
object_adr=int(model.jnt_qposadr[model.body_jntadr[object_bid]])
scales=np.r_[np.full(3,.01),np.full(10,.1)]
report=dict(schema='v2d.articulated-source-contact-samples.v2',episode=37,
            producer_sha256=sha(Path(__file__)),frozen_manifest_sha256=sha(P/'manifest.json'),
            native_control_sha256=sha(native_proof),native_model_sha256=sha(model_path),
            predetermined_samples=[dict(frame=18,hand='right',reason='worst hand/spoon penetration'),dict(frame=23,hand='right',reason='worst right hand/support penetration'),dict(frame=64,hand='left',reason='worst left hand/support penetration')],
            bounds=dict(root_norm_m=.025,root_component_m=.014,wrist_rotation_norm_rad=.15,finger_change_rad=.25,optimization_contact_clearance_m=.0005,acceptance_penetration_m=.001),
            source_inputs_modified=False,source_confidence_modified=False,object_targets_modified=False,
            physics_stepped=False,new_training_steps=0,trajectory_promoted=False,physical_task_accepted=False,
            full_trajectory_or_velocity_validation_completed=False,
            numerical_change='Pixel-norm constraints replace equivalent squared errors; central differences use 1 um root/1e-5 rad increments, with a stricter 0.001 px interior margin. Acceptance gates unchanged.',cases=[])
save=lambda:(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
started=time.monotonic()
selected=[]
try:
    for case in report['predetermined_samples']:
        frame,side=case['frame'],case['hand'];bind=binding[side];si=bind['si']
        assert source['hand_valid'][frame,si] and source['hand_directly_observed'][frame,si]
        observed=source['hand_joints_world'][frame,si,[0,4,8,12]]
        assert np.isfinite(observed).all()
        camera=source['camera_c2w'][frame]
        world_to_camera=np.einsum('ji,kj->ik',camera[:3,:3],stage,optimize=False)
        def pixels(camera_points):
            assert np.isfinite(camera_points).all() and (camera_points[:,2]>.01).all()
            return np.c_[K[0,0]*camera_points[:,0]/camera_points[:,2]+K[0,2],K[1,1]*camera_points[:,1]/camera_points[:,2]+K[1,2]]
        observed_camera=np.einsum('ni,ij->nj',observed-camera[:3,3],camera[:3,:3],optimize=False)
        observed_uv=pixels(observed_camera)
        base=model.qpos0.copy()
        obj=original['object_poses'][frame,0]
        base[object_adr:object_adr+3]=obj[:3];base[object_adr+3:object_adr+7]=obj[[6,3,4,5]]
        for hand,h in binding.items():
            q=original['robot_qpos'][frame,h['si']]
            for col,j in enumerate(h['wrist'][:3]):base[int(model.jnt_qposadr[j])]=q[col]
            adr=int(model.jnt_qposadr[h['wrist'][3]]);base[adr:adr+4]=q[[6,3,4,5]]
            for j,v in zip(h['fingers'],q[7:],strict=True):base[int(model.jnt_qposadr[j])]=v
        pairs=[(a,b) for a in bind['hand_geoms'] for b in target_geoms]
        cache={}
        def evaluate(x):
            if time.monotonic()-started>180:raise TimeoutError('Bounded CPU articulated probe deadline')
            if 'x' in cache and np.array_equal(x,cache['x']):return cache['value']
            qpos=base.copy();v=np.zeros(model.nv);v[bind['dofs']]=x*scales
            mujoco.mj_integratePos(model,qpos,v,1.)
            data.qpos[:]=qpos;mujoco.mj_fwdPosition(model,data)
            point_camera=np.einsum('ni,ji->nj',data.xpos[bind['points']],world_to_camera,optimize=False)-camera[:3,:3].T.dot(camera[:3,3])
            uv=pixels(point_camera)
            errors=np.linalg.norm(uv-observed_uv,axis=1)
            gaps=np.array([mujoco.mj_geomDistance(model,data,a,b,.1,None) for a,b in pairs])
            command=np.r_[[qpos[int(model.jnt_qposadr[j])] for j in bind['wrist'][:3]],qpos[int(model.jnt_qposadr[bind['wrist'][3]]):int(model.jnt_qposadr[bind['wrist'][3]])+4][[1,2,3,0]],[qpos[int(model.jnt_qposadr[j])] for j in bind['fingers']]]
            value=(errors,gaps,command)
            cache.update(x=x.copy(),value=value)
            return value
        zero=np.zeros(13);before_errors,before_gaps,before_command=evaluate(zero)
        np.testing.assert_allclose(before_command,original['robot_qpos'][frame,si],atol=1e-12,rtol=0)
        np.savez_compressed(OUT/f'f{frame}-{side}-baseline-distances.npz',pairs=pairs,signed_distances_m=before_gaps)
        # Constraints cover every native hand/target convex pair, not only old contacts.
        f0=original['robot_qpos'][frame,si,7:]
        lower=np.r_[np.full(3,-1.4),np.full(3,-1.5),np.maximum(-2.5,(bind['finger_lower']-f0)/.1)]
        upper=np.r_[np.full(3,1.4),np.full(3,1.5),np.minimum(2.5,(bind['finger_upper']-f0)/.1)]
        def constraints(x):
            errors,gaps,_=evaluate(x)
            return np.r_[(gaps-.0005)/.01,(before_errors-np.minimum(.001,.01*before_errors)-errors)/np.maximum(before_errors,.05),
                          (.025**2-np.dot(x[:3]*.01,x[:3]*.01))/.025**2,
                          (.15**2-np.dot(x[3:6]*.1,x[3:6]*.1))/.15**2]
        def constraint_jacobian(x,step=1e-4):
            columns=[]
            for index in range(len(x)):
                dx=np.zeros_like(x);dx[index]=step
                columns.append((constraints(x+dx)-constraints(x-dx))/(2*step))
            return np.stack(columns,axis=1)
        starts=[zero.copy()]
        ray=np.einsum('ij,j->i',stage,camera[:3,2],optimize=False)
        start=zero.copy();start[:3]=-ray*.008/.01;starts.append(start)
        start=zero.copy();start[6:]=np.clip(-f0,.0,.1)/.1;starts.append(np.clip(start,lower,upper))
        row=dict(**case,source_metric_supported=bool(source['hand_metric_valid'][frame,si]),
                 source_observed=True,native_pair_count=len(pairs),
                 before_pixel_errors=before_errors.tolist(),before_max_penetration_m=max(0.,float(-before_gaps.min())),attempts=[])
        jac=constraint_jacobian(zero)
        jac_small=constraint_jacobian(zero,step=1e-5)
        row['central_jacobian_step_comparison']=dict(maximum_absolute_difference=float(abs(jac-jac_small).max()),relative_frobenius_difference=float(np.linalg.norm(jac-jac_small)/max(np.linalg.norm(jac),1e-12)),root_step_m=1e-6,angle_step_rad=1e-5)
        accepted=[]
        for initial in starts:
            t=time.monotonic()
            solution=minimize(lambda x:float(np.dot(x,x)),np.clip(initial,lower,upper),jac=lambda x:2*x,method='SLSQP',bounds=list(zip(lower,upper,strict=True)),constraints=[dict(type='ineq',fun=constraints,jac=constraint_jacobian)],options=dict(maxiter=200,ftol=1e-11))
            errors,gaps,command=evaluate(solution.x)
            penetration=max(0.,float(-gaps.min()))
            source_pass=bool((errors<=before_errors+1e-6).all() and np.median(errors)<=np.median(before_errors) and np.quantile(errors,.95)<=np.quantile(before_errors,.95))
            bound_pass=bool(np.linalg.norm(solution.x[:3]*.01)<=.025+1e-8 and np.linalg.norm(solution.x[3:6]*.1)<=.15+1e-8 and (command[7:]>=bind['finger_lower']-1e-7).all() and (command[7:]<=bind['finger_upper']+1e-7).all() and abs(command[7:]-f0).max()<=.25+1e-7)
            passes=bool(solution.success and source_pass and bound_pass and penetration<=.001)
            attempt=dict(optimizer_success=bool(solution.success),message=str(solution.message),source_projection_passed=source_pass,command_bounds_passed=bound_pass,sample_contact_source_passed=passes,
                         after_pixel_errors=errors.tolist(),maximum_penetration_m=penetration,root_correction_norm_m=float(np.linalg.norm(solution.x[:3]*.01)),wrist_rotation_correction_rad=float(np.linalg.norm(solution.x[3:6]*.1)),maximum_finger_change_rad=float(abs(command[7:]-f0).max()),function_evaluations=int(solution.nfev),wall_seconds=time.monotonic()-t,optimizer_variables=solution.x.tolist())
            row['attempts'].append(attempt)
            if passes:accepted.append((float(np.dot(solution.x,solution.x)),command.copy()))
            report['cases']=report['cases'][:len(selected)]+[row];save()
            print(json.dumps(dict(frame=frame,hand=side,accepted=passes,source_passed=source_pass,penetration_mm=penetration*1000,wall_seconds=attempt['wall_seconds'])),flush=True)
        best=min(accepted,key=lambda x:x[0],default=None)
        row['any_sample_passed']=best is not None
        selected.append(dict(frame=frame,side=si,command=original['robot_qpos'][frame,si].copy() if best is None else best[1]))
        report['cases'][-1]=row;save()
    np.savez_compressed(OUT/'sample-command-proposals.npz',frames=[v['frame'] for v in selected],sides=[v['side'] for v in selected],robot_qpos=np.stack([v['command'] for v in selected]))
    report['accepted_samples']=sum(r['any_sample_passed'] for r in report['cases'])
    for name,h in manifest['files'].items():assert sha(P/name)==h
    report['frozen_source_closure_verified_after']=True
except BaseException as error:
    report['error']=repr(error);raise
finally:
    report['wall_seconds']=time.monotonic()-started;save()

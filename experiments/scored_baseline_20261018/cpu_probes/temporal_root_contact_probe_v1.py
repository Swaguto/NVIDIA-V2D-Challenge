"""Bounded native contact projection of inferred wrist translation commands.

All 151 knots are solved together. Fingers, wrist orientations, objects and
observation flags are fixed. This contact-stage diagnostic cannot qualify the
whole task: original object/support intersections and source-fit checks remain.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation

S = Path(__file__).resolve().parents[1]
R = Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0,str(R/'scripts/dex3_training'))
from audit_hand_reference_contacts import canonical_id
from v2d_dex3_flashchord.kinematics import URDFKinematics

P = S/'scored-cohort-14-results/policy-e037'
inputs = P/'frozen'
OUT = S/'temporal-root-contact-probe-v1'
OUT.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
control_path = S/'local-native-contact-control-v1/report.json'
control = json.loads(control_path.read_text())
assert control['all_full_reference_controls_passed']
m = json.loads((inputs/'manifest.json').read_text())
for name,h in m['files'].items():
    assert sha(inputs/name)==h
with np.load(inputs/'targets.npz',allow_pickle=False) as z:
    original = {k:z[k].copy() for k in z.files}
q = original['robot_qpos']
count = len(q)
assert count == 151
model_path = S/'local-native-contact-control-v1/native-model.mjb'
model = mujoco.MjModel.from_binary_path(str(model_path))
data = mujoco.MjData(model)
bodies = [model.body(i).name for i in range(model.nbody)]
joints = [model.joint(i).name for i in range(model.njnt)]
joint_ids,wrist_ids,hand_bodies = {},{},{}
for si,side in enumerate(['left','right']):
    kin = URDFKinematics(inputs/f'robot/urdfs/g1/dex3_{side}.urdf')
    link_names = kin.poses(q[0,si,:7],dict(zip(m['joint_names'][side],q[0,si,7:],strict=True)))
    hand_bodies[side] = {bid for bid,n in enumerate(bodies) if any(n==link or n.endswith('/'+link) or n.endswith('_'+link) for link in link_names)}
    wrist_ids[side] = [canonical_id(joints,f'{side}_wrist_{a}') for a in ['x','y','z','rotvec']]
    joint_ids[side] = [canonical_id(joints,n) for n in m['joint_names'][side]]
object_bodies = {canonical_id(bodies,'wooden_spoon/object'):'wooden_spoon',canonical_id(bodies,'support'):'support'}
object_bid = next(b for b,n in object_bodies.items() if n=='wooden_spoon')
object_adr = int(model.jnt_qposadr[model.body_jntadr[object_bid]])
started = time.monotonic()
report = dict(schema='v2d.temporal-native-root-contact-probe.v1',episode=37,
              producer_sha256=sha(Path(__file__)),control_report_sha256=sha(control_path),
              native_model_sha256=sha(model_path),frozen_manifest_sha256=sha(inputs/'manifest.json'),
              source_observations_modified=False,source_confidence_flags_modified=False,
              finger_targets_modified=False,wrist_orientations_modified=False,
              object_targets_modified=False,physics_stepped=False,new_training_steps=0,
              trajectory_promoted=False,physical_task_accepted=False,
              source_fit_qualified=False,iterations=[],
              limits=dict(root_correction_norm_m=.025,root_component_bound_m=.014,
                          adjacent_correction_component_step_m=.0015,
                          second_difference_correction_component_m=.0015,
                          contact_target_clearance_m=.0005,
                          contact_acceptance_penetration_m=.001))
save = lambda: (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')


def set_frame(frame,offset):
    data.qpos[:] = model.qpos0
    obj = original['object_poses'][frame,0]
    data.qpos[object_adr:object_adr+3] = obj[:3]
    data.qpos[object_adr+3:object_adr+7] = obj[[6,3,4,5]]
    for si,side in enumerate(['left','right']):
        target = q[frame,si]
        for col,jid in enumerate(wrist_ids[side][:3]):
            data.qpos[int(model.jnt_qposadr[jid])] = target[col]+offset[si,col]
        adr = int(model.jnt_qposadr[wrist_ids[side][3]])
        data.qpos[adr:adr+4] = target[[6,3,4,5]]
        for jid,val in zip(joint_ids[side],target[7:],strict=True):
            data.qpos[int(model.jnt_qposadr[jid])] = val
    mujoco.mj_fwdPosition(model,data)


def contacts(offsets):
    records = []
    slots = {}
    for frame in range(count):
        set_frame(frame,offsets[frame])
        for contact in data.contact[:data.ncon]:
            a,b = [int(model.geom_bodyid[g]) for g in [contact.geom1,contact.geom2]]
            for si,side in enumerate(['left','right']):
                if a in hand_bodies[side] and b in object_bodies:
                    target,sign = b,-1
                elif b in hand_bodies[side] and a in object_bodies:
                    target,sign = a,1
                else:
                    continue
                gap = float(contact.dist)
                gradient = sign*np.asarray(contact.frame[:3]).copy()
                records.append(dict(frame=frame,hand=side,side=si,target=object_bodies[target],
                                    gap=gap,gradient=gradient,geoms=(int(contact.geom1),int(contact.geom2))))
                slots[(frame,si)] = max(slots.get((frame,si),0.),max(0.,-gap))
    return records,slots


offsets = np.zeros((count,2,3))
try:
    records,slots = contacts(offsets)
    report['control'] = dict(hand_frame_slots_over_1mm=sum(v>.001 for v in slots.values()),
                             maximum_penetration_m=max(slots.values(),default=0.))
    # Verify normal/translation signs against the native geometric distance,
    # without stepping physics or relying on the optimizer implementation.
    derivative_checks=[]
    for record in sorted(records,key=lambda r:r['gap'])[:6]:
        frame,si = record['frame'],record['side']
        numeric=[]
        for axis in range(3):
            samples=[]
            for sign in [-1,1]:
                delta=offsets[frame].copy();delta[si,axis]+=sign*1e-6
                set_frame(frame,delta)
                samples.append(mujoco.mj_geomDistance(model,data,*record['geoms'],.1,None))
            numeric.append((samples[1]-samples[0])/2e-6)
        error=float(np.max(abs(np.asarray(numeric)-record['gradient'])))
        derivative_checks.append(dict(frame=frame,hand=record['hand'],maximum_derivative_error=error))
    report['native_contact_derivative_checks']=derivative_checks
    assert max(v['maximum_derivative_error'] for v in derivative_checks)<1e-3
    n=count*6
    eye=sparse.eye(n,format='csc')
    d1=sparse.kron(sparse.diags([-np.ones(count-1),np.ones(count-1)],[0,1],shape=(count-1,count)),sparse.eye(6),format='csc')
    d2=sparse.kron(sparse.diags([np.ones(count-2),-2*np.ones(count-2),np.ones(count-2)],[0,1,2],shape=(count-2,count)),sparse.eye(6),format='csc')
    H=2*(eye+20*d1.T@d1+100*d2.T@d2)
    fixed=sparse.vstack([eye[:6],eye[-6:]],format='csc')
    base=sparse.vstack([eye,d1,d2,fixed],format='csc')
    limit=np.r_[np.full(n,.014),np.full(d1.shape[0],.0015),np.full(d2.shape[0],.0015),np.zeros(12)]
    constraints=[]
    for iteration in range(12):
        records,slots = contacts(offsets)
        for record in records:
            if record['gap'] >= .0005:
                continue
            f,si=record['frame'],record['side']
            constraints.append((f*6+si*3,record['gradient'],.0005-record['gap']+np.dot(record['gradient'],offsets[f,si])))
        row_indices=[];col_indices=[];values=[];rhs=[]
        for i,(col,gradient,bound) in enumerate(constraints):
            row_indices.extend([i]*3);col_indices.extend([col,col+1,col+2]);values.extend(gradient);rhs.append(bound)
        C=sparse.csc_matrix((values,(row_indices,col_indices)),shape=(len(constraints),n))
        A=sparse.vstack([base,C],format='csc')
        solver=osqp.OSQP();solver.setup(P=sparse.triu(H,format='csc'),q=np.zeros(n),A=A,
             l=np.r_[-limit,rhs],u=np.r_[limit,np.full(len(rhs),np.inf)],
             eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polishing=True,verbose=False)
        result=solver.solve(raise_error=False)
        entry=dict(iteration=iteration,status=result.info.status,constraints=len(constraints),
                   before_slots_over_1mm=sum(v>.001 for v in slots.values()))
        if result.info.status!='solved':
            report['iterations'].append(entry);save();break
        candidate=result.x.reshape(count,2,3)
        candidate_records,candidate_slots=contacts(candidate)
        entry.update(after_slots_over_1mm=sum(v>.001 for v in candidate_slots.values()),
                     maximum_penetration_m=max(candidate_slots.values(),default=0.),
                     max_correction_m=float(np.linalg.norm(candidate,axis=-1).max()))
        report['iterations'].append(entry);offsets=candidate;save();print(json.dumps(entry),flush=True)
        if max(candidate_slots.values(),default=0.)<=1e-6:
            break
    records,slots=contacts(offsets)
    candidate=q.copy();candidate[:,:,:3]+=offsets
    np.testing.assert_array_equal(candidate[:,:,3:],q[:,:,3:])
    limits_ok=True
    for si,side in enumerate(['left','right']):
        root=ET.parse(inputs/f'robot/urdfs/g1/dex3_{side}.urdf').getroot()
        bounds={j.attrib['name']:j.find('limit').attrib for j in root.findall('joint') if j.attrib['type']!='fixed'}
        for j,name in enumerate(m['joint_names'][side]):
            lo,hi,velocity=[float(bounds[name][k]) for k in ['lower','upper','velocity']]
            values=candidate[:,si,7+j]
            limits_ok &= bool(values.min()>=lo-1e-6 and values.max()<=hi+1e-6 and (abs(np.diff(values)/np.diff(original['timestamps_s']))<=velocity+1e-6).all())
    report['final']=dict(hand_frame_slots_over_1mm=sum(v>.001 for v in slots.values()),
                         maximum_penetration_m=max(slots.values(),default=0.),
                         maximum_correction_norm_m=float(np.linalg.norm(offsets,axis=-1).max()),
                         maximum_adjacent_correction_component_m=float(abs(np.diff(offsets,axis=0)).max()),
                         maximum_second_difference_correction_component_m=float(abs(np.diff(offsets,n=2,axis=0)).max()),
                         original_finger_limits_and_velocities_pass=limits_ok,
                         unresolved_object_support_control_retained=True)
    report['contact_stage_passed']=bool(report['final']['hand_frame_slots_over_1mm']==0 and
              report['final']['maximum_correction_norm_m']<=.025 and
              report['final']['maximum_adjacent_correction_component_m']<=.0015+1e-7 and
              report['final']['maximum_second_difference_correction_component_m']<=.0015+1e-7 and limits_ok)
    np.savez_compressed(OUT/'inferred-robot-command-candidate.npz',robot_qpos=candidate,
                        root_translation_correction_m=offsets,source_confidence_valid=original['confidence_valid'],
                        frame_indices=original['frame_indices'],timestamps_s=original['timestamps_s'])
    report['candidate_sha256']=sha(OUT/'inferred-robot-command-candidate.npz')
    for name,h in m['files'].items():
        assert sha(inputs/name)==h
except BaseException as error:
    report['error']=repr(error);raise
finally:
    report['wall_seconds']=time.monotonic()-started;save()
    print(json.dumps({k:v for k,v in report.items() if k not in ['iterations','native_contact_derivative_checks']}),flush=True)

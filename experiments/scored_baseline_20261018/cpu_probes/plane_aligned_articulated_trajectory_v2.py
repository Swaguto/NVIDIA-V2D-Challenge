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

P=S/'plane-aligned-native-reference-v1/frozen'
OUT=S/'plane-aligned-articulated-trajectory-v2'
OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
native_proof=S/'plane-aligned-native-reference-v1/report.json'
assert json.loads(native_proof.read_text())['native_hand_body_fk_verified']
assert all(r['passed'] for r in json.loads(native_proof.read_text())['fresh_geometry_checks'])
manifest=json.loads((P/'manifest.json').read_text())
for name,h in manifest['files'].items():
    assert sha(P/name)==h
model_path=S/'plane-aligned-native-reference-v1/native-model.mjb'
model=mujoco.MjModel.from_binary_path(str(model_path))
data=mujoco.MjData(model)
bodies=[model.body(i).name for i in range(model.nbody)]
joints=[model.joint(i).name for i in range(model.njnt)]
with np.load(P/'targets.npz',allow_pickle=False) as z:
    original={k:z[k].copy() for k in z.files}
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz',allow_pickle=False) as z:
    source={k:z[k].copy() for k in z.files}
# Fresh morph corrections are relative to the ORIGINAL robot reference under
# the same inferred world alignment, never to the previous repaired hand.
aligned_reference=P/'aligned-source-reference.npz'
with np.load(aligned_reference,allow_pickle=False) as z:
    original['robot_qpos']=z['robot_qpos'].copy()
    source['camera_c2w']=z['camera_c2w'].copy()
    source['hand_joints_world']=z['hand_joints_world'].copy()
    for key in ['hand_valid','hand_directly_observed','hand_metric_valid']:
        np.testing.assert_array_equal(z[key],source[key])
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
from scipy import sparse
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import osqp

started=time.monotonic()
q=original['robot_qpos'];count=len(q);n=count*26
report=dict(schema='v2d.plane-aligned-articulated-source-reference.v2',episode=37,
    producer_sha256=sha(Path(__file__)),native_model_sha256=sha(model_path),
    frozen_manifest_sha256=sha(P/'manifest.json'),
    retained_historical_sample_control_sha256=sha(S/'articulated-source-contact-samples-v5/report.json'),
    source_aligned_reference_sha256=sha(aligned_reference),new_scene_requires_fresh_qualification=True,
    source_inputs_modified=False,source_confidence_modified=False,object_targets_modified=False,
    physics_stepped=False,new_training_steps=0,trajectory_promoted=False,physical_task_accepted=False,
    full_task_gate_passed=False,iterations=[],
    numerical_change='Generate worst-vertex cuts; full conservative surface checks still test every vertex. Finite-checked explicit contractions avoid platform matmul warnings. Source/contact/motion gates unchanged.',
    limits=dict(root_component_m=.014,root_norm_m=.025,wrist_rotation_norm_rad=.15,
                finger_change_rad=.25,adjacent_added_root_component_m=.0015,
                adjacent_added_rotation_component_rad=.02,adjacent_added_finger_rad=.05,
                second_added_root_component_m=.0015,second_added_rotation_component_rad=.02,
                second_added_finger_rad=.05,contact_plane_clearance_m=.0005,acceptance_penetration_m=.001))
save=lambda:(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save()
def mm(a,b):
    a,b=np.asarray(a),np.asarray(b)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    if a.ndim==2 and b.ndim==2:out=np.einsum('ij,jk->ik',a,b,optimize=False)
    elif a.ndim==2 and b.ndim==1:out=np.einsum('ij,j->i',a,b,optimize=False)
    elif a.ndim==1 and b.ndim==2:out=np.einsum('i,ij->j',a,b,optimize=False)
    else:raise ValueError('Unsupported finite contraction')
    assert np.isfinite(out).all()
    return out
local_vertices={}
for geom in range(model.ngeom):
    assert model.geom_type[geom]==mujoco.mjtGeom.mjGEOM_MESH
    mesh=int(model.geom_dataid[geom]);adr=int(model.mesh_vertadr[mesh]);num=int(model.mesh_vertnum[mesh])
    points=np.asarray(model.mesh_vert[adr:adr+num],dtype=float)
    local_vertices[geom]=points[np.unique(ConvexHull(points).vertices)]
base=[]
for frame in range(count):
    row=model.qpos0.copy();obj=original['object_poses'][frame,0]
    row[object_adr:object_adr+3]=obj[:3];row[object_adr+3:object_adr+7]=obj[[6,3,4,5]]
    for side,h in binding.items():
        target=q[frame,h['si']]
        for col,j in enumerate(h['wrist'][:3]):row[int(model.jnt_qposadr[j])]=target[col]
        adr=int(model.jnt_qposadr[h['wrist'][3]]);row[adr:adr+4]=target[[6,3,4,5]]
        for j,v in zip(h['fingers'],target[7:],strict=True):row[int(model.jnt_qposadr[j])]=v
    base.append(row)

def deadline():
    if time.monotonic()-started>300:raise TimeoutError('Five-minute bounded CPU trajectory probe')

def native_qpos(frame,x):
    velocity=np.zeros(model.nv)
    for side,h in binding.items():velocity[h['dofs']]=x[h['si']]*scales
    position=base[frame].copy();mujoco.mj_integratePos(model,position,velocity,1.)
    return position

def set_frame(frame,x):
    data.qpos[:]=native_qpos(frame,x);mujoco.mj_fwdPosition(model,data)

def vertices(geom):
    return np.einsum('ni,ji->nj',local_vertices[geom],data.geom_xmat[geom].reshape(3,3),optimize=False)+data.geom_xpos[geom]

def tangent_map(frame,x,si):
    h=binding[['left','right'][si]]
    T=np.zeros((model.nv,13));T[h['dofs'],np.arange(13)]=scales
    for index in range(3,6):
        delta=np.zeros_like(x);delta[si,index]=1e-5
        minus,plus=native_qpos(frame,x-delta),native_qpos(frame,x+delta)
        velocity=np.zeros(model.nv);mujoco.mj_differentiatePos(model,velocity,2e-5,minus,plus)
        T[:,index]=velocity
    return T

source_specs={};planes={};original_errors={0:[],1:[]}
for frame in range(count):
    set_frame(frame,np.zeros((2,13)))
    camera=source['camera_c2w'][frame]
    A=np.einsum('ji,kj->ik',camera[:3,:3],stage,optimize=False)
    offset=np.einsum('ij,j->i',camera[:3,:3].T,camera[:3,3],optimize=False)
    for side,h in binding.items():
        si=h['si'];key=(frame,si);planes[key]=[]
        if source['hand_valid'][frame,si] and source['hand_directly_observed'][frame,si] and np.isfinite(source['hand_joints_world'][frame,si,[0,4,8,12]]).all():
            observed=np.einsum('ni,ij->nj',source['hand_joints_world'][frame,si,[0,4,8,12]]-camera[:3,3],camera[:3,:3],optimize=False)
            projected=np.einsum('ni,ji->nj',data.xpos[h['points']],A,optimize=False)-offset
            assert (observed[:,2]>.01).all() and (projected[:,2]>.01).all()
            uv=np.c_[K[0,0]*observed[:,0]/observed[:,2]+K[0,2],K[1,1]*observed[:,1]/observed[:,2]+K[1,2]]
            prediction=np.c_[K[0,0]*projected[:,0]/projected[:,2]+K[0,2],K[1,1]*projected[:,1]/projected[:,2]+K[1,2]]
            errors=np.linalg.norm(prediction-uv,axis=1);original_errors[si].extend(errors)
            specs=[]
            for pid,p,obs,err,ev in zip(h['points'],projected,uv,errors,prediction-uv,strict=True):
                M=np.array([[K[0,0],0,K[0,2]-obs[0]],[0,K[1,1],K[1,2]-obs[1]]])
                angle=np.arctan2(ev[1],ev[0]);theta=angle+np.pi/16+np.arange(16)*2*np.pi/16
                normals=np.c_[np.cos(theta),np.sin(theta)]
                radius=max(0.,err-min(.001,.01*err))
                cone=mm(normals,M);cone[:,2]-=radius*np.cos(np.pi/16)
                specs.append(dict(body=pid,observed_uv=obs,baseline_error=err,cone=cone,normalization=max(err,.05)*p[2]))
            source_specs[key]=dict(A=A,offset=offset,points=specs)

def add_contact_planes(frame):
    added=0
    for c in data.contact[:data.ncon]:
        a,b=int(c.geom1),int(c.geom2)
        for side,h in binding.items():
            if a in h['hand_geoms'] and b in target_geoms:hand,target,normal=a,b,-np.array(c.frame[:3])
            elif b in h['hand_geoms'] and a in target_geoms:hand,target,normal=b,a,np.array(c.frame[:3])
            else:continue
            entries=planes[(frame,h['si'])]
            if any(p['hand']==hand and p['target']==target for p in entries):continue
            support=float(np.einsum('ni,i->n',vertices(target),normal,optimize=False).max())
            projected=np.einsum('ni,i->n',vertices(hand),normal,optimize=False)-support
            selected=set(np.flatnonzero(projected<=projected.min()+1e-9).tolist())
            entries.append(dict(hand=hand,target=target,normal=normal,support=support,active_vertex_indices=selected));added+=1
    return added

def measure(x,add_planes=False):
    violations={};errors={0:[],1:[]};maximum_point_regression=0.;commands=q.copy();minimum_plane_gap=.1;added=0
    for frame in range(count):
        deadline();set_frame(frame,x[frame])
        if add_planes:added+=add_contact_planes(frame)
        for side,h in binding.items():
            si=h['si'];key=(frame,si)
            for col,j in enumerate(h['wrist'][:3]):commands[frame,si,col]=data.qpos[int(model.jnt_qposadr[j])]
            adr=int(model.jnt_qposadr[h['wrist'][3]]);commands[frame,si,3:7]=data.qpos[adr:adr+4][[1,2,3,0]]
            commands[frame,si,7:]=[data.qpos[int(model.jnt_qposadr[j])] for j in h['fingers']]
            for p in planes[key]:
                gaps=np.einsum('ni,i->n',vertices(p['hand']),p['normal'],optimize=False)-p['support']
                minimum_plane_gap=min(minimum_plane_gap,float(gaps.min()))
                if add_planes and gaps.min()<.0005-1e-7:p['active_vertex_indices'].update(np.flatnonzero(gaps<=gaps.min()+1e-9).tolist())
            if key in source_specs:
                spec=source_specs[key]
                for p in spec['points']:
                    camera=mm(spec['A'],data.xpos[p['body']])-spec['offset'];assert camera[2]>.01
                    uv=np.array([K[0,0]*camera[0]/camera[2]+K[0,2],K[1,1]*camera[1]/camera[2]+K[1,2]])
                    error=float(np.linalg.norm(uv-p['observed_uv']));errors[si].append(error)
                    maximum_point_regression=max(maximum_point_regression,error-p['baseline_error'])
        for c in data.contact[:data.ncon]:
            a,b=int(c.geom1),int(c.geom2)
            for side,h in binding.items():
                if (a in h['hand_geoms'] and b in target_geoms) or (b in h['hand_geoms'] and a in target_geoms):
                    key=(frame,h['si']);violations[key]=max(violations.get(key,0.),max(0.,-float(c.dist)))
    source_summary=[];source_pass=maximum_point_regression<=1e-6
    for si in range(2):
        before,after=np.asarray(original_errors[si]),np.asarray(errors[si])
        row=dict(hand=['left','right'][si],observed_points=len(after),before_median_px=float(np.median(before)),after_median_px=float(np.median(after)),before_p95_px=float(np.quantile(before,.95)),after_p95_px=float(np.quantile(after,.95)))
        source_pass &= row['after_median_px']<=row['before_median_px'] and row['after_p95_px']<=row['before_p95_px'];source_summary.append(row)
    return dict(hand_slots_over_1mm=sum(v>.001 for v in violations.values()),maximum_native_penetration_m=max(violations.values(),default=0.),minimum_conservative_plane_gap_m=minimum_plane_gap,maximum_source_point_regression_px=maximum_point_regression,source_projection_passed=bool(source_pass),source_summary=source_summary,new_planes=added),commands

x=np.zeros((count,2,13))
control,_=measure(x,add_planes=False);report['control']=control
# Reuse the previously source-qualified articulated hand only as a warm start.
# The optimization variables and ALL correction bounds still refer to the
# aligned ORIGINAL reference. This is a fresh linearization, not extra bounds.
with np.load(P/'targets.npz',allow_pickle=False) as z:retained_commands=z['robot_qpos'].copy()
x[:,:,:3]=(retained_commands[:,:,:3]-q[:,:,:3])/.01
for si in range(2):
    x[:,si,3:6]=(Rotation.from_quat(q[:,si,3:7]).inv()*Rotation.from_quat(retained_commands[:,si,3:7])).as_rotvec()/.1
x[:,:,6:]=(retained_commands[:,:,7:]-q[:,:,7:])/.1
x[:,:,:3]=np.clip(x[:,:,:3],-1.4,1.4)
x[:,:,6:]=np.clip(x[:,:,6:],-2.5,2.5)
rotation_norm=np.linalg.norm(x[:,:,3:6],axis=-1)
x[:,:,3:6]*=np.minimum(1.,1.5/np.maximum(rotation_norm,1e-12))[:,:,None]
warm,_=measure(x,add_planes=True)
report['warm_start']=warm
report['warm_start_method']='Prior source-qualified hand transformed consistently, clipped to unchanged absolute bounds; all gates relative to aligned ORIGINAL reference.'

eye=sparse.eye(n,format='csc')
d1=sparse.kron(sparse.diags([-np.ones(count-1),np.ones(count-1)],[0,1],shape=(count-1,count)),sparse.eye(26),format='csc')
d2=sparse.kron(sparse.diags([np.ones(count-2),-2*np.ones(count-2),np.ones(count-2)],[0,1,2],shape=(count-2,count)),sparse.eye(26),format='csc')
H=sparse.triu(2*(eye+20*d1.T@d1+100*d2.T@d2),format='csc')
lower=np.tile(np.r_[np.full(3,-1.4),np.full(3,-1.5),np.full(7,-2.5)],(count,2,1));upper=-lower
for side,h in binding.items():
    lower[:,h['si'],6:]=np.maximum(lower[:,h['si'],6:],(h['finger_lower']-q[:,h['si'],7:])/.1)
    upper[:,h['si'],6:]=np.minimum(upper[:,h['si'],6:],(h['finger_upper']-q[:,h['si'],7:])/.1)
step=np.tile(np.r_[np.full(3,.15),np.full(3,.2),np.full(7,.5)],2)
base_A=sparse.vstack([eye,d1,d2],format='csc')
base_l=np.r_[lower.ravel(),np.tile(-step,count-1),np.tile(-step,count-2)]
base_u=np.r_[upper.ravel(),np.tile(step,count-1),np.tile(step,count-2)]
# Real finger-joint velocity constraints, including original target changes.
vrows=[];vcols=[];vvalues=[];vlo=[];vhi=[]
for side,h in binding.items():
    limits={j.attrib['name']:j.find('limit').attrib for j in ET.parse(P/f'robot/urdfs/g1/dex3_{side}.urdf').getroot().findall('joint') if j.attrib['type']!='fixed'}
    for frame,dt in enumerate(np.diff(original['timestamps_s'])):
        for j,name in enumerate(manifest['joint_names'][side]):
            delta=q[frame+1,h['si'],7+j]-q[frame,h['si'],7+j];speed=float(limits[name]['velocity'])
            row=len(vlo);col=frame*26+h['si']*13+6+j
            vrows.extend([row,row]);vcols.extend([col,col+26]);vvalues.extend([-.1,.1]);vlo.append(-speed*dt-delta);vhi.append(speed*dt-delta)
V=sparse.csc_matrix((vvalues,(vrows,vcols)),shape=(len(vlo),n));base_A=sparse.vstack([base_A,V],format='csc');base_l=np.r_[base_l,vlo];base_u=np.r_[base_u,vhi]
commands=q.copy()
try:
    for iteration in range(12):
        deadline();rows=[];cols=[];values=[];low=[];high=[];jac_checks=[]
        def add(block,lo,hi,col):
            block=np.atleast_2d(block);index=len(low)
            rr,cc=np.nonzero(abs(block)>1e-14);rows.extend((rr+index).tolist());cols.extend((cc+col).tolist());values.extend(block[rr,cc].tolist())
            low.extend(np.broadcast_to(lo,(len(block),)).tolist());high.extend(np.broadcast_to(hi,(len(block),)).tolist())
        for frame in range(count):
            deadline();set_frame(frame,x[frame])
            for side,h in binding.items():
                si=h['si'];key=(frame,si);col=frame*26+si*13;T=tangent_map(frame,x[frame],si);current=x[frame,si]
                for p in planes[key]:
                    geom=p['hand'];verts=vertices(geom)[sorted(p['active_vertex_indices'])];r=verts-data.geom_xpos[geom];normal=p['normal']
                    jp=np.zeros((3,model.nv));jr=jp.copy();mujoco.mj_jac(model,data,jp,jr,data.geom_xpos[geom],int(model.geom_bodyid[geom]))
                    gradient=mm(np.einsum('i,ij->j',normal,jp,optimize=False)[None,:]+mm(np.cross(r,normal),jr),T)/.01
                    gap=(np.einsum('ni,i->n',verts,normal,optimize=False)-p['support'])/.01
                    add(gradient,.05-gap+mm(gradient,current),np.inf,col)
                if key in source_specs:
                    spec=source_specs[key]
                    for p in spec['points']:
                        jp=np.zeros((3,model.nv));jr=jp.copy();point=data.xpos[p['body']]
                        mujoco.mj_jac(model,data,jp,jr,point,p['body'])
                        camera=mm(spec['A'],point)-spec['offset'];J=mm(mm(spec['A'],jp),T)
                        gradient=mm(p['cone'],J)/p['normalization'];expr=mm(p['cone'],camera)/p['normalization']
                        add(gradient,-np.inf,-expr+mm(gradient,current),col)
                norm=np.linalg.norm(current[3:6]*.1)
                if norm>.15:
                    gradient=np.zeros(13);gradient[3:6]=current[3:6]/np.linalg.norm(current[3:6]);add(gradient,-np.inf,1.5,col)
        A=sparse.csc_matrix((values,(rows,cols)),shape=(len(low),n));A=sparse.vstack([base_A,A],format='csc')
        assert np.isfinite(A.data).all() and np.isfinite(H.data).all()
        assert np.isfinite(np.r_[base_l,low][np.isfinite(np.r_[base_l,low])]).all()
        solver=osqp.OSQP();solver.setup(P=H,q=np.zeros(n),A=A,l=np.r_[base_l,low],u=np.r_[base_u,high],eps_abs=1e-7,eps_rel=1e-7,max_iter=100000,polishing=True,verbose=False,time_limit=min(45.,max(1.,300-(time.monotonic()-started))))
        solution=solver.solve(raise_error=False)
        entry=dict(iteration=iteration,status=solution.info.status,linear_constraints=int(A.shape[0]),solve_seconds=float(solution.info.run_time),finite_linear_model_verified=True,primal_residual=float(solution.info.prim_res),dual_residual=float(solution.info.dual_res),solver_iterations=int(solution.info.iter))
        if solution.info.status!='solved':report['iterations'].append(entry);save();break
        x=solution.x.reshape(count,2,13);measurements,commands=measure(x,add_planes=True)
        entry.update(measurements);report['iterations'].append(entry);save();print(json.dumps({k:v for k,v in entry.items() if k!='source_summary'}),flush=True)
        if measurements['source_projection_passed'] and measurements['hand_slots_over_1mm']==0 and measurements['minimum_conservative_plane_gap_m']>=.0005-1e-7 and np.linalg.norm(x[:,:,3:6]*.1,axis=-1).max()<=.15+1e-8:break
    final,commands=measure(x)
    limits_pass=True;velocity_records=[]
    for side,h in binding.items():
        limits={j.attrib['name']:j.find('limit').attrib for j in ET.parse(P/f'robot/urdfs/g1/dex3_{side}.urdf').getroot().findall('joint') if j.attrib['type']!='fixed'}
        for j,name in enumerate(manifest['joint_names'][side]):
            speed=float(limits[name]['velocity']);actual=float(abs(np.diff(commands[:,h['si'],7+j])/np.diff(original['timestamps_s'])).max())
            passed=bool(actual<=speed+1e-6 and commands[:,h['si'],7+j].min()>=h['finger_lower'][j]-1e-6 and commands[:,h['si'],7+j].max()<=h['finger_upper'][j]+1e-6);limits_pass &= passed
            velocity_records.append(dict(hand=side,joint=name,actual_maximum_rad_s=actual,urdf_limit_rad_s=speed,passed=passed))
    correction=commands[:,:,:3]-q[:,:,:3]
    final.update(finger_joint_limits_and_velocities_passed=bool(limits_pass),maximum_root_correction_norm_m=float(np.linalg.norm(correction,axis=-1).max()),maximum_wrist_correction_rad=float(np.linalg.norm(x[:,:,3:6]*.1,axis=-1).max()),maximum_finger_change_rad=float(abs(commands[:,:,7:]-q[:,:,7:]).max()),maximum_adjacent_root_correction_component_m=float(abs(np.diff(correction,axis=0)).max()),maximum_second_root_correction_component_m=float(abs(np.diff(correction,n=2,axis=0)).max()))
    report['final']=final;report['finger_velocity_checks']=velocity_records
    report['hand_source_contact_motion_stage_passed']=bool(final['source_projection_passed'] and final['hand_slots_over_1mm']==0 and final['minimum_conservative_plane_gap_m']>=.0005-1e-7 and limits_pass and final['maximum_root_correction_norm_m']<=.025 and final['maximum_wrist_correction_rad']<=.15+1e-8 and final['maximum_finger_change_rad']<=.25+1e-8 and final['maximum_adjacent_root_correction_component_m']<=.0015+1e-7 and final['maximum_second_root_correction_component_m']<=.0015+1e-7)
    np.savez_compressed(OUT/'inferred-robot-command-candidate.npz',robot_qpos=commands,normalized_corrections=x,frame_indices=original['frame_indices'],timestamps_s=original['timestamps_s'],source_confidence_valid=original['confidence_valid'])
    report['candidate_sha256']=sha(OUT/'inferred-robot-command-candidate.npz')
    for name,h in manifest['files'].items():assert sha(P/name)==h
    report['frozen_source_closure_verified_after']=True
except BaseException as error:
    report['error']=repr(error);raise
finally:
    report['wall_seconds']=time.monotonic()-started;save()

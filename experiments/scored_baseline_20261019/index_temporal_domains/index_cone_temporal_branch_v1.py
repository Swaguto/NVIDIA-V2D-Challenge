"""NEW continuous index/source/temporal branch method at the frozen wrist mode.

The angle domain is not restricted to old finite neighbors or inner boxes.
Camera-cone tangents and globally bounded sinusoid Taylor remainders give
necessary linear relaxations. An inscribed polygon with the opposite remainder
sign gives sufficient constructive LPs. Numerical LP results are not formal
directed-rounding exclusion certificates or physical success.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-index-source-temporal-cone-branch-v1'
sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
import thumb_conic_geometry_v1 as g
from planar_support_critical_points_v1 import point
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
from scipy.spatial.transform import Rotation
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
j=(O/'all-continuous-domain-proposals-LP-returns-and-source-outcomes.jsonl').open('w')

def taylor_plane(H,n,box):
    box=np.array(box);mid=box.mean(axis=1);half=np.diff(box,axis=1)[:,0]/2
    a,b=mid;P,Q=np.asarray(H)[:,:2].T@np.asarray(n)
    f=float(n@H[:,2]+.0458*(P*np.cos(a)+Q*np.sin(a))+.035*(P*np.cos(a+b)+Q*np.sin(a+b)))
    v=.035*(-P*np.sin(a+b)+Q*np.cos(a+b))
    derivative=np.array([.0458*(-P*np.sin(a)+Q*np.cos(a))+v,v])
    remainder=.5*np.hypot(P,Q)*(.0458*half[0]**2+.035*sum(half)**2)+1e-12
    return dict(center=mid,center_value=f,derivative=derivative,remainder=float(remainder),
                affine_constant=float(f-derivative@mid),normal=n,box=box)

controls=[];rng=np.random.default_rng(1919109)
for k in range(3):
    H=rng.normal(size=(3,3));n=rng.normal(size=3);n/=np.linalg.norm(n)
    box=np.array([[.1,.4],[.65,1.03]])+k*.03;record=taylor_plane(H,n,box)
    samples=rng.uniform(box[:,0],box[:,1],size=(256,2));residual=[]
    for angles in samples:
        exact=float(n@(H@np.r_[point(*angles),1.]));linear=record['affine_constant']+record['derivative']@angles
        residual.append(float(exact-linear))
    passed=bool(max(abs(np.array(residual)))<=record['remainder'])
    g.journal(j,dict(kind='new_sinusoid_Taylor_necessary_sufficient_control',record=record,sample_angles=samples,residuals=residual,passed=passed))
    controls.append(dict(name='independent_sinusoid_remainder_'+str(k),passed=passed));assert passed
N=16;theta=2*np.pi*np.arange(N)/N;duals=np.c_[np.cos(theta),np.sin(theta)]
# Vertices of the inscribed polygon have exactly the original disk radius.
vertices=np.c_[np.cos(theta+np.pi/N),np.sin(theta+np.pi/N)]
polygon_bound=np.cos(np.pi/N)
passed=bool(np.max(vertices@duals.T)<=polygon_bound+1e-14 and np.max(np.linalg.norm(vertices,axis=1))<=1.+1e-14)
controls.append(dict(name='inscribed_source_polygon_contains_no_outside_disk_vertices',passed=passed));g.journal(j,dict(kind='new_source_inner_polygon_control',duals=duals,vertices=vertices,apothem=polygon_bound,passed=passed));assert passed

with np.load(S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz') as z:
    commands=z['robot_qpos'].copy();weights=z['weights'].copy()
active=np.flatnonzero(weights>1e-14);assert np.array_equal(active,np.arange(103,122))
T=len(c.q);D=2;Qn=T*D
original=c.aligned[:,1,7:9]
source_rows={v['frame']:v for v in json.loads((S/'e037-thumb-ray-independent-validation-v1/report.json').read_text())['source_rows'] if v['hand']=='right'}
domains={}
initial=np.stack([np.maximum(c.limits[:2,0]-1e-8,original-.25-1e-7),np.minimum(c.limits[:2,1]+1e-8,original+.25+1e-7)],axis=-1)
for frame in range(T):
    if frame not in active:initial[frame]=np.c_[commands[frame,1,7:9],commands[frame,1,7:9]]
    else:
        camera=c.cameras[frame];pixel=c.uv((c.human[frame,1,[8]]-camera[:3,3])@camera[:3,:3])[0]
        A=camera[:3,:3].T@c.stage.T;R=Rotation.from_quat(commands[frame,1,3:7]).as_matrix()
        H=np.c_[A@R@np.eye(3)[:,:2],A@(commands[frame,1,:3]+R@np.array([.0777,-.0016,.0285]))-camera[:3,:3].T@camera[:3,3]]
        B=np.array([[c.K[0,0],0,c.K[0,2]-pixel[0]],[0,c.K[1,1],c.K[1,2]-pixel[1]]])
        domains[int(frame)]=dict(H=H,B=B,pixel=pixel,radius=source_rows[frame]['source_before_after_px'][0][2],duals=[v.copy() for v in duals])

# The full clock is kept explicitly. Outside the declared root mode all index
# commands remain exact; source missingness, other fingers and roots are fixed.
ii=[];jj=[];vv=[];rhs=[]
def base_row(entries,limit):
    k=len(rhs)
    for index,v in entries:ii.append(k);jj.append(index);vv.append(v)
    rhs.append(limit)
for t in range(T):
    for d in range(2):
        p=t*2+d;base_row([(p,1),(Qn+p,-1)],original[t,d]);base_row([(p,-1),(Qn+p,-1)],-original[t,d])
for t in range(1,T):
    for d in range(2):
        e=[(t*2+d,1),((t-1)*2+d,-1)];delta=original[t,d]-original[t-1,d]
        base_row(e,delta+.05+1e-7);base_row([(i,-v) for i,v in e],-delta+.05+1e-7)
        base_row(e,12*(c.ts[t]-c.ts[t-1])+1e-7);base_row([(i,-v) for i,v in e],12*(c.ts[t]-c.ts[t-1])+1e-7)
for t in range(2,T):
    for d in range(2):
        e=[(t*2+d,1),((t-1)*2+d,-2),((t-2)*2+d,1)];delta=original[t,d]-2*original[t-1,d]+original[t-2,d]
        base_row(e,delta+.05+1e-7);base_row([(i,-v) for i,v in e],-delta+.05+1e-7)
base=coo_matrix((vv,(ii,jj)),shape=(len(rhs),2*Qn)).tocsr();rhs=np.array(rhs)
objective=np.r_[np.zeros(Qn),np.ones(Qn)]

def source_relaxation(box,inner):
    ii=[];jj=[];vv=[];rhs=[];cert=[]
    for frame,domain in domains.items():
        H=domain['H'];B=domain['B'];r=domain['radius']+1e-6
        directions=duals if inner else domain['duals']
        for u in directions:
            camera_plane=B.T@u-r*(polygon_bound if inner else 1.)*np.array([0.,0.,1.])
            camera_plane/=np.linalg.norm(camera_plane)
            record=taylor_plane(H,camera_plane,box[frame]);record['frame']=frame;record['kind']='inner' if inner else 'outer'
            cert.append(record);k=len(rhs)
            for d in range(2):ii.append(k);jj.append(frame*2+d);vv.append(record['derivative'][d])
            rhs.append((-record['remainder'] if inner else record['remainder'])-record['affine_constant'])
        record=taylor_plane(H,-np.array([0.,0.,1.]),box[frame]);record['frame']=frame;record['kind']='positive_depth';cert.append(record);k=len(rhs)
        for d in range(2):ii.append(k);jj.append(frame*2+d);vv.append(record['derivative'][d])
        rhs.append(((-record['remainder'] if inner else record['remainder'])-record['affine_constant'])-(.0100000001 if inner else .01))
    return coo_matrix((vv,(ii,jj)),shape=(len(rhs),2*Qn)).tocsr(),np.array(rhs),cert

def evaluate(angles):
    d=angles-original;motion=dict(component_rad=float(abs(d).max()),adjacent_added_rad=float(abs(np.diff(d,axis=0)).max()),
        second_added_rad=float(abs(np.diff(d,n=2,axis=0)).max()),maximum_velocity_rad_s=float((abs(np.diff(angles,axis=0))/np.diff(c.ts)[:,None]).max()))
    source=[]
    for frame,domain in domains.items():
        C=domain['H']@np.r_[point(*angles[frame]),1.];error=float(np.linalg.norm(c.uv(C[None])[0]-domain['pixel'])) if C[2]>.01 else None
        source.append(dict(frame=frame,angles=angles[frame],camera_point=C,error_px=error,original_cap_px=domain['radius'],
            passed=bool(error is not None and error<=domain['radius']+1e-6)))
    passed=bool(all(v['passed'] for v in source) and motion['component_rad']<=.25+1e-7 and motion['adjacent_added_rad']<=.05+1e-7 and motion['second_added_rad']<=.05+1e-7 and motion['maximum_velocity_rad_s']<=12+1e-7 and ((angles>=c.limits[:2,0]-1e-8)&(angles<=c.limits[:2,1]+1e-8)).all())
    return dict(source=source,motion=motion,passed_analytic_source_original_motion=passed)

queue=[dict(id=0,parent=None,box=initial,depth=0)];processed=[];nextid=1;proposal=None;lp_count=0;cut_count=0
while queue and len(processed)<255 and time.monotonic()-start<120:
    node=queue.pop();box=node['box'];g.journal(j,dict(kind='BEFORE_NEW_continuous_source_temporal_domain',node=node))
    outcome=None;last=None
    for inner in [False,True]:
        A,b,certificate=source_relaxation(box,inner);bounds=list(zip(box[:,:,0].ravel(),box[:,:,1].ravel(),strict=True))+[(0,None)]*Qn
        g.journal(j,dict(kind='BEFORE_NEW_source_temporal_LP',node_id=node['id'],inner=inner,all_source_Taylor_witnesses=certificate))
        result=linprog(objective,A_ub=vstack([base,A]),b_ub=np.r_[rhs,b],bounds=bounds,method='highs',options=dict(time_limit=10.))
        returned=dict(success=bool(result.success),status=int(result.status),message=result.message,x=None if result.x is None else result.x,
            objective=None if result.fun is None else float(result.fun),iterations=int(result.nit))
        g.journal(j,dict(kind='AFTER_NEW_source_temporal_LP_RETURN',node_id=node['id'],inner=inner,returned=returned));lp_count+=1
        if not result.success:
            if not inner and result.status==2:outcome='necessary_linear_outer_relaxation_empty';break
            if not inner and result.status!=2:outcome='unresolved_outer_LP_return';break
            continue
        angles=result.x[:Qn].reshape(T,2);test=evaluate(angles);g.journal(j,dict(kind='AFTER_NEW_source_temporal_analytic_candidate',node_id=node['id'],inner=inner,angles=angles,assessment=test))
        last=(angles,test)
        if test['passed_analytic_source_original_motion']:
            proposal=angles;outcome='constructive_full_clock_index_source_motion_point';break
        if not inner:
            # A violated cone supplies a globally valid new dual direction.
            # This is not a source observation, a root prior or a gate change.
            for row in test['source']:
                if row['passed']:continue
                domain=domains[row['frame']];residual=domain['B']@np.array(row['camera_point'])
                if np.linalg.norm(residual)==0:continue
                u=residual/np.linalg.norm(residual)
                if not any(np.linalg.norm(u-v)<1e-8 for v in domain['duals']):
                    domain['duals'].append(u);cut_count+=1;g.journal(j,dict(kind='new_valid_global_source_cone_dual',node_id=node['id'],frame=row['frame'],unit_dual=u))
    if outcome is None:
        if last is None:outcome='unresolved_no_returned_candidate'
        else:
            angles,test=last;fail=[v for v in test['source'] if not v['passed']]
            # Source violation times interval width identifies the currently
            # unresolved kinematic coordinate. No physical metric is used.
            frame=max(fail,key=lambda v:(v['error_px']-v['original_cap_px']) if v['error_px'] is not None else 1e9)['frame'] if fail else int(active[np.argmax(np.diff(box[active],axis=2).max(axis=(1,2)))])
            widths=box[frame,:,1]-box[frame,:,0];dimension=int(np.argmax(widths));middle=float(box[frame,dimension].mean())
            if widths[dimension]<1e-8:outcome='unresolved_minimum_split_width'
            else:
                left=box.copy();right=box.copy();left[frame,dimension,1]=middle;right[frame,dimension,0]=middle
                children=[dict(id=nextid,parent=node['id'],box=left,depth=node['depth']+1),dict(id=nextid+1,parent=node['id'],box=right,depth=node['depth']+1)];nextid+=2
                # Explore the child containing the returned source candidate
                # first. Both halves and their shared boundary are retained.
                preferred=0 if angles[frame,dimension]<=middle else 1
                queue.extend([children[1-preferred],children[preferred]])
                outcome='split_full_continuous_joint_domain';g.journal(j,dict(kind='NEW_domain_binary_split',node_id=node['id'],frame=frame,dimension=dimension,midpoint=middle,children=children))
    processed.append(dict(id=node['id'],parent=node['parent'],depth=node['depth'],outcome=outcome))
    g.journal(j,dict(kind='AFTER_NEW_continuous_source_temporal_domain',node_id=node['id'],outcome=outcome))
    if proposal is not None:break
g.journal(j,dict(kind='final_unresolved_domains',domains=queue,processed=processed,budget_exhausted=len(processed)>=255,time_cap_reached=time.monotonic()-start>=120))
if proposal is not None:
    with np.load(S/'e037-saved-middle-source-component-v1/UNQUALIFIED-middle-source-motion-component.npz') as z:combined=z['robot_qpos'].copy()
    combined[:,1,7:9]=proposal
    np.savez_compressed(O/'UNQUALIFIED-index-middle-source-motion-proposal.npz',robot_qpos=combined)
j.close()
(O/'processed-and-unresolved-domains.json').write_text(json.dumps(g.plain(dict(processed=processed,unresolved=queue)),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.continuous-index-source-temporal-cone-branch.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    producer_sha256=sha(__file__),new_analytic_controls=controls,processed_domains=len(processed),declared_maximum_domains=255,
    unresolved_domains=len(queue),linear_programs=lp_count,new_valid_camera_cone_dual_directions=cut_count,
    domain_outcomes={k:sum(v['outcome']==k for v in processed) for k in set(v['outcome'] for v in processed)},
    full_clock_analytic_index_source_motion_proposal_exists=proposal is not None,
    original_wrist_mode_axis_amplitude_profile_and_source_authority_unchanged=True,
    finite_old_neighbors_not_mandatory_or_resampled=True,no_old_graph_source_constructor_or_optimizer_repeated=True,
    numerical_outer_bounds_not_formal_exclusion=True,current_verified_reference_unchanged=True,
    independent_full_clock_URDF_source_motion_checks_pending=proposal is not None,
    new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_task_qualified=False,
    all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

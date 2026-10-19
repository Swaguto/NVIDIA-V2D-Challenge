"""NEW bounded-domain Farkas witnesses for eight saved necessary LP leaves.

The prior primal LPs, Taylor bounds and source jobs do not run. L1 objective
auxiliaries are omitted: for any bounded joint vector they can always be made
large enough. Independent dual residual correction uses FINITE joint boxes.
"""
from pathlib import Path
import datetime,hashlib,json,math,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-index-temporal-cone-dual-certificates-v1'
sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
import thumb_conic_geometry_v1 as g
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,csr_matrix,diags,vstack
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
j=(O/'all-new-dual-proposals-returns-and-independent-certificates.jsonl').open('w')

def dual_certificate(A,b,lower,upper,labels,name):
    A=csr_matrix(A);b=np.array(b);lower=np.array(lower);upper=np.array(upper)
    center=(lower+upper)/2;half=(upper-lower)/2;D=len(lower)
    eye=diags(np.ones(D),format='csr')
    allA=vstack([A,eye,-eye]).tocsr();allb=np.r_[b-A@center,half,half]
    labels=labels+[dict(kind='upper_box',coordinate=i) for i in range(D)]+[dict(kind='lower_box',coordinate=i) for i in range(D)]
    scales=np.maximum(np.sqrt(np.asarray(allA.multiply(allA).sum(axis=1)).ravel()),1e-12)
    normalized=diags(1/scales)@allA;normalized_rhs=allb/scales
    g.journal(j,dict(kind='BEFORE_NEW_dual_certificate',name=name,bounded_box=np.c_[lower,upper],labels=labels,
        normalized_matrix=dict(shape=normalized.shape,indptr=normalized.indptr,indices=normalized.indices,data=normalized.data),normalized_rhs=normalized_rhs))
    result=linprog(normalized_rhs,A_eq=vstack([normalized.T,csr_matrix(np.ones((1,len(normalized_rhs))))]),
        b_eq=np.r_[np.zeros(D),1.],bounds=(0,None),method='highs',options=dict(time_limit=10.))
    returned=dict(success=bool(result.success),status=int(result.status),message=result.message,weights=None if result.x is None else result.x,
        objective=None if result.fun is None else float(result.fun),iterations=int(result.nit))
    g.journal(j,dict(kind='AFTER_NEW_dual_LP_RETURN',name=name,returned=returned))
    verified=False;certificate=None
    if result.x is not None:
        y=result.x;r=np.asarray(normalized.T@y).ravel()
        weighted_rhs=math.fsum(float(a)*float(bb) for a,bb in zip(y,normalized_rhs,strict=True))
        finite_box_residual_penalty=math.fsum(float(abs(a))*float(h) for a,h in zip(r,half,strict=True))
        guard=1e-9;gap=-weighted_rhs-finite_box_residual_penalty-guard
        verified=bool((y>=0).all() and gap>0)
        active=[dict(row=i,label=labels[i],normalized_weight=float(y[i]),original_row_weight=float(y[i]/scales[i])) for i in np.flatnonzero(y>1e-8)]
        certificate=dict(weights_nonnegative=bool((y>=0).all()),dual_column_residual=r,
            maximum_column_residual=float(abs(r).max()),weighted_centered_rhs=weighted_rhs,
            residual_correction_over_entire_finite_box=finite_box_residual_penalty,numerical_guard=guard,
            verified_strict_contradiction_margin=gap,verified_bounded_linear_infeasibility=verified,
            active_rows=active,weights_not_a_physical_force_or_causal_importance_score=True)
    g.journal(j,dict(kind='AFTER_independent_NEW_finite_box_dual_verification',name=name,certificate=certificate,verified=verified))
    return dict(name=name,dual_return=returned,certificate=certificate,verified=verified)

controls=[]
for name,A,b,lo,hi,expected in [
    ('incompatible_scalar_halfspaces',[[1.],[-1.]],[.1,-.2],[-1.],[1.],True),
    ('compatible_scalar_halfspaces',[[1.],[-1.]],[.1,.1],[-1.],[1.],False),
    ('box_itself_supplies_missing_opposite_constraint',[[1.]],[.1],[.2],[.8],True),
]:
    record=dual_certificate(A,b,lo,hi,[dict(kind='analytic_halfspace',row=i) for i in range(len(b))],name)
    passed=record['verified']==expected;controls.append(dict(name=name,passed=passed));assert passed,name

P=S/'e037-index-source-temporal-cone-branch-v1';cover=json.loads((P/'processed-and-unresolved-domains.json').read_text())
selected=[v['id'] for v in cover['processed'] if v['outcome']=='necessary_linear_outer_relaxation_empty'][:8]
assert len(selected)==8
path=P/'all-continuous-domain-proposals-LP-returns-and-source-outcomes.jsonl'
assert sha(path)==json.loads((P/'report.json').read_text())['all_outputs_sha256'][path.name]
nodes={};planes={};prior_returns={}
for line in path.open():
    row=json.loads(line);k=row['kind']
    if k=='BEFORE_NEW_continuous_source_temporal_domain' and row['node']['id'] in selected:nodes[row['node']['id']]=row['node']
    elif k=='BEFORE_NEW_source_temporal_LP' and not row['inner'] and row['node_id'] in selected:planes[row['node_id']]=row['all_source_Taylor_witnesses']
    elif k=='AFTER_NEW_source_temporal_LP_RETURN' and not row['inner'] and row['node_id'] in selected:prior_returns[row['node_id']]=row['returned']
T=len(c.q);D=T*2;original=c.aligned[:,1,7:9]
ii=[];jj=[];vv=[];rhs=[];labels=[]
def row(entries,b,label):
    k=len(rhs)
    for index,v in entries:ii.append(k);jj.append(index);vv.append(v)
    rhs.append(b);labels.append(label)
for t in range(1,T):
    for d in range(2):
        e=[(t*2+d,1),((t-1)*2+d,-1)];delta=original[t,d]-original[t-1,d]
        row(e,delta+.05+1e-7,dict(kind='adjacent_added_upper',frame=t,joint=d))
        row([(i,-v) for i,v in e],-delta+.05+1e-7,dict(kind='adjacent_added_lower',frame=t,joint=d))
        row(e,12*(c.ts[t]-c.ts[t-1])+1e-7,dict(kind='actual_velocity_upper',frame=t,joint=d))
        row([(i,-v) for i,v in e],12*(c.ts[t]-c.ts[t-1])+1e-7,dict(kind='actual_velocity_lower',frame=t,joint=d))
for t in range(2,T):
    for d in range(2):
        e=[(t*2+d,1),((t-1)*2+d,-2),((t-2)*2+d,1)];delta=original[t,d]-2*original[t-1,d]+original[t-2,d]
        row(e,delta+.05+1e-7,dict(kind='second_added_upper',frame=t,joint=d))
        row([(i,-v) for i,v in e],-delta+.05+1e-7,dict(kind='second_added_lower',frame=t,joint=d))
base=coo_matrix((vv,(ii,jj)),shape=(len(rhs),D)).tocsr();base_rhs=np.array(rhs);base_labels=labels.copy()
results=[]
for node_id in selected:
    assert prior_returns[node_id]['status']==2
    source_i=[];source_j=[];source_v=[];source_b=[];source_labels=[]
    for k,p in enumerate(planes[node_id]):
        for d in range(2):source_i.append(k);source_j.append(p['frame']*2+d);source_v.append(p['derivative'][d])
        source_b.append(p['remainder']-p['affine_constant']-(.01 if p['kind']=='positive_depth' else 0.))
        source_labels.append(dict(kind='positive_depth' if p['kind']=='positive_depth' else 'source_cone_Taylor_outer',frame=p['frame'],saved_plane=p))
    source=coo_matrix((source_v,(source_i,source_j)),shape=(len(source_b),D)).tocsr();box=np.array(nodes[node_id]['box'])
    result=dual_certificate(vstack([base,source]),np.r_[base_rhs,source_b],box[:,:,0].ravel(),box[:,:,1].ravel(),base_labels+source_labels,'saved_outer_leaf_'+str(node_id))
    result['node_id']=node_id;results.append(result)
j.close()
(O/'eight-saved-leaf-dual-certificates.json').write_text(json.dumps(g.plain(results),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.saved-index-temporal-cone-dual-certificates.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),
    new_analytic_dual_controls=controls,selected_first_eight_saved_empty_leaf_ids=selected,
    new_dual_programs=len(results),verified_finite_box_dual_contradictions=sum(v['verified'] for v in results),
    minimum_verified_residual_corrected_margin=min((v['certificate']['verified_strict_contradiction_margin'] for v in results if v['verified']),default=None),
    old_primal_LPs_Taylor_bounds_charts_and_source_jobs_not_repeated=True,all_dual_returns_before_assertions_retained=True,
    L1_objective_auxiliaries_eliminated_without_altering_joint_feasibility=True,
    numerical_witnesses_not_formal_directed_rounding_or_global_root_mode_exclusion=True,
    remaining12outer_regions_still_unresolved=True,source_authority_and_current_verified_reference_unchanged=True,
    new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_task_qualified=False,
    all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

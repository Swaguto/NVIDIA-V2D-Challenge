"""New direct SDP vectorization, outer rounding and conic dual-sign contracts."""
from pathlib import Path
from fractions import Fraction as F
from math import comb,sqrt
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-direct-SDP-contract-v1';sys.path[:0]=[str(S/'compute'),str(S/'e037-direct-SDP-runtime-v2/site-packages')]
import numpy as np
from scipy import sparse
import clarabel
import direct_conditional_moment_sdp_v1 as d
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();j=(O/'new-direct-SDP-contract-proposals-and-returns.jsonl').open('w');emit=lambda v:g.journal(j,v);controls=[];start=time.monotonic()
group=dict(id=0,scope=['t'],axis=0,conditioning_index=[],columns=list(range(5)))
def check(name,work):
    emit(dict(kind='BEFORE_NEW_direct_SDP_contract',name=name));v=work();emit(dict(kind='AFTER_NEW_direct_SDP_contract_RETURN',name=name,result=v));controls.append(dict(name=name,passed=bool(v['passed'])));assert v['passed']
def packing():
    A=np.array([[2.,-3.,5.],[-3.,7.,11.],[5.,11.,13.]]);B=np.array([[17.,19.,23.],[19.,29.,31.],[23.,31.,37.]])
    a=d.svec(A);b=d.svec(B);error=abs(np.dot(a,b)-np.trace(A@B));unpacked=np.array(d.smat(a,3),dtype=float)
    return dict(passed=d.layout(3)==[(0,0),(0,1),(1,1),(0,2),(1,2),(2,2)]and error<1e-12 and np.max(abs(unpacked-A))<1e-14,trace_error=error,omitted_sqrt_two_error=abs(sum(A[i,k]*B[i,k]for i,k in d.layout(3))-np.trace(A@B)),svec=a)
check('column_major_svec_and_Frobenius_adjoint_offdiagonal_factor',packing)
def rounding():
    lo,hi=d.root_two_interval();valid=lo*lo<2<hi*hi;counts=0;maxerror=0.
    for kind in ['Hankel','localizer']:
        b=d.packed_outer(group,kind);maps=d.exact_maps(group,kind);n=b['dimension'];guard=F(b['outward_diagonal_guard']);valid&=guard>=c.dec(b['spectral_error_upper'])
        for mask in range(32):
            w=[F((mask>>k)&1)for k in range(5)];packed=[sum(F(z)*v for z,v in zip(row,w))for row in b['packed_float_maps']]
            for p,(i,k)in enumerate(d.layout(n)):
                actual=sum(v*x for v,x in zip(maps[i][k],w));err=c.dec(b['unpacked_entry_error_bounds'][i][k]);z=packed[p]
                lows=[z/lo,z/hi]if i!=k else[z];valid&=max(abs(v-actual)for v in lows)<=err;maxerror=max(maxerror,float(err));counts+=1
    return dict(passed=bool(valid),checked_unit_box_vertices_entries=counts,max_entry_error=maxerror,sqrt_two_interval=[c.rec(lo),c.rec(hi)])
check('exact_root_two_bracket_and_all_unit_box_vertex_packing_error_bounds',rounding)
def linear():
    row=dict(name='new_nonbinary',coefficients=[dict(column=0,value=c.rec(F(1,3))),dict(column=1,value=c.rec(F(-2,7)))],bound=c.rec(F(3,11)));v,b,record=d.outer_linear(row)
    margin=F(b)-c.dec(row['bound'])-sum(abs(F(z)-c.dec(x['value']))for (_,z),x in zip(v,row['coefficients']))
    return dict(passed=margin>=0,outward_margin=c.rec(margin),record=record)
check('nonbinary_linear_coefficients_have_necessary_outward_rhs',linear)
def source_model(weights):
    return dict(columns=[dict(scope=['t'],index=[k])for k in range(5)],equalities=[dict(name='fixed:'+str(k),coefficients=[dict(column=k,value=c.rec(1))],bound=c.rec(w))for k,w in enumerate(weights)],inequalities=[],variable_bounds=[0,1])
def bound_signs():
    model=source_model([F(1,16),F(4,16),F(6,16),F(4,16),F(1,16)]);A,b,p=d.assemble(model,emit);x=np.array([1,4,6,4,1])/16;s=b-A@x
    return dict(passed=np.max(abs(s[:5]))==0 and np.allclose(s[5:10],x)and np.allclose(s[10:15],1-x)and p['nonnegative_rows']==10,zero_eq_rows=p['equalities'],lower_box_slack=s[5:10],upper_box_slack=s[10:15])
check('fixed_equalities_and_both_original_unit_box_cone_signs',bound_signs)
def no_normalization():
    model=source_model([0]*5);A,b,p=d.assemble(model,emit);signed=np.array([1.,-1.,0.,0.,0.]);stored=A[p['PSD_blocks'][0]['row_start']:p['PSD_blocks'][0]['row_end']]@signed
    expected=-np.array(d.svec(c.matrices(c.moments(map(F,signed)))['Hankel']))
    return dict(passed=np.max(abs(stored-expected))<1e-15 and np.isfinite(A@np.zeros(5)).all(),signed_zero_mass_preserved=True,normalization_never_applied=True)
check('canonical_conditioned_maps_preserve_zero_and_signed_mass',no_normalization)
def equality_reject():
    model=source_model([F(1,3),0,0,0,0]);error=None
    try:d.assemble(model,emit)
    except ValueError as e:error=str(e)
    return dict(passed=error is not None,retained_failure=error)
check('nonbinary_equalities_are_rejected_without_silent_rounding',equality_reject)
def dual_signs():
    model=dict(columns=[{}],equalities=[dict(name='x0',coefficients=[dict(column=0,value=c.rec(1))],bound=c.rec(0))],inequalities=[dict(name='x_ge1',coefficients=[dict(column=0,value=c.rec(-1))],bound=c.rec(-1))]);p=dict(equalities=1,original_inequalities=1,PSD_blocks=[])
    result=d.certificate(model,p,[1.,1.],emit);wrong=d.certificate(model,p,[1.,-1.],emit)
    return dict(passed=c.dec(result['strict_gap'])==1 and wrong['classification']=='UNRESOLVED_negative_dual_multipliers',positive_certificate=result,wrong_sign_retained=wrong)
check('exact_conic_dual_equality_inequality_sign_and_finite_box_residual',dual_signs)
def gram():
    raw=[[F(1),F(2)],[F(2),F(1)]];G,B=d.rational_gram(raw,emit)
    identity=all(G[i][k]==sum(B[i][a]*B[k][a]for a in range(2))for i in range(2)for k in range(2));minors=c.principal_minors(G)
    return dict(passed=identity and all(c.dec(v['value'])>=0 for v in minors)and raw==[[F(1),F(2)],[F(2),F(1)]],raw_indefinite_dual_preserved=True,certificate_is_a_separate_rational_Gram_product=True,principal_minors=minors)
check('rational_Gram_certificate_does_not_qualify_or_change_raw_returns',gram)
def solve_tiny(name,weights,expected):
    model=source_model(weights);A,b,p=d.assemble(model,emit);cones=[clarabel.ZeroConeT(p['equalities']),clarabel.NonnegativeConeT(p['nonnegative_rows'])]+[clarabel.PSDTriangleConeT(v['dimension'])for v in p['PSD_blocks']];settings=clarabel.DefaultSettings();settings.verbose=False;settings.time_limit=3.;settings.max_iter=50;settings.max_threads=1
    emit(dict(kind='BEFORE_NEW_tiny_direct_SDP_control',name=name,A=A.toarray(),b=b,packing=p,settings=repr(settings)))
    solver=clarabel.DefaultSolver(sparse.csc_matrix((5,5)),np.zeros(5),A,b,cones,settings);result=d.solver_return(solver.solve());emit(dict(kind='AFTER_NEW_tiny_direct_SDP_solver_RETURN',name=name,result=result));(O/(name+'-RETURN.json')).write_text(json.dumps(g.plain(result),allow_nan=False,indent=2)+'\n')
    cert=d.certificate(model,p,result['z'],emit);emit(dict(kind='AFTER_NEW_tiny_direct_SDP_certificate_RETURN',name=name,certificate=cert))
    passed=result['status']=='Solved'if expected=='feasible'else cert['classification']=='EXACT_conditional_frozen_moment_model_exclusion'
    return dict(passed=passed,status=result['status'],certificate_classification=cert['classification'],strict_gap=cert.get('strict_gap'))
check('direct_PSD_feasible_dyadic_measure_control',lambda:solve_tiny('feasible-control',[F(1,16),F(4,16),F(6,16),F(4,16),F(1,16)],'feasible'))
check('direct_PSD_infeasible_fixed_invalid_moments_and_exact_dual_control',lambda:solve_tiny('infeasible-control',[0,1,0,0,0],'infeasible'))
j.close();out=dict(schema='v2d.direct-SDP-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controls=controls,all_controls_passed=all(v['passed']for v in controls),new_analytical_controls=len(controls),new_tiny_direct_SDPs=2,episode_solves=0,helper_sha256=sha(S/'compute/direct_conditional_moment_sdp_v1.py'),runtime_inventory_sha256=sha(S/'e037-direct-SDP-runtime-v2/runtime-inventory.json'),no_old_contract_audit_or_algorithm_repeated=True,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

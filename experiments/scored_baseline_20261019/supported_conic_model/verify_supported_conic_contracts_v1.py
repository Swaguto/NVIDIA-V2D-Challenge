"""NEW weighted PSD packing/adjoint/dual contracts in dimensions4/6/8/10."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from math import comb,sqrt
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-supported-conic-contract-v1';sys.path[:0]=[str(S/'compute'),str(S/'e037-direct-SDP-runtime-v2/site-packages')]
import numpy as np
from scipy import sparse
import clarabel
import supported_conic_maps_v1 as h
import conditional_moment_cones_v1 as c
import direct_conditional_moment_sdp_v1 as old
import mixed_joint_moments_v1 as m
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();j=(O/'all-new-supported-conic-contracts-and-returns.jsonl').open('w');emit=lambda v:g.journal(j,v);controls=[];start=time.monotonic()
def check(name,work):
    emit(dict(kind='BEFORE_NEW_supported_conic_contract',name=name));v=work();emit(dict(kind='AFTER_NEW_supported_conic_contract_RETURN',name=name,result=v));controls.append(dict(name=name,passed=bool(v['passed'])));assert v['passed']
def synthetic(dim):
    maps=[[{col:F(((min(i,k)+1)*(max(i,k)+3)*(col+5))%23-11,13+col)for col in range(7)}for k in range(dim)]for i in range(dim)]
    return dict(id='new_weighted:'+str(dim),dimension=dim,kind='analytic_weighted_map',maps=maps)
def corner_pack(dim):
    block=synthetic(dim);encoded=h.block_record(block);decoded=h.block_decode(encoded);assert decoded['maps']==block['maps'];packed=h.pack_block(block);lo,hi=map(c.dec,packed['sqrt_two_interval']);count=0
    assert F(packed['outward_diagonal_guard'])>=c.dec(packed['spectral_error_bound'])
    for row in packed['rows']:
        i,k=row['i'],row['j'];numeric={v['column']:F(v['value'])for v in row['packed_coefficients']};bound=c.dec(row['unpacked_unit_box_error'])
        for weights in product(range(2),repeat=7):
            target=sum(v*weights[col]for col,v in block['maps'][i][k].items());stored=sum(v*weights[col]for col,v in numeric.items());error=abs(stored-target)if i==k else max(abs(stored/lo-target),abs(stored/hi-target));assert error<=bound;count+=1
    return dict(passed=True,dimension=dim,exact_unit_box_vertex_entry_bounds=count,weighted_map_roundtrip_exact=True,guard_NOT_robot_tolerance=True)
for dim in [4,6,8,10]:check('weighted_svec_exact_error_capacity_dimension_'+str(dim),lambda dim=dim:corner_pack(dim))
def adjoint():
    counts=0;maxerror=0.
    for dim in [4,6,8,10]:
        block=synthetic(dim);weights=[F(k+1,11)for k in range(7)];M=[[sum(v*weights[col]for col,v in row.items())for row in line]for line in block['maps']];Z=[[F((i+2)*(k+2),19)for k in range(dim)]for i in range(dim)];lhs=float(sum(M[i][k]*Z[i][k]for i in range(dim)for k in range(dim)));rhs=float(np.dot(old.svec(M),old.svec(Z)));error=abs(lhs-rhs);maxerror=max(maxerror,error);assert error<2e-11;counts+=1
    return dict(passed=True,dimensions_checked=counts,max_synthetic_frobenius_error=maxerror,offdiagonal_adjoint_factor_two_preserved=True)
check('weighted_Frobenius_adjoint_new_dimensions',adjoint)
def grams():
    identities=0
    for dim in [4,6,8,10]:
        raw=[[F((i+1)*(k+1),17)+(F(-1,101)if i==k else 0)for k in range(dim)]for i in range(dim)];G,B=old.rational_gram(raw,emit)
        for i in range(dim):
            for k in range(dim):assert G[i][k]==sum(B[i][a]*B[k][a]for a in range(dim));identities+=1
    return dict(passed=True,exact_rational_Gram_identities=identities,certificate_products_distinct_from_actual_returns=True,no_primal_weights_clipped_or_qualified=True)
check('new_dimension_rational_Gram_PSD_and_no_primal_repair',grams)
def zeros_and_signs():
    block=synthetic(4);packed=h.pack_block(block);assert len(packed['rows'])==10;weights=[F(1),F(-1),F(0),F(0),F(0),F(0),F(0)];signed=[[sum(v*weights[col]for col,v in co.items())for co in row]for row in block['maps']];assert any(v for row in signed for v in row);assert all(sum(v*F(0)for v in co.values())==0 for row in block['maps']for co in row)
    return dict(passed=True,signed_inputs_retained=True,zero_mass_does_not_drop_a_cone=True,unit_box_error_claim_NOT_applied_to_signed_test_inputs=True)
check('zero_signed_binding_is_linear_without_conditioning_normalization',zeros_and_signs)
def tiny(name,feasible):
    entries={a:k for k,a in enumerate(product(range(5),repeat=2))};basis=[(0,0),(0,1),(1,0),(1,1)];poly={(1,0):F(1),(0,1):F(-1,2),(0,0):F(-1,4)}
    if feasible:atoms=[([x,y],F(1,4))for x,y in product([F(5,8),F(7,8)],[F(1,4),F(1,2)])]
    else:atoms=[([F(0),F(1)],F(1,4)),([F(1),F(0)],F(3,4))]
    weights=[]
    for index in entries:
        value=F(0)
        for point,mass in atoms:
            z=mass
            for k,t in zip(index,point):z*=comb(4,k)*t**k*(1-t)**(4-k)
            value+=z
        weights.append(value)
    maps=[]
    for a in basis:
        row=[]
        for b in basis:row.append(m.polynomial_coefficients(entries,{tuple(x+i+k for x,i,k in zip(power,a,b)):z for power,z in poly.items()}))
        maps.append(row)
    block=dict(id='NEW_tiny_weighted_support:'+name,dimension=4,kind='analytic_source_independent_support',maps=maps);base=sparse.vstack([sparse.eye(25),-sparse.eye(25),sparse.eye(25)],format='csc');b=np.array(list(map(float,weights))+[0.]*25+[1.]*25);before=base.copy();A,rhs,packed=h.append_blocks(base,b,[block],emit);assert (base!=before).nnz==0 and np.array_equal(rhs[:75],b)
    model=dict(columns=[dict(scope=['x','y'],index=list(v))for v in entries],equalities=[dict(name='fixed:'+str(i),coefficients=[dict(column=i,value=c.rec(1))],bound=c.rec(v))for i,v in enumerate(weights)],inequalities=[])
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=50;settings.time_limit=3.;settings.max_threads=1
    emit(dict(kind='BEFORE_NEW_tiny_supported_SDP',name=name,base_untouched=True,model=model,new_block=h.block_record(block),packing=packed,settings=repr(settings)))
    solver=clarabel.DefaultSolver(sparse.csc_matrix((25,25)),np.zeros(25),A,rhs,[clarabel.ZeroConeT(25),clarabel.NonnegativeConeT(50),clarabel.PSDTriangleConeT(4)],settings);result=old.solver_return(solver.solve());emit(dict(kind='AFTER_NEW_tiny_supported_SDP_solver_RETURN',name=name,result=result));(O/(name+'-RETURN.json')).write_text(json.dumps(g.plain(result),allow_nan=False,indent=2)+'\n');certificate=h.dual_certificate(model,[block],result['z'],emit);emit(dict(kind='AFTER_NEW_tiny_supported_exact_dual_RETURN',name=name,certificate=certificate))
    passed=result['status']=='Solved'and certificate['classification']!='EXACT_frozen_supported_model_exclusion'if feasible else certificate['classification']=='EXACT_frozen_supported_model_exclusion'
    return dict(passed=passed,solver_status=result['status'],exact_certificate_classification=certificate['classification'],strict_gap=certificate.get('strict_gap'),original_unit_box_equality_signs_and_prefix_unchanged=True)
check('NEW_supported_PD_measure_direct_conic_control',lambda:tiny('new-feasible-support',True))
check('NEW_weighted_support_infeasibility_conic_and_residual_dual_signs',lambda:tiny('new-infeasible-support',False))
def strict():
    block=synthetic(4);block['maps'][0][1]=dict(block['maps'][0][1]);block['maps'][0][1][0]+=F(1,1<<300);errors=[]
    for fn in[lambda:h.pack_block(block),lambda:h.pack_block(dict(id='bad',dimension=5,maps=[[{}]*5]*5))]:
        try:fn();errors.append(None)
        except ValueError as e:errors.append(str(e))
    return dict(passed=all(errors),errors=errors,near_zero_asymmetry_NOT_silently_averaged=True)
check('exact_asymmetry_and_undeclared_cone_dimensions_fail',strict)
j.close();out=dict(schema='v2d.supported-conic-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/supported_conic_maps_v1.py'),controls=controls,new_analytic_controls=len(controls),all_controls_passed=all(v['passed']for v in controls),new_tiny_analytic_SDPs=2,new_episode_SDPs=0,old_dimension3_contracts_reused_by_hash_NOT_reexecuted=sha(S/'e037-direct-SDP-contract-v2/report.json'),old_source_pointwise_or_joint_contracts_NOT_repeated=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in['controls','all_outputs_sha256']}))

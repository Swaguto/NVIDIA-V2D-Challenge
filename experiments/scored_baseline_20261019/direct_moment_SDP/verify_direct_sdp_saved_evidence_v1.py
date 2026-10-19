"""Independent saved direct-SDP packing, actual table and Gram dual audit.

No solver call, projection, normalization, source polynomial regeneration,
or old LP/cone iteration. Exact checks concern the saved rational models.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
from math import comb
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-direct-SDP-independent-v1';P=S/'e037-direct-conditional-SDP-v1';sys.path.insert(0,str(S/'compute'))
import numpy as np
from scipy import sparse
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'all-independent-direct-SDP-packing-primal-and-dual-certificates.jsonl').open('w');emit=lambda v:g.journal(j,v)
report=read(P/'report.json')
for name,digest in report['all_outputs_sha256'].items():assert sha(P/name)==digest
packing=read(P/'complete-direct-SDP-packing-and-error-bounds.json');model=read(packing['base_model']);assert sha(packing['base_model'])==packing['base_model_sha256'];A=sparse.load_npz(P/'canonical-A.npz').tocsr();b=np.load(P/'canonical-b.npy');result=read(P/'single-direct-SDP-solver-RETURN.json');cert=read(P/'residual-corrected-rational-conic-certificate.json');n=len(model['columns']);x=list(map(F,result['x']));assert len(x)==n
emit(dict(kind='BEFORE_NEW_independent_saved_direct_SDP_verification',solver_RETURN_sha256=sha(P/'single-direct-SDP-solver-RETURN.json'),packing_sha256=sha(P/'complete-direct-SDP-packing-and-error-bounds.json'),no_solver_or_source_job_reexecuted=True,no_actual_weight_modified=True))
def row_values(index):
    r=A.getrow(index);return {int(col):F(float(v))for col,v in zip(r.indices,r.data)if v}
def check_row(index,expected,rhs):
    assert row_values(index)=={k:F(v)for k,v in expected.items()if v}and F(float(b[index]))==F(rhs)
def value(row):return sum(c.dec(v['value'])*x[v['column']]for v in row['coefficients'])-c.dec(row['bound'])
eqres=[];ineqres=[];linear_packing_coefficients=0
for i,row in enumerate(model['equalities']):
    expected={v['column']:c.dec(v['value'])for v in row['coefficients']};check_row(i,expected,c.dec(row['bound']));res=value(row);eqres.append(res);emit(dict(kind='AFTER_NEW_exact_equality_binding_and_actual_residual',row=i,name=row['name'],residual=c.rec(res)))
offset=len(eqres)
for k,row in enumerate(model['inequalities']):
    rec=packing['linear_rounding'][k];actual={v['column']:F(float(c.dec(v['value'])))for v in row['coefficients']};error=sum(abs(actual[v['column']]-c.dec(v['value']))for v in row['coefficients']);original=c.dec(row['bound']);assert F(rec['packed_bound'])>=original+error and c.dec(rec['coefficient_absolute_error'])==error
    check_row(offset+k,actual,rec['packed_bound']);linear_packing_coefficients+=len(actual);res=value(row);ineqres.append(res);emit(dict(kind='AFTER_NEW_independent_linear_outer_row_and_actual_residual',row=k,packing_error=c.rec(error),actual_original_model_residual=c.rec(res)))
offset+=len(ineqres)
for side,(sign,rhs)in enumerate([(-1,0),(1,1)]):
    for col in range(n):check_row(offset+side*n+col,{col:sign},rhs)
offset+=2*n;assert offset==packing['equalities']+packing['nonnegative_rows']
def moment_map(r,k):return F(comb(k,r),comb(4,r))if k>=r else F(0)
def map_entry(kind,i,k):
    degree=i+k
    return [moment_map(degree,t)if kind=='Hankel'else moment_map(degree+1,t)-moment_map(degree+2,t)for t in range(5)]
def det(M):
    if len(M)==1:return M[0][0]
    return sum((-1)**i*M[0][i]*det([row[:i]+row[i+1:]for row in M[1:]])for i in range(len(M)))
def minors(M):
    return [det([[M[i][k]for k in ids]for i in ids])for d in range(1,len(M)+1)for ids in combinations(range(len(M)),d)]
matrix_bad=0;guarded_matrix_bad=0;minor_count=0;entry_count=0;maps_by_block=[]
for block in packing['PSD_blocks']:
    emit(dict(kind='BEFORE_NEW_independent_direct_PSD_block',group=block['group'],cone=block['cone']))
    dim=block['dimension'];kind=block['cone'];columns=block['group']['columns'];maps=[[map_entry(kind,i,k)for k in range(dim)]for i in range(dim)];maps_by_block.append(maps);assert [[[c.dec(v)for v in row]for row in M]for M in block['exact_entry_maps']]==maps
    lo,hi=map(c.dec,block['sqrt_two_interval']);assert 0<lo<hi and lo*lo<2<hi*hi;errors=[[F(0)for _ in range(dim)]for _ in range(dim)];M=[[sum(v*x[col]for v,col in zip(maps[i][k],columns))for k in range(dim)]for i in range(dim)]
    assert block['row_start']==offset
    for q,(i,k)in enumerate([(i,k)for k in range(dim)for i in range(k+1)]):
        numeric=block['packed_float_maps'][q];err=sum(abs(F(z)-v)if i==k else max(abs(F(z)/lo-v),abs(F(z)/hi-v))for z,v in zip(numeric,maps[i][k]));assert err==c.dec(block['unpacked_entry_error_bounds'][i][k]);errors[i][k]=errors[k][i]=err
        expected={col:-F(z)for col,z in zip(columns,numeric)};rhs=block['outward_diagonal_guard']if i==k else 0;check_row(offset+q,expected,rhs);entry_count+=len(numeric)
    radius=max(map(sum,errors));guard=F(block['outward_diagonal_guard']);assert radius==c.dec(block['spectral_error_upper'])and guard>=radius;offset=block['row_end'];original_minors=minors(M);guarded=[[v+(guard if i==k else 0)for k,v in enumerate(row)]for i,row in enumerate(M)];guarded_minors=minors(guarded);bad=any(v<0 for v in original_minors);gbad=any(v<0 for v in guarded_minors);matrix_bad+=bad;guarded_matrix_bad+=gbad;minor_count+=len(original_minors)
    emit(dict(kind='AFTER_NEW_independent_direct_PSD_block_RETURN',group_id=block['group']['id'],cone=kind,actual_matrix=[[c.rec(v)for v in row]for row in M],original_principal_minors=list(map(c.rec,original_minors)),necessary_guarded_principal_minors=list(map(c.rec,guarded_minors)),exact_original_PSD=not bad,exact_guarded_PSD=not gbad,packing_spectral_error=c.rec(radius)))
assert offset==len(b)==A.shape[0]
emit(dict(kind='BEFORE_NEW_independent_rational_Gram_dual_certificate'))
certificate_valid=False;dual_gap=None
if cert['classification'].startswith(('EXACT_','UNRESOLVED_no_strict')):
    alpha=list(map(c.dec,cert['alpha']));beta=list(map(c.dec,cert['beta']));assert all(v>=0 for v in alpha)and len(alpha)==len(ineqres)and len(beta)==len(eqres);r=[F(0)]*n;rhs=F(0)
    for rows,weights in [(model['equalities'],beta),(model['inequalities'],alpha)]:
        for row,w in zip(rows,weights,strict=True):
            rhs+=w*c.dec(row['bound'])
            for v in row['coefficients']:r[v['column']]+=w*c.dec(v['value'])
    assert len(cert['matrices'])==len(packing['PSD_blocks'])
    for record,block,maps in zip(cert['matrices'],packing['PSD_blocks'],maps_by_block,strict=True):
        B=[[c.dec(v)for v in row]for row in record['Gram_factor']];G=[[c.dec(v)for v in row]for row in record['certificate_matrix']];dim=len(G);assert all(G[i][k]==sum(B[i][a]*B[k][a]for a in range(dim))for i in range(dim)for k in range(dim));psd=minors(G);assert all(v>=0 for v in psd)
        for a,col in enumerate(block['group']['columns']):r[col]-=sum(G[i][k]*maps[i][k][a]for i in range(dim)for k in range(dim))
        emit(dict(kind='AFTER_NEW_independent_dual_Gram_identity_RETURN',group_id=block['group']['id'],cone=block['cone'],exact_principal_minors=list(map(c.rec,psd))))
    minimum=sum(min(F(0),v)for v in r);dual_gap=minimum-rhs;assert r==list(map(c.dec,cert['residual']))and rhs==c.dec(cert['RHS'])and minimum==c.dec(cert['finite_box_minimum'])and dual_gap==c.dec(cert['strict_gap']);certificate_valid=dual_gap>0;assert certificate_valid==(cert['classification']=='EXACT_conditional_frozen_moment_model_exclusion')
    emit(dict(kind='AFTER_NEW_independent_finite_box_dual_RETURN',RHS=c.rec(rhs),minimum=c.rec(minimum),strict_gap=c.rec(dual_gap),certified_exclusion=certificate_valid))
bound_bad=[dict(column=i,actual=c.rec(v))for i,v in enumerate(x)if not 0<=v<=1];exact_eq=max(map(abs,eqres),default=F(0));exact_ineq=max([F(0),*ineqres]);primal_exact=(not bound_bad and exact_eq==0 and exact_ineq==0 and matrix_bad==0)
j.close();out=dict(schema='v2d.direct-SDP-independent-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),all_independent_packing_and_certificate_checks_passed=True,solver_status=result['status'],canonical_rows_checked=A.shape[0],original_equality_rows_checked=len(eqres),original_inequality_rows_checked=len(ineqres),linear_coefficients_checked=linear_packing_coefficients,PSD_blocks_checked=len(packing['PSD_blocks']),PSD_packing_coefficients_checked=entry_count,actual_original_PSD_matrices_failing=matrix_bad,actual_guarded_PSD_matrices_failing=guarded_matrix_bad,original_principal_minors_checked=minor_count,actual_unit_box_failures=bound_bad,exact_equality_residual=c.rec(exact_eq),exact_original_inequality_residual=c.rec(exact_ineq),actual_table_exactly_qualifies_saved_constraints=primal_exact,dual_Gram_matrices_checked=len(cert.get('matrices',[])),independent_strict_gap=c.rec(dual_gap)if dual_gap is not None else None,conditional_model_excluded=certificate_valid,classification='EXACT_conditional_model_exclusion'if certificate_valid else('local_exact_PSD_table_NOT_deterministic_path'if primal_exact else'UNRESOLVED_numerical_direct_SDP_return_NOT_exact_table_or_exclusion'),no_actual_weights_clipped_normalized_or_repaired=True,no_solver_source_or_old_audit_job_reexecuted=True,numerical_diagnostics_NOT_original_candidate_tolerances=True,no_reference_or_robot_command_promoted=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

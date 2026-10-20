"""Independent exact NEW return audit and residual-corrected conic certificate.

Every raw weight/status is retained. No old return/audit/solver is rerun.
"""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from math import sqrt
import datetime,hashlib,json,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-global-conic-return-independent-v1';P=S/'e037-global-conic-model-v1';E=S/'e037-global-conic-evaluation-v1';sys.path.insert(0,str(S/'compute'))
import conditional_moment_cones_v1 as c
import mixed_joint_moments_v1 as m
import thumb_conic_geometry_v1 as g
import global_conic24_v1 as global24
import supported_conic_maps_v1 as generic
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'all-NEW-return-primal-matrices-and-conic-dual-certificates.jsonl').open('w');emit=lambda v:g.journal(j,v)
def decode(row):return {v['column']:c.dec(v['value'])for v in row}
report=read(E/'report.json')
for name,digest in report['all_outputs_sha256'].items():assert sha(E/name)==digest
binding=read(P/'global-model-binding.json');assert sha(P/'global-model-binding.json')==report['model_binding_sha256'];model=read(P/'extended-original-exact-linear-model.json');assert sha(P/'extended-original-exact-linear-model.json')==binding['exact_linear_model_sha256'];raw=read(E/'single-global-conic-solver-RETURN.json');certificate=read(E/'global-exact-Gram-finite-box-certificate.json');x=list(map(F,raw['x']));z=list(map(F,raw['z']));assert len(x)==1295 and len(z)==16870
blocks,_=global24.saved_base_blocks(binding['prefix_image_folder']);assert len(blocks)==1297
path=P/'global24-original-rational-block.json';assert sha(path)==binding['global_block_sha256'];blocks.append(generic.block_decode(read(path)));assert len(blocks)==1298
assert blocks[-1]['row_start']==16570 and blocks[-1]['row_end']==16870
with (O/'declared-ONE-NEW-global-return-audit.json').open('w')as f:
    f.write(json.dumps(dict(schema='v2d.global-conic-return-audit-declaration.v1',raw_RETURN_sha256=sha(E/'single-global-conic-solver-RETURN.json'),raw_vector_length=1295,original_columns=1125,auxiliary_columns=170,original_scalar_equalities=549,original_scalar_inequalities=5990,original_PSDscopes=1297,new_global_PSDscope=1,exact_Gaussian_congruence_NOT_exponential_minor_enumeration=True,new_solver_calls=0,actual_raw_x_z_s_unchanged=True,finite_box_dual_residual_includes_all_1295_variables=True,scope='Frozen coefficient/region2/fixed unqualified wrist/global necessary model ONLY; no trajectory qualification'),indent=2)+'\n');f.flush();os.fsync(f.fileno())
emit(dict(kind='BEFORE_NEW_exact_supported_return_audit',actual_RETURN_sha256=sha(E/'single-global-conic-solver-RETURN.json'),actual_status=raw['status'],no_actual_x_z_s_clipped_normalized_or_repaired=True,NOT_a_robot_trajectory_candidate=True))
bound_bad=[dict(column=i,value=c.rec(v))for i,v in enumerate(x)if not 0<=v<=1];eqres=[];ineqres=[]
for rows,values,name in [(model['equalities'],eqres,'equality'),(model['inequalities'],ineqres,'inequality')]:
    for i,row in enumerate(rows):
        residual=sum(c.dec(v['value'])*x[v['column']]for v in row['coefficients'])-c.dec(row['bound']);values.append(residual);emit(dict(kind='AFTER_NEW_return_exact_scalar_constraint',row=i,constraint_kind=name,residual=c.rec(residual)))
matrix_counts=Counter();PSD_bad=[];primal_summaries=[]
for block in blocks:
    d=block['dimension'];M=[[sum(v*x[col]for col,v in co.items())for co in row]for row in block['maps']];emit(dict(kind='BEFORE_NEW_return_exact_matrix',id=block['id'],kind_of_matrix=block['kind'],actual_matrix=[[c.rec(v)for v in row]for row in M]));cert=m.congruence_certificate(M,lambda v:emit(dict(block_id=block['id'],new_RETURN_only=True,**v)));emit(dict(kind='AFTER_NEW_exact_congruence_constructor_RETURN',id=block['id'],RETURN=cert));group='global'if block['kind'].startswith('global:')else'conditional'if block['kind'].startswith('saved_')else('joint'if block['kind'].startswith('joint:')else'image'if block['kind'].startswith('image:')else'support')
    if cert['classification']=='exact_negative_mixed_square':
        vector=list(map(c.dec,cert['vector']));expectation=sum(a*M[i][k]*b for i,a in enumerate(vector)for k,b in enumerate(vector));emit(dict(kind='BEFORE_NEW_negative_square_independent_identity',id=block['id'],expectation=c.rec(expectation)));assert expectation==c.dec(cert['expectation'])<0;PSD_bad.append(dict(id=block['id'],kind=block['kind'],dimension=d,expectation=c.rec(expectation)));matrix_counts[group+'_negative']+=1
    else:
        B=[[c.dec(v)for v in col]for col in cert['basis_columns']];D=list(map(c.dec,cert['diagonal']));assert all(B[i][i]==1 and all(B[i][k]==0 for k in range(i+1,d))for i in range(d))and all(v>=0 for v in D)
        BM=[[sum(B[i][a]*M[a][b]for a in range(d)if B[i][a])for b in range(d)]for i in range(d)]
        emit(dict(kind='AFTER_NEW_independent_exact_congruence_product_RETURN',id=block['id'],basis_times_matrix=[[c.rec(v)for v in row]for row in BM]))
        for i in range(d):
            for k in range(d):assert sum(BM[i][b]*B[k][b]for b in range(d)if B[k][b])==(D[i]if i==k else 0)
        matrix_counts[group+'_PSD']+=1
    primal_summaries.append(dict(id=block['id'],kind=block['kind'],classification=cert['classification'],rank=cert.get('rank')));emit(dict(kind='AFTER_NEW_return_exact_matrix_certificate',id=block['id'],classification=cert['classification'],certificate=cert))
dual_checked=False;gap=None;Gram_entries=0
if certificate['classification']in['EXACT_frozen_supported_model_exclusion','UNRESOLVED_no_strict_supported_conic_gap']:
    alpha=list(map(c.dec,certificate['alpha']));beta=list(map(c.dec,certificate['beta']));assert beta==z[:549]and alpha==z[549:549+5990]and all(v>=0 for v in alpha);residual=[F(0)]*1295;rhs=F(0)
    for rows,values in [(model['equalities'],beta),(model['inequalities'],alpha)]:
        for row,w in zip(rows,values,strict=True):
            rhs+=w*c.dec(row['bound'])
            for v in row['coefficients']:residual[v['column']]+=w*c.dec(v['value'])
    assert len(certificate['PSD_Gram_certificates'])==1298
    for block,cert in zip(blocks,certificate['PSD_Gram_certificates'],strict=True):
        assert cert['id']==block['id']and cert['dimension']==block['dimension'];d=block['dimension'];B=[[c.dec(v)for v in row]for row in cert['Gram_factor']];G=[[c.dec(v)for v in row]for row in cert['certificate_matrix']];solver_z=raw['z'][block['row_start']:block['row_end']];recovered=[[F(0)for _ in range(d)]for _ in range(d)]
        for val,(i,k)in zip(solver_z,[(i,k)for k in range(d)for i in range(k+1)],strict=True):recovered[i][k]=recovered[k][i]=F(float(val)/(1 if i==k else sqrt(2)))
        assert recovered==[[c.dec(v)for v in row]for row in cert['raw_dual_matrix']]
        for i in range(d):
            for k in range(d):
                assert G[i][k]==sum(B[i][a]*B[k][a]for a in range(d));Gram_entries+=1
                for col,v in block['maps'][i][k].items():residual[col]-=G[i][k]*v
        emit(dict(kind='AFTER_NEW_independent_supported_dual_Gram_identity',id=block['id'],dimension=d,all_exact_Gram_entries_checked=d*d,raw_solver_dual_preserved=True))
    minimum=sum(min(F(0),v)for v in residual);gap=minimum-rhs;assert residual==list(map(c.dec,certificate['exact_residual']))and minimum==c.dec(certificate['finite_unit_box_minimum'])and rhs==c.dec(certificate['RHS'])and gap==c.dec(certificate['strict_gap']);assert(gap>0)==(certificate['classification']=='EXACT_frozen_supported_model_exclusion');dual_checked=True;emit(dict(kind='AFTER_NEW_independent_supported_finite_box_RETURN',RHS=c.rec(rhs),minimum=c.rec(minimum),strict_gap=c.rec(gap),conditional_frozen_model_excluded=gap>0))
else:
    assert certificate['classification']=='UNRESOLVED_negative_scalar_dual'and any(v<0 for v in z[549:549+5990]);emit(dict(kind='AFTER_NEW_dual_negative_scalar_sign_retained',indices=certificate['indices'],no_clipping_or_retry=True))
eq=max(map(abs,eqres));ineq=max([F(0),*ineqres]);exact_primal=not bound_bad and eq==0 and ineq==0 and not PSD_bad;excluded=dual_checked and gap>0;j.close();out=dict(schema='v2d.global-conic-return-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),all_independent_checks_passed=True,actual_solver_status=raw['status'],actual_solver_iterations=raw['iterations'],actual_solver_tolerances_NOT_candidate_gates=True,new_RETURN_scalar_equalities_checked=549,new_RETURN_scalar_inequalities_checked=5990,actual_unit_box_failures=bound_bad,original_unit_box_failures=sum(v['column']<1125 for v in bound_bad),auxiliary_unit_box_failures=sum(v['column']>=1125 for v in bound_bad),all_1295_exact_unit_boxes_checked=True,maximum_unit_box_violation=max([0.]+[max(-float(x[v['column']]),float(x[v['column']])-1)for v in bound_bad]),exact_equality_residual=c.rec(eq),exact_original_inequality_residual=c.rec(ineq),new_RETURN_matrices_checked=1298,actual_negative_PSD_matrices=len(PSD_bad),actual_matrix_counts=dict(matrix_counts),actual_negative_matrices=PSD_bad,exact_nonzero_equalities=sum(v!=0 for v in eqres),exact_original_positive_inequalities=sum(v>0 for v in ineqres),actual_table_exactly_realizes_all_saved_constraints=exact_primal,independent_conic_Gram_certificate_checked=dual_checked,independent_Gram_matrix_count=1298 if dual_checked else 0,independent_Gram_entry_identities=Gram_entries,independent_strict_gap=c.rec(gap)if gap is not None else None,conditional_frozen_coefficient_model_excluded=excluded,classification='EXACT_frozen_global_necessary_coefficient_model_exclusion'if excluded else'UNRESOLVED_global_necessary_RETURN_NOT_trajectory_or_exclusion',old_table_25support_failures_and_all_old_statuses_preserved=True,actual_x_z_s_NOT_clipped_normalized_repaired_or_promoted=True,no_common_measure_robot_path_source_accuracy_or_task_qualification=True,all_auxiliary_symbols_UNOBSERVED=True,positive_unitmass_cube_necessity_ONLY=True,no_old_source_algebra_solver_or_old_return_audit_reexecuted=True,new_solver_calls=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
with (O/'report.json').open('w')as f:f.write(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
print(json.dumps({k:v for k,v in out.items()if k not in['all_outputs_sha256','actual_negative_matrices','independent_strict_gap','actual_unit_box_failures']}))

"""New saved-return audit after strict unit-box failure; no solver/control rerun.

Actual weights are never clipped/renormalized. Four controls and247row checks
are reused by hash from the retained failed verifier, not rerun or reclassified.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product,combinations
from math import comb
import datetime,hashlib,json,sys,time
import numpy as np
S=Path(__file__).resolve().parents[1];O=S/'e037-common-moment-saved-table-audit-v2';sys.path.insert(0,str(S/'compute'))
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'all-actual-bound-failures-and-saved-moment-PSD-outcomes.jsonl').open('w')
def f(v):return F(int(v['numerator_hex'],16),int(v['denominator_hex'],16))
def rec(v):return dict(numerator_hex=hex(v.numerator),denominator_hex=hex(v.denominator),approx=float(v))
def determinant(A):
    if len(A)==1:return A[0][0]
    return sum((-1)**k*A[0][k]*determinant([r[:k]+r[k+1:]for r in A[1:]])for k in range(len(A)))
def matrices(y):return [('Hankel_M2',[[y[i+k]for k in range(3)]for i in range(3)]),('unit_interval_localizer',[[y[i+k+1]-y[i+k+2]for k in range(2)]for i in range(2)])]
def minors(A):return [dict(indices=list(k),determinant=rec(determinant([[A[i][z]for z in k]for i in k])))for n in range(1,len(A)+1)for k in combinations(range(len(A)),n)]
failed=S/'e037-common-moment-PSD-independent-v1';guard=read(failed/'guard.json');assert guard['worker_exit_code']==1;old=list(map(json.loads,(failed/'all-square-repair-checks-and-exact-moment-PSD-certificates.jsonl').open()));controls=[v for v in old if v['kind']=='AFTER_NEW_PSD_analytic_control'];assert len(controls)==4 and all(v['passed']for v in controls);assert old[-1]['kind']=='AFTER_NEW_independent_247_constraint_and_unchanged_prefix_check'and old[-1]['passed']
g.journal(j,dict(kind='REUSED_completed_controls_and_prefix_check_NOT_reexecuted',failed_log_sha256=sha(failed/'worker.log'),completed_records_sha256=sha(failed/'all-square-repair-checks-and-exact-moment-PSD-certificates.jsonl'),four_PSD_controls_passed_before_failure=True,independent247rows_and_unchanged_prefix_passed_before_failure=True,strict_actual_unit_box_failure_retained=True,unsaved_dual_check_NOT_claimed_independently_verified=True))
P=S/'e037-common-moment-square-repair-v1';report=read(P/'report.json')
for n,d in report['all_outputs_sha256'].items():assert sha(P/n)==d
M=read(P/'corrected-common-moment-LP-model.json');solver=read(P/'corrective-phase-one-solver-RETURN.json');x=list(map(F,solver['primal'][:-1]));violations=[dict(column=i,actual_value=rec(v),unit_box_violation=rec(max(F(0),-v,v-1)))for i,v in enumerate(x)if not 0<=v<=1]
g.journal(j,dict(kind='BEFORE_NEW_saved_return_audit',solver_return_sha256=sha(P/'corrective-phase-one-solver-RETURN.json'),actual_weights_unit_box_passed=not violations,all_bound_failures=violations,NO_clipping_renormalization_or_tolerance_changes=True))
eq=[sum(f(v['value'])*x[v['column']]for v in row['coefficients'])-f(row['bound'])for row in M['equalities']];ineq=[sum(f(v['value'])*x[v['column']]for v in row['coefficients'])-f(row['bound'])for row in M['inequalities']];maps={}
for c,col in enumerate(M['columns']):maps.setdefault(tuple(col['scope']),{})[tuple(col['index'])]=c
groups=zero=bad_matrices=bad_groups=bad_uni=square_witnesses=robust_witnesses=0;records=[]
for scope,table in maps.items():
    for axis in range(len(scope)):
        others=[k for k in range(len(scope))if k!=axis]
        for other in product(range(5),repeat=len(others)):
            selected={k[axis]:c for k,c in table.items()if tuple(k[a]for a in others)==other};y=[sum((F(comb(k,n),comb(4,n))if k>=n else F(0))*x[c]for k,c in selected.items())for n in range(5)];record=dict(scope=list(scope),axis=axis,conditioning_index=list(other),moments=[rec(v)for v in y]);g.journal(j,dict(kind='BEFORE_NEW_saved_table_PSD_audit',record=record));group_bad=False;results=[]
            if not y[0]:zero+=1;record['zero_mass_other_moments_all_zero']=not any(y)
            for name,A in matrices(y):
                mm=minors(A);bad=any(f(v['determinant'])<0 for v in mm);result=dict(matrix=name,entries=[[rec(v)for v in row]for row in A],principal_minors=mm,PSD_necessary_principal_minors_passed=not bad)
                if bad:
                    bad_matrices+=1;group_bad=True;values,vectors=np.linalg.eigh(np.array(A,dtype=float));vector=[F(v)for v in vectors[:,0]];g.journal(j,dict(kind='AFTER_NEW_saved_table_eigendecomposition_RETURN',scope=list(scope),axis=axis,conditioning_index=list(other),matrix=name,eigenvalues=values,eigenvector=[rec(v)for v in vector]))
                    polynomial={}
                    for i,a in enumerate(vector):
                        for k,b in enumerate(vector):polynomial[i+k]=polynomial.get(i+k,F(0))+a*b
                    if name=='unit_interval_localizer':
                        p={}
                        for k,v in polynomial.items():p[k+1]=p.get(k+1,F(0))+v;p[k+2]=p.get(k+2,F(0))-v
                        polynomial=p
                    coeff={k:sum(v*F(comb(k,n),comb(4,n))for n,v in polynomial.items()if n<=k)for k in range(5)};expectation=sum(coeff[k]*x[c]for k,c in selected.items());direct=sum(vector[i]*A[i][k]*vector[k]for i in range(len(A))for k in range(len(A)));assert expectation==direct
                    bound_error=sum(abs(coeff[k])*max(F(0),-x[c],x[c]-1)for k,c in selected.items());robust=expectation+bound_error<0
                    result.update(rational_square_vector=[rec(v)for v in vector],polynomial_power=[dict(degree=k,value=rec(v))for k,v in sorted(polynomial.items())],exact_square_expectation=rec(expectation),square_identity='(sum(v_i*t^i))^2'if name=='Hankel_M2'else't*(1-t)*(v0+v1*t)^2',negative_square_witness_verified=expectation<0,upper_effect_of_correcting_ONLY_bound_violating_entries=rec(bound_error),negative_even_under_that_DIAGNOSTIC_bound=robust,actual_entries_NOT_changed=True);square_witnesses+=expectation<0;robust_witnesses+=robust
                results.append(result)
            record['matrices']=results;record['real_measure_necessary_PSD_failed']=group_bad;bad_groups+=group_bad;bad_uni+=group_bad and len(scope)==1;records.append(record);groups+=1;g.journal(j,dict(kind='AFTER_NEW_saved_table_PSD_audit',record=record))
j.close();(O/'actual-corrected-table-bound-and-PSD-diagnosis.json').write_text(json.dumps(g.plain(dict(bound_failures=violations,groups=records,no_source_time_family_exclusion=True,no_actual_weights_mutated=True)),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.saved-corrected-moment-table-audit.v2',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),retained_previous_verifier_failure=True,four_completed_PSD_controls_and_247row_checks_reused_by_hash=True,unsaved_prior_dual_check_NOT_claimed_verified=True,
    actual_solver_weight_unit_box_failures=len(violations),maximum_actual_unit_box_violation=rec(max((f(v['unit_box_violation'])for v in violations),default=F(0))),original_exact_bound_gate_NOT_relaxed=True,no_actual_weights_clipped_or_normalized=True,
    exact_binary_primal_equality_residual=rec(max(map(abs,eq))),exact_binary_primal_inequality_residual=rec(max(ineq)),moment_groups_checked=groups,zero_conditioning_mass_groups_retained=zero,groups_failing_real_measure_PSD=bad_groups,univariate_groups_failing_real_measure_PSD=bad_uni,matrices_with_negative_exact_principal_minors=bad_matrices,exact_negative_polynomial_square_witnesses=square_witnesses,square_witnesses_robust_to_bound_violating_entries_alone=robust_witnesses,
    corrected_saved_table_NOT_a_real_common_measure_or_pose=True,no_source_time_family_exclusion_or_reference_qualified=True,no_LP_native_or_old_controls_reexecuted=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,
    all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

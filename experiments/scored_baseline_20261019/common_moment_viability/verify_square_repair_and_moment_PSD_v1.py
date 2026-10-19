"""Independent changed-row/dual checks and continuous positivity diagnostics.

No LP rerun. Hankel and t(1-t) localizer PSD are necessary for a measure on
[0,1]. Exact principal minors diagnose failures in saved moment tables.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product,combinations
from math import factorial,comb
import datetime,hashlib,json,sys,time
import numpy as np
S=Path(__file__).resolve().parents[1];O=S/'e037-common-moment-PSD-independent-v1';sys.path.insert(0,str(S/'compute'))
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'all-square-repair-checks-and-exact-moment-PSD-certificates.jsonl').open('w')
def f(v):return F(int(v['numerator_hex'],16),int(v['denominator_hex'],16))
def rec(v):return dict(numerator_hex=hex(v.numerator),denominator_hex=hex(v.denominator),approx=float(v))
def determinant(A):
    if len(A)==1:return A[0][0]
    return sum((-1)**k*A[0][k]*determinant([r[:k]+r[k+1:]for r in A[1:]])for k in range(len(A)))
def matrices(y):return [('Hankel_M2',[[y[i+k]for k in range(3)]for i in range(3)]),('unit_interval_localizer',[[y[i+k+1]-y[i+k+2]for k in range(2)]for i in range(2)])]
def minors(A):return [dict(indices=list(k),determinant=rec(determinant([[A[i][z]for z in k]for i in k])))for n in range(1,len(A)+1)for k in combinations(range(len(A)),n)]
controls=[]
for name,y,expected in[('deterministic_inside',[F(1),F(1,3),F(1,9),F(1,27),F(1,81)],True),('two_point_measure',[(F(1,5)**k+F(4,5)**k)/2 for k in range(5)],True),('negative_variance_pseudo_weight',[F(1),F(1,2),F(1,6),F(0),F(0)],False),('outside_unit_interval_localizer',[F(2)**k for k in range(5)],False)]:
    g.journal(j,dict(kind='BEFORE_NEW_PSD_analytic_control',name=name,moments=[rec(v)for v in y]));checks={n:minors(A)for n,A in matrices(y)};actual=all(f(v['determinant'])>=0 for rows in checks.values()for v in rows);passed=actual==expected;g.journal(j,dict(kind='AFTER_NEW_PSD_analytic_control',name=name,passed=passed,principal_minors=checks));controls.append(dict(name=name,passed=passed));assert passed
P=S/'e037-common-moment-square-repair-v1';report=read(P/'report.json')
for n,d in report['all_outputs_sha256'].items():assert sha(P/n)==d
base=read(S/'e037-common-moment-model-v1/common-moment-LP-model.json');model=read(P/'corrected-common-moment-LP-model.json');solver=read(P/'corrective-phase-one-solver-RETURN.json');certificate=read(P/'corrective-finite-box-dual.json');cuts_path=S/'e037-common-moment-independent-v1/saved-table-negative-square-separators.json';cuts=read(cuts_path)['cuts']
assert model['columns']==base['columns']and model['equalities']==base['equalities']and model['inequalities'][:len(base['inequalities'])]==base['inequalities'];assert len(cuts)==247 and len(model['inequalities'])-len(base['inequalities'])==247
for row,cut in zip(model['inequalities'][len(base['inequalities']):],cuts,strict=True):
    co={v['column']:f(v['value'])for v in cut['coefficients']};scale=max(map(abs,co.values()));assert {v['column']:f(v['value'])for v in row['coefficients']}=={k:v/scale for k,v in co.items()}and f(row['bound'])==0 and row['meta']['certificate_sha256']==sha(cuts_path)
g.journal(j,dict(kind='AFTER_NEW_independent_247_constraint_and_unchanged_prefix_check',passed=True))
alpha=list(map(f,certificate['alpha']));beta=list(map(f,certificate['beta']));r=[F(0)]*len(model['columns']);rhs=F(0);assert all(v>=0 for v in alpha)
for rows,multipliers in[(model['inequalities'],alpha),(model['equalities'],beta)]:
    for row,m in zip(rows,multipliers,strict=True):
        rhs+=m*f(row['bound'])
        for v in row['coefficients']:r[v['column']]+=m*f(v['value'])
gap=sum(min(F(0),v)for v in r)-rhs;assert r==list(map(f,certificate['residual']))and gap==f(certificate['strict_gap'])and gap==0
x=list(map(F,solver['primal'][:-1]));assert all(0<=v<=1 for v in x);eq=[sum(f(v['value'])*x[v['column']]for v in row['coefficients'])-f(row['bound'])for row in model['equalities']];ineq=[sum(f(v['value'])*x[v['column']]for v in row['coefficients'])-f(row['bound'])for row in model['inequalities']];maps={}
for c,col in enumerate(model['columns']):maps.setdefault(tuple(col['scope']),{})[tuple(col['index'])]=c
groups=0;zero=0;bad_matrices=0;bad_groups=0;bad_univariate=0;square_witnesses=0;records=[]
for scope,table in maps.items():
    for axis in range(len(scope)):
        others=[k for k in range(len(scope))if k!=axis]
        for other in product(range(5),repeat=len(others)):
            selected={k[axis]:c for k,c in table.items()if tuple(k[a]for a in others)==other};y=[sum((F(comb(k,n),comb(4,n))if k>=n else F(0))*x[c]for k,c in selected.items())for n in range(5)];record=dict(scope=list(scope),axis=axis,conditioning_index=list(other),moments=[rec(v)for v in y]);g.journal(j,dict(kind='BEFORE_NEW_saved_corrected_table_PSD_check',record=record));group_bad=False;results=[]
            if not y[0]:assert not any(y);zero+=1
            for name,A in matrices(y):
                mm=minors(A);bad=any(f(v['determinant'])<0 for v in mm);result=dict(matrix=name,entries=[[rec(v)for v in row]for row in A],principal_minors=mm,PSD_necessary_principal_minors_passed=not bad)
                if bad:
                    bad_matrices+=1;group_bad=True;values,vectors=np.linalg.eigh(np.array(A,dtype=float));vector=[F(v)for v in vectors[:,0]];g.journal(j,dict(kind='AFTER_NEW_saved_table_eigendecomposition_RETURN',scope=list(scope),axis=axis,conditioning_index=list(other),matrix=name,eigenvalues=values,eigenvector=[rec(v)for v in vector]));expectation=sum(vector[i]*A[i][k]*vector[k]for i in range(len(A))for k in range(len(A)));result['rational_square_vector']=[rec(v)for v in vector];result['exact_square_expectation']=rec(expectation)
                    result['square_identity']='(sum(v_i*t^i))^2'if name=='Hankel_M2'else't*(1-t)*(v0+v1*t)^2';result['negative_square_witness_verified']=expectation<0;square_witnesses+=expectation<0
                results.append(result)
            record['matrices']=results;record['real_measure_necessary_PSD_failed']=group_bad;bad_groups+=group_bad;bad_univariate+=group_bad and len(scope)==1;records.append(record);groups+=1;g.journal(j,dict(kind='AFTER_NEW_saved_corrected_table_PSD_check',record=record))
j.close();(O/'corrected-table-PSD-diagnosis.json').write_text(json.dumps(g.plain(dict(groups=records,PSD_necessity_only_NOT_source_time_family_exclusion=True,no_next_LP_or_eigenvector_pose_promoted=True)),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.common-moment-square-repair-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),new_PSD_analytic_controls=controls,all_new_controls_passed=True,all_247_new_rows_and_original_model_prefix_verified=True,dual_gap=rec(gap),exact_binary_primal_equality_residual=rec(max(map(abs,eq))),exact_binary_primal_inequality_residual=rec(max(ineq)),moment_groups_checked=groups,zero_conditioning_mass_groups_retained=zero,groups_failing_real_measure_PSD=bad_groups,univariate_groups_failing_real_measure_PSD=bad_univariate,matrices_with_negative_exact_principal_minors=bad_matrices,exact_negative_polynomial_square_witnesses=square_witnesses,
    corrected_relaxation_numerically_feasible_NOT_a_common_measure_or_pose=True,no_source_time_family_exclusion_or_reference_qualified=True,no_model_root_source_gate_parameter_or_physical_mutation=True,LPs_reexecuted=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,
    all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

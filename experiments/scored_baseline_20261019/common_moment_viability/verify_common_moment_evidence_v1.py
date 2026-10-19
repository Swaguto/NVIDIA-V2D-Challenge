"""Independent common-marginal semantics, grid and saved-return certificates.

No LP/contractor/source producer reruns. Negative square expectations separate
the saved table from any real measure; they do not exclude other solutions.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from math import comb,prod
import ast,datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-common-moment-independent-v1';sys.path.insert(0,str(S/'compute'))
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'all-independent-common-moment-semantics-and-square-certificates.jsonl').open('w')
def f(r):return F(int(r['numerator_hex'],16),int(r['denominator_hex'],16))
def rec(v):return dict(numerator_hex=hex(v.numerator),denominator_hex=hex(v.denominator),approx=float(v))
def grid(rows):return {tuple(v['index']):f(v['value'])for v in rows}
def normalized(co):
    co={k:v for k,v in co.items()if v};scale=max(map(abs,co.values()),default=F(1));return {k:v/scale for k,v in co.items()}
def curve_weights(n,k,a,b):
    return [sum(F(comb(n-k,p)*comb(k,i-p))*a**p*(1-a)**(n-k-p)*b**(i-p)*(1-b)**(k-i+p)for p in range(n-k+1)if 0<=i-p<=k)for i in range(n+1)]
def blossom(old,degree,ranges):
    tables=[{k:curve_weights(n,k,a,b)for k in range(n+1)}for n,(a,b)in zip(degree,ranges,strict=True)]
    return {k:sum(v*prod(tables[a][k[a]][e[a]]for a in range(len(degree)))for e,v in old.items())for k in product(*(range(n+1)for n in degree))}
def elevate_weights(n,j):
    return {i:F(comb(n,i)*comb(4-n,j-i),comb(4,j))for i in range(n+1)if 0<=j-i<=4-n}
def positive_power(name):
    if name.startswith('square_'):
        a=F(name.removeprefix('square_'));return {0:a*a,1:-2*a,2:F(1)}
    a,b=map(F,name.removeprefix('double_square_').split('_'))
    return {0:a*a*b*b,1:-2*a*b*(a+b),2:a*a+4*a*b+b*b,3:-2*(a+b),4:F(1)}
P=S/'e037-common-moment-model-v1';rp=P/'report.json';report=read(rp)
for n,d in report['all_outputs_sha256'].items():assert sha(P/n)==d
M=read(P/'common-moment-LP-model.json');representations=read(P/'common-degree-factor-representations.json');solver=read(P/'phase-one-solver-RETURN.json');certificate=read(P/'finite-box-dual-certificate.json');last={}
oldjournal=S/'e037-correlated-source-time-clipping-v1/all-correlated-domain-proposals-coefficients-clips-and-outcomes.jsonl';assert sha(oldjournal)==M['saved_clipping_journal_sha256']
for row in map(json.loads,oldjournal.open()):
    if row['kind']=='AFTER_NEW_correlated_factor_contraction':last[row['factor']]=row
grids={};coefficient_checks=0
for row in representations['factors']:
    state=last[row['id']];bounds=[[f(v)for v in r]for r in state['grid_bounds']];target=[[f(v)for v in r]for r in row['target_bounds']];ranges=[((a-c)/(d-c),(b-c)/(d-c))for(a,b),(c,d)in zip(target,bounds,strict=True)];pre=grid(row['pre_elevation_grid']);exact=blossom(grid(state['lower_grid']),row['old_degree'],ranges)
    assert all(v-F(1,2**64)<pre[k]<=v for k,v in exact.items());elevated=grid(row['degree4_grid'])
    for k,v in elevated.items():
        ww=[elevate_weights(n,jj)for n,jj in zip(row['old_degree'],k,strict=True)];direct=sum(z*prod(ww[a].get(e[a],F(0))for a in range(len(k)))for e,z in pre.items());assert direct==v;coefficient_checks+=1
    grids[row['id']]=row;g.journal(j,dict(kind='AFTER_NEW_independent_common_degree_certificate',factor=row['id'],degree4_coefficients=len(elevated),passed=True))
columns=M['columns'];maps={}
for index,c in enumerate(columns):maps.setdefault(tuple(c['scope']),{})[tuple(c['index'])]=index
for scope,table in maps.items():assert set(table)==set(product(range(5),repeat=len(scope)))
equality_checks=inequality_checks=0
for row in M['equalities']:
    actual={v['column']:f(v['value'])for v in row['coefficients']}
    if row['name'].startswith('normalize:'):
        scope=tuple(row['name'].split(':',1)[1].split(','));expected=dict.fromkeys(maps[scope].values(),F(1));assert f(row['bound'])==1
    else:
        text=row['name'].removeprefix('marginal:');pair,index=text.rsplit(':',1);parent,child=pair.split('->');parent=tuple(parent.split(','));child=tuple(child.split(','));k=ast.literal_eval(index);axes=[parent.index(v)for v in child];expected={c:F(1)for e,c in maps[parent].items()if tuple(e[a]for a in axes)==k};col=maps[child][k];expected[col]=expected.get(col,F(0))-1;assert f(row['bound'])==0
    assert actual==expected;equality_checks+=1
for row in M['inequalities']:
    actual={v['column']:f(v['value'])for v in row['coefficients']};meta=row['meta'];assert f(row['bound'])==0
    if meta['kind']=='necessary_source_or_time_lower_grid':
        r=grids[meta['factor']];scope=tuple(sorted(r['variables'],key=M['variables'].index));permutation=[r['variables'].index(v)for v in scope];expected={maps[scope][tuple(k[a]for a in permutation)]:v for k,v in grid(r['degree4_grid']).items()}
    else:
        scope=tuple(meta['scope']);axis=meta['axis'];others=[i for i in range(len(scope))if i!=axis];p=positive_power(meta['test']);coeff={k:sum(v*F(comb(k,n),comb(4,n))for n,v in p.items()if n<=k)for k in range(5)};expected={c:-coeff[k[axis]]for k,c in maps[scope].items()if tuple(k[a]for a in others)==tuple(meta['other'])}
    assert actual==normalized(expected);inequality_checks+=1
g.journal(j,dict(kind='AFTER_NEW_independent_model_semantics',equalities=equality_checks,inequalities=inequality_checks,passed=True))
# Independently verify the recorded finite-box dual, without solving any LP.
alpha=list(map(f,certificate['alpha']));beta=list(map(f,certificate['beta']));residual=[F(0)]*len(columns);rhs=F(0);assert all(v>=0 for v in alpha)
for rows,multipliers in[(M['inequalities'],alpha),(M['equalities'],beta)]:
    for row,m in zip(rows,multipliers,strict=True):
        rhs+=m*f(row['bound'])
        for c in row['coefficients']:residual[c['column']]+=m*f(c['value'])
lower=sum(min(F(0),v)for v in residual);gap=lower-rhs;assert residual==list(map(f,certificate['residual']))and gap==f(certificate['strict_gap'])and gap==0
g.journal(j,dict(kind='AFTER_NEW_independent_dual_diagnostic',gap=rec(gap),no_exclusion_proved=True))
x=list(map(F,solver['primal'][:-1]));assert len(x)==len(columns)and all(0<=v<=1 for v in x);cuts=[];diagnostics=[];zero_mass=0;univariate_negative=0
for scope,table in maps.items():
    for axis,name in enumerate(scope):
        others=[i for i in range(len(scope))if i!=axis]
        for other in product(range(5),repeat=len(others)):
            selected={k[axis]:c for k,c in table.items()if tuple(k[a]for a in others)==other};s0=sum(x[c]for c in selected.values());s1=sum(F(k,4)*x[c]for k,c in selected.items());s2=sum(F(k*(k-1),12)*x[c]for k,c in selected.items())
            record=dict(scope=list(scope),axis=axis,variable=name,conditioning_index=list(other),mass=rec(s0),first_moment=rec(s1),second_moment=rec(s2))
            g.journal(j,dict(kind='BEFORE_NEW_saved_table_square_separation',record=record))
            if not s0:
                assert s1==s2==0;zero_mass+=1;record['classification']='zero_conditioning_mass_NOT_missing_source'
            else:
                a=s1/s0;det=s0*s2-s1*s1;value=s2-2*a*s1+a*a*s0;assert value==det/s0;record.update(anchor=rec(a),moment_determinant=rec(det),minimum_square_expectation=rec(value),classification='negative_square_UNREALIZABLE_saved_table'if value<0 else'no_negative_square_in_this_group')
                if value<0:
                    coeff={k:F(k*(k-1),12)-2*a*F(k,4)+a*a for k in range(5)};co={c:-coeff[k]for k,c in selected.items()};actual=sum(v*x[c]for c,v in co.items());assert actual==-value and actual>0
                    cut=dict(scope=list(scope),axis=axis,conditioning_index=list(other),anchor=rec(a),coefficients=[dict(column=c,value=rec(v))for c,v in sorted(co.items())if v],bound=rec(F(0)),violated_by_exact_saved_table=rec(actual),nonnegative_polynomial_identity='(t-anchor)^2*product(other Bernstein weights)>=0',candidate_gates_or_source_truth_NOT_changed=True);cuts.append(cut)
                    if len(scope)==1:univariate_negative+=1
            g.journal(j,dict(kind='AFTER_NEW_saved_table_square_separation',record=record));diagnostics.append(record)
(O/'saved-table-negative-square-separators.json').write_text(json.dumps(g.plain(dict(cuts=cuts,diagnostics=diagnostics,not_evaluated_in_any_new_LP=True,these_certificates_reject_only_the_saved_table_NOT_the_source_time_family=True)),allow_nan=False,indent=2)+'\n')
j.close();out=dict(schema='v2d.common-moment-independent-evidence.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),independent_common_degree_coefficients=coefficient_checks,independent_equality_semantics=equality_checks,independent_inequality_semantics=inequality_checks,dual_gap=rec(gap),all_model_and_transfer_checks_passed=True,
    conditional_moment_groups_checked=len(diagnostics),zero_conditioning_mass_groups_retained=zero_mass,exact_negative_square_separators=len(cuts),univariate_negative_variance_groups=univariate_negative,
    numerical_relaxed_table_proven_NOT_a_real_common_measure=True,no_source_time_family_exclusion_or_pose_qualified=True,no_barycenter_or_root_reference_promoted=True,
    future_continuous_square_constraints_derived_NOT_solved=True,LPs_reexecuted=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,
    all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

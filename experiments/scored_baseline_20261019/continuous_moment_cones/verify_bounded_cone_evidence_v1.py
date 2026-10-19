"""Independent exact recorded cone/cut/selection/dual checks, no solver rerun."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
from math import comb
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-moment-cone-independent-v1';sys.path.insert(0,str(S/'compute'))
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'all-independent-cone-square-selection-and-dual-certificates.jsonl').open('w')
def f(v):return F(int(v['numerator_hex'],16),int(v['denominator_hex'],16))
def rec(v):return dict(numerator_hex=hex(v.numerator),denominator_hex=hex(v.denominator),approx=float(v))
def determinant(A):
    if len(A)==1:return A[0][0]
    if len(A)==2:return A[0][0]*A[1][1]-A[0][1]*A[1][0]
    a,b,c=A[0];d,e,z=A[1];h,i,k=A[2]
    return a*e*k+b*z*h+c*d*i-c*e*h-b*d*k-a*z*i
def signature(row):
    value=[(v['column'],hex(f(v['value']).numerator),hex(f(v['value']).denominator))for v in row['coefficients']]
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()
P=S/'e037-bounded-moment-cone-v1';report=read(P/'report.json')
for n,d in report['all_outputs_sha256'].items():assert sha(P/n)==d
outcome=read(P/'bounded-cone-outcome.json');base=read(S/'e037-common-moment-square-repair-v1/corrected-common-moment-LP-model.json');current=read(S/'e037-common-moment-square-repair-v1/corrective-phase-one-solver-RETURN.json');x=list(map(F,current['primal'][:-1]));previous=base;seen={signature(r)for r in base['inequalities']};candidates={};group_moments={};groups_checked=matrices_checked=minors_checked=square_checks=selected_checks=dual_checks=0;last_cone_bad=0
for line in(P/'all-continuous-cone-matrices-vectors-cuts-and-returns.jsonl').open():
    v=json.loads(line);kind=v['kind']
    if kind=='BEFORE_NEW_algorithm_cone_audit':
        iteration=v['iteration'];candidates={};group_moments={};last_cone_bad=0;assert len(x)==len(base['columns'])
    elif kind=='BEFORE_NEW_continuous_cone_group':
        group=v['group'];scope=group['scope'];axis=group['axis'];other=group['conditioning_index'];weights=[x[c]for c in group['columns']];assert weights==list(map(f,v['actual_weights']))
        for k,col in enumerate(group['columns']):
            descriptor=base['columns'][col];assert descriptor['scope']==scope and descriptor['index'][axis]==k and [z for a,z in enumerate(descriptor['index'])if a!=axis]==other
        y=[sum((F(comb(k,r),comb(4,r))if k>=r else F(0))*w for k,w in enumerate(weights))for r in range(5)];assert y==list(map(f,v['moments']));group_moments[group['id']]=(group,y,weights);groups_checked+=1
    elif kind=='AFTER_NEW_continuous_cone_principal_minors':
        group,y,weights=group_moments[v['group_id']];r=v['record'];A=[[f(z)for z in row]for row in r['matrix']];expected=[[y[i+k]for k in range(3)]for i in range(3)]if r['cone']=='Hankel'else[[y[i+k+1]-y[i+k+2]for k in range(2)]for i in range(2)];assert A==expected
        for minor in r['principal_minors']:
            idx=minor['indices'];assert determinant([[A[i][k]for k in idx]for i in idx])==f(minor['value']);minors_checked+=1
        PSD=all(f(z['value'])>=0 for z in r['principal_minors']);assert PSD==r['exact_principal_minors_PSD'];last_cone_bad+=not PSD;matrices_checked+=1
    elif kind=='AFTER_NEW_exact_cone_square_RETURN':
        cut=v['cut'];group,y,weights=group_moments[v['group_id']];vector=list(map(f,cut['vector']));power={}
        for i,a in enumerate(vector):
            for k,b in enumerate(vector):power[i+k]=power.get(i+k,F(0))+a*b
        if cut['cone']=='localizer':
            z={}
            for k,w in power.items():z[k+1]=z.get(k+1,F(0))+w;z[k+2]=z.get(k+2,F(0))-w
            power=z
        assert power=={z['degree']:f(z['value'])for z in cut['power']};B=[sum(w*F(comb(k,r),comb(4,r))for r,w in power.items()if r<=k)for k in range(5)];expectation=sum(a*b for a,b in zip(B,weights,strict=True));assert expectation==f(cut['exact_square_expectation']);co={c:-w for c,w in zip(group['columns'],B,strict=True)if w};scale=max(map(abs,co.values()));normalized={k:w/scale for k,w in co.items()};assert normalized=={z['column']:f(z['value'])for z in cut['coefficients']};assert f(cut['normalized_violation'])==-expectation/scale;assert signature(cut)==cut['cut_sha256'];square_checks+=1
        if cut['classification']=='exact_violated_necessary_cone_plane':
            assert expectation<0
            if cut['cut_sha256']not in seen:candidates.setdefault(cut['cut_sha256'],cut)
        else:assert expectation>=0
    elif kind=='AFTER_NEW_algorithm_cone_audit_RETURN':
        ranked=sorted(candidates.values(),key=lambda r:(-f(r['normalized_violation']),r['cut_sha256']));assert len(ranked)==v['audit']['available_unique_new_planes'];assert not v['audit']['all_exact_principal_minors_PSD']
        if iteration<6:
            selected=read(P/f'update-{iteration:02d}-selected-exact-planes.json');assert [r['cut_sha256']for r in selected]==[r['cut_sha256']for r in ranked[:64]]and len(selected)==64
            for r in selected:seen.add(r['cut_sha256']);selected_checks+=1
        g.journal(j,dict(kind='AFTER_NEW_independent_cone_audit_certificates',iteration=iteration,groups=590,negative_exact_matrices=last_cone_bad,new_candidate_planes=len(ranked),selected=64 if iteration<6 else 0,passed=True))
    elif kind=='BEFORE_NEW_cone_algorithm_LP_subproblem':
        model=read(v['model']);assert sha(v['model'])==v['model_sha256'];assert model['columns']==previous['columns']and model['equalities']==previous['equalities']and model['inequalities'][:len(previous['inequalities'])]==previous['inequalities'];assert len(model['inequalities'])==len(previous['inequalities'])+64
        selected=read(P/f'update-{iteration:02d}-selected-exact-planes.json')
        for row,cut in zip(model['inequalities'][-64:],selected,strict=True):assert row['coefficients']==cut['coefficients']and f(row['bound'])==0 and row['meta']['cut_sha256']==cut['cut_sha256']
        assert{key:z for key,z in model.items()if key!='inequalities'}=={key:z for key,z in previous.items()if key!='inequalities'};previous=model
    elif kind=='AFTER_NEW_cone_LP_solver_RETURN':
        current=read(v['return_file']);assert sha(v['return_file'])==v['return_sha256']and current['status']==v['status'];x=list(map(F,current['primal'][:-1]));assert current['success']and current['objective']==0
    elif kind=='AFTER_NEW_cone_LP_exact_dual_RETURN':
        dual=read(P/f'update-{iteration:02d}-finite-box-dual.json');alpha=list(map(f,dual['alpha']));beta=list(map(f,dual['beta']));r=[F(0)]*len(previous['columns']);rhs=F(0);assert all(z>=0 for z in alpha)
        for rows,multipliers in[(previous['inequalities'],alpha),(previous['equalities'],beta)]:
            for row,m in zip(rows,multipliers,strict=True):
                if not m:continue
                rhs+=m*f(row['bound'])
                for z in row['coefficients']:r[z['column']]+=m*f(z['value'])
        gap=sum(min(F(0),z)for z in r)-rhs;assert r==list(map(f,dual['residual']))and gap==f(dual['strict_gap'])and gap==0;dual_checks+=1
assert len(outcome['updates'])==6 and len(outcome['audits'])==7 and selected_checks==384 and groups_checked==4130 and matrices_checked==8260 and dual_checks==6
eq=[sum(f(v['value'])*x[v['column']]for v in row['coefficients'])-f(row['bound'])for row in previous['equalities']];ineq=[sum(f(v['value'])*x[v['column']]for v in row['coefficients'])-f(row['bound'])for row in previous['inequalities']];violations=[dict(column=i,value=rec(v),violation=rec(max(F(0),-v,v-1)))for i,v in enumerate(x)if not 0<=v<=1]
j.close();data=dict(schema='v2d.bounded-cone-independent-certificates.v1',groups_checked=groups_checked,matrices_checked=matrices_checked,principal_minor_certificates_checked=minors_checked,all_rational_square_planes_checked=square_checks,selected_planes_checked=selected_checks,exact_top64_selections_checked=6,finite_box_duals_checked=dual_checks,all_dual_gaps_zero=True,last_negative_exact_cone_matrices=last_cone_bad,last_actual_unit_box_failures=violations,last_exact_primal_equality_residual=rec(max(map(abs,eq))),last_exact_primal_inequality_residual=rec(max(ineq)),all_independent_checks_passed=True,
    initial_saved_table_rechecked_under_new_cone_maps_NOT_new_source_or_physical_evidence=True,classification='UNRESOLVED_declared_conic_algorithm_budget',no_exclusion_or_realizable_pose_qualified=True,cone_algorithm_or_eigendecompositions_or_LPs_NOT_rerun=True)
(O/'independent-cone-certificate-results.json').write_text(json.dumps(g.plain(data),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.bounded-cone-independent-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),**data,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

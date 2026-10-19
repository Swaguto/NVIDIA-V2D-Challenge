"""New analytical contracts for joint squares in saved degree4 tensor maps."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from math import comb
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-mixed-joint-contract-v1';sys.path.insert(0,str(S/'compute'))
import mixed_joint_moments_v1 as m
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();j=(O/'all-new-mixed-joint-contracts-and-intermediates.jsonl').open('w');emit=lambda v:g.journal(j,v);controls=[];start=time.monotonic()
def check(name,work):
    emit(dict(kind='BEFORE_NEW_mixed_joint_contract',name=name));v=work();emit(dict(kind='AFTER_NEW_mixed_joint_contract_RETURN',name=name,result=v));controls.append(dict(name=name,passed=bool(v['passed'])));assert v['passed']
def fixture(dim,atoms):
    entries={a:k for k,a in enumerate(product(range(5),repeat=dim))};weights=[sum(mass*__import__('functools').reduce(lambda a,b:a*b,[F(comb(4,k))*t**k*(1-t)**(4-k)for k,t in zip(index,point)],F(1))for point,mass in atoms)for index in entries];return entries,weights
def moments(entries,weights):return {a:sum(v*weights[col]for col,v in m.monomial_coefficients(entries,a).items())for a in m.basis(len(next(iter(entries))),4)}
def tensor_identity():
    count=0
    for dim in [2,3]:
        point=[F(3,11),F(2,7),F(5,13)][:dim];e,w=fixture(dim,[(point,F(7,9))]);y=moments(e,w)
        for a,v in y.items():
            expected=F(7,9)
            for t,r in zip(point,a):expected*=t**r
            assert v==expected;count+=1
    return dict(passed=True,exact_mixed_monomial_identities=count,existing_per_axis_degree=4,no_source_or_episode_input=True)
check('bivariate_trivariate_tensor_to_cross_moment_binding',tensor_identity)
def square_identity():
    count=0
    for dim in [2,3]:
        atoms=[([F(2,9),F(3,8),F(4,7)][:dim],F(5,6)),([F(6,7),F(1,3),F(2,5)][:dim],F(-5,6))];e,w=fixture(dim,atoms);y=moments(e,w)
        assert sum(w)==0
        for axis in [None,*range(dim)]:
            basis=m.basis(dim,2 if axis is None else 1);v=[F((-1)**k*(k+1),k+3)for k in range(len(basis))];M=m.matrix_from_moments(basis,y,axis);poly=m.squared_polynomial(basis,v,axis);co=m.polynomial_coefficients(e,poly);expectation=sum(a*w[col]for col,a in co.items());direct=F(0)
            for point,mass in atoms:
                q=F(0)
                for power,a in zip(basis,v):
                    term=a
                    for t,r in zip(point,power):term*=t**r
                    q+=term
                direct+=mass*q*q*(point[axis]*(1-point[axis])if axis is not None else 1)
            assert direct==expectation==m.quadratic(M,v);count+=1
    return dict(passed=True,mixed_square_and_localizer_identities=count,zero_signed_mass_preserved=True,actual_weight_normalization=False)
check('joint_square_and_axis_localizer_identity_with_zero_signed_mass',square_identity)
def localized_boundary():
    counts=[]
    for dim in [2,3]:
        for axis in range(dim):
            for endpoint in [F(0),F(1)]:
                point=[F(2,7)]*dim;point[axis]=endpoint;e,w=fixture(dim,[(point,F(1))]);M=m.matrix_from_moments(m.basis(dim,1),moments(e,w),axis);assert all(v==0 for row in M for v in row);result=m.congruence_certificate(M,emit);assert result['classification']=='exact_mixed_matrix_PSD'and result['rank']==0;counts.append([dim,axis,int(endpoint)])
    return dict(passed=True,all_boundary_localizers_zero=counts,zero_mass_or_zero_pivots_not_skipped=True)
check('interval_localizer_support_and_zero_boundary_matrices',localized_boundary)
def congruence():
    cases=[([[F(2),F(1)],[F(1),F(3)]],True),([[F(1),F(3),F(5)],[F(3),F(13),F(23)],[F(5),F(23),F(41)]],True),([[F(-1),F(0)],[F(0),F(2)]],False),([[F(0),F(2)],[F(2),F(3)]],False)];returns=[]
    for M,psd in cases:
        cert=m.congruence_certificate(M,emit)
        if psd:
            B=[[c.dec(v)for v in col]for col in cert['basis_columns']];D=list(map(c.dec,cert['diagonal']));assert all(sum(B[i][a]*M[a][b]*B[k][b]for a in range(len(M))for b in range(len(M)))==(D[i]if i==k else 0)for i in range(len(M))for k in range(len(M)));assert all(v>=0 for v in D)
        else:assert m.quadratic(M,list(map(c.dec,cert['vector'])))==c.dec(cert['expectation'])<0
        returns.append(dict(expected_PSD=psd,classification=cert['classification'],reason=cert.get('reason'),rank=cert.get('rank')))
    return dict(passed=True,cases=returns)
check('exact_congruence_PD_rank_deficient_negative_and_zero_cross_pivots',congruence)
def marginal_false_positive():
    e={a:k for k,a in enumerate(product(range(5),repeat=2))};w=[F(int(i==k),5)for i,k in e];y=moments(e,w);basis=m.basis(2,2);v=[F(int(a==(1,0))-int(a==(0,1)))for a in basis];M=m.matrix_from_moments(basis,y);E=m.quadratic(M,v);witness=m.square_witness(e,basis,None,v,w);assert E==F(-1,12)==c.dec(witness['actual_exact_expectation'])
    for axis in range(2):assert all(y[tuple(r if i==axis else 0 for i in range(2))]==F(1,r+1)for r in range(5))
    return dict(passed=True,negative_joint_square=c.rec(E),uniform_individual_marginals=True,NOT_claiming_all_old_conditional_PSD_pass_for_this_fixture=True,witness=witness)
check('consistent_univariate_marginals_do_not_establish_joint_square_positivity',marginal_false_positive)
def distance_bound():
    e={a:k for k,a in enumerate(product(range(5),repeat=2))};w=[F(int(i==k),5)for i,k in e];basis=m.basis(2,2);v=[F(int(a==(1,0))-int(a==(0,1)))for a in basis];z=m.square_witness(e,basis,None,v,w);co={v['column']:c.dec(v['value'])for v in z['Bernstein_expectation_coefficients']};E=c.dec(z['actual_exact_expectation']);N=c.dec(z['coefficient_linf']);distance=c.dec(z['necessary_l1_weight_change_to_remove_THIS_witness']);assert distance==-E/N>0
    perturb=[F((-1)**i,1000*(i+1))for i in range(25)];change=sum(co.get(i,0)*v for i,v in enumerate(perturb));bound=N*sum(map(abs,perturb));assert abs(change)<=bound
    return dict(passed=True,exact_l1_necessity=c.rec(distance),perturbation_dual_norm_bound=c.rec(bound),actual_weights_NOT_repaired=True)
check('square_certificate_residual_bound_is_not_a_table_repair',distance_bound)
def permutation():
    point=[F(2,9),F(3,7),F(5,8)];e,w=fixture(3,[(point,F(1))]);y=moments(e,w);other,ww=fixture(3,[(point[::-1],F(1))]);yy=moments(other,ww);assert all(v==yy[a[::-1]]for a,v in y.items());return dict(passed=True,axis_permutation_moments_checked=len(y),no_scope_hull_or_correlation_dropped=True)
check('mixed_axis_order_and_clique_correlations_are_explicit',permutation)
def mixture_not_point():
    e,w=fixture(2,[([F(1,4),F(1,3)],F(1,2)),([F(3,4),F(2,3)],F(1,2))]);y=moments(e,w);M=m.matrix_from_moments(m.basis(2,2),y);cert=m.congruence_certificate(M,emit);variance=y[(2,0)]-y[(1,0)]**2;assert cert['classification']=='exact_mixed_matrix_PSD'and cert['rank']==2 and variance==F(1,16)
    return dict(passed=True,PSD_mixed_matrix_rank=2,nonzero_marginal_variance=c.rec(variance),PSD_NOT_deterministic_assignment=True)
check('mixed_PSD_finite_measure_is_not_a_deterministic_joint_path',mixture_not_point)
def invalid():
    errors=[];e={a:k for k,a in enumerate(product(range(5),repeat=2))}
    calls=[lambda:m.monomial_coefficients(e,(5,0)),lambda:m.monomial_coefficients(e,(1,)),lambda:m.tables([dict(scope=['a','a'],index=[0,0])]),lambda:m.tables([dict(scope=['a','b'],index=[0,0])]),lambda:m.congruence_certificate([[F(0),F(1)],[F(2),F(0)]],emit)]
    for fn in calls:
        try:fn();errors.append(None)
        except ValueError as ex:errors.append(str(ex))
    return dict(passed=all(errors),retained_invalid_inputs=errors,no_unavailable_moment_silently_inferred=True)
check('strict_degree_scope_symmetry_and_complete_tensor_inputs',invalid)
j.close();out=dict(schema='v2d.mixed-joint-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/mixed_joint_moments_v1.py'),controls=controls,new_analytic_controls=len(controls),all_controls_passed=all(v['passed']for v in controls),new_solver_calls=0,old_control_or_producer_reexecutions=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

"""New analytical support-localizer and saved-coefficient error contracts."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from math import comb
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-pointwise-support-contract-v1';sys.path.insert(0,str(S/'compute'))
import pointwise_support_moments_v1 as p
import mixed_joint_moments_v1 as m
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest();j=(O/'all-new-support-contracts-and-intermediate-returns.jsonl').open('w');emit=lambda v:g.journal(j,v);controls=[];start=time.monotonic()
def check(name,work):
    emit(dict(kind='BEFORE_NEW_pointwise_support_contract',name=name));v=work();emit(dict(kind='AFTER_NEW_pointwise_support_contract_RETURN',name=name,result=v));controls.append(dict(name=name,passed=bool(v['passed'])));assert v['passed']
def fixture(atoms,dim=2):
    e={a:k for k,a in enumerate(product(range(5),repeat=dim))};w=[]
    for index in e:
        value=F(0)
        for point,mass in atoms:
            term=mass
            for t,k in zip(point,index):term*=comb(4,k)*t**k*(1-t)**(4-k)
            value+=term
        w.append(value)
    return e,w
def good_support():
    e,w=fixture([([F(3,5),F(1,4)],F(2,7)),([F(9,10),F(3,4)],F(5,7))]);poly={(1,0):F(1),(0,0):F(-1,2)};basis,_=p.multiplier_basis([1,0]);M,_=p.localizer_maps(e,poly,basis,w,emit);certificate=m.congruence_certificate(M,emit);return dict(passed=certificate['classification']=='exact_mixed_matrix_PSD',weighted_support_matrix_rank=certificate.get('rank'),atoms_satisfy_original_support=True)
check('positive_measure_supported_on_g_nonnegative_gives_weighted_PSD',good_support)
def mean_not_support():
    e,w=fixture([([F(0),F(1,3)],F(2,5)),([F(1),F(2,3)],F(3,5))]);poly={(1,0):F(1),(0,0):F(-1,2)};basis,_=p.multiplier_basis([1,0]);M,_=p.localizer_maps(e,poly,basis,w,emit);v=[F(int(a==(0,0))-int(a==(1,0)))for a in basis];witness=p.support_square(e,poly,basis,v,w);expected=c.dec(witness['actual_exact_expectation']);assert M[0][0]==F(1,10)and expected==F(-1,5)
    return dict(passed=True,scalar_expected_gate_passes=c.rec(M[0][0]),support_square_fails=c.rec(expected),actual_nonnegative_measure_can_have_invalid_support=True,witness=witness)
check('positive_scalar_mean_does_not_certify_individual_state_motion_support',mean_not_support)
def enclosure():
    intervals=[dict(exponent=[0,0],coefficient_lower=c.rec(F(-2,7)),coefficient_upper=c.rec(F(-1,7))),dict(exponent=[1,1],coefficient_lower=c.rec(F(3,11)),coefficient_upper=c.rec(F(4,11))),dict(exponent=[3,0],coefficient_lower=c.rec(F(1,1<<201)),coefficient_upper=c.rec(F(1,1<<200)))];bounds=[[F(-2,3),F(3,4)],[F(1,5),F(4,5)]];upper,degrees,record=p.upper_polynomial(intervals,bounds,emit);checked=0
    for select in product(range(2),repeat=len(intervals)):
        actual={tuple(row['exponent']):c.dec(row['coefficient_lower'if k==0 else'coefficient_upper'])for row,k in zip(intervals,select)}
        for t in product([F(-2,3),F(0),F(3,4)],[F(1,5),F(1,2),F(4,5)]):assert p.evaluate(upper,t)>=p.evaluate(actual,t);checked+=1
    tiny=record['coefficient_records'][-1];assert tiny['rounded_center_zero_is_NOT_structural_zero']and degrees==[3,1]
    return dict(passed=True,coefficient_endpoint_point_bounds=checked,retained_structural_degrees=degrees,tiny_nonzero_not_relabelled_zero=True,arithmetic_guard_NOT_candidate_tolerance=True)
check('uniform_upper_bound_preserves_tiny_structural_terms_and_uncertainty',enclosure)
def affine():
    poly={(0,0):F(3,7),(1,0):F(-2,9),(2,1):F(5,11)};bounds=[[F(2,7),F(3,4)],[F(-1,5),F(2,3)]];pulled=p.affine_pullback(poly,bounds,emit);reordered=p.reorder(pulled,['b','a'],['a','b']);count=0
    for a,b in product([F(0),F(2,5),F(1)],repeat=2):
        actual=[lo+(hi-lo)*t for (lo,hi),t in zip(bounds,[a,b])];assert p.evaluate(poly,actual)==p.evaluate(pulled,[a,b])==p.evaluate(reordered,[b,a]);count+=1
    return dict(passed=True,exact_affine_ordered_closures=count)
check('saved_half_angle_affine_pullback_and_explicit_scope_permutation',affine)
def availability():
    cases={};expected={(1,1):4,(1,2,1):8,(2,2):4,(4,4):1}
    for degree,size in expected.items():
        basis,rejected=p.multiplier_basis(list(degree));assert len(basis)==size
        for a,b in product(basis,repeat=2):assert all(r+x+y<=4 for r,x,y in zip(degree,a,b))
        cases[str(degree)]=dict(basis=[list(a)for a in basis],unavailable=rejected)
    return dict(passed=True,cases=cases,source_degree44_nonconstant_UNRESOLVED=True,no_missing_moment_added=True)
check('structural_degree_capacity_and_nonconstant_source_unavailability',availability)
def weighted_identity():
    atoms=[([F(2,9),F(1,4),F(3,7)],F(4,5)),([F(5,8),F(4,9),F(1,3)],F(-4,5))];e,w=fixture(atoms,3);poly={(0,0,0):F(2,7),(1,2,1):F(-3,11),(1,0,1):F(4,9)};basis,_=p.multiplier_basis([1,2,1]);M,_=p.localizer_maps(e,poly,basis,w,emit);v=[F((-1)**k*(k+1),k+3)for k in range(8)];witness=p.support_square(e,poly,basis,v,w);direct=F(0)
    for point,mass in atoms:
        q={a:z for a,z in zip(basis,v)};direct+=mass*p.evaluate(poly,point)*p.evaluate(q,point)**2
    # support_square scales CERTIFICATE coefficients only.
    scale=max(map(abs,v));assert direct/scale**2==c.dec(witness['actual_exact_expectation'])and direct==m.quadratic(M,v)and sum(w)==0
    return dict(passed=True,exact_signed_zero_mass_weighted_square_identity=c.rec(direct),raw_weights_NOT_normalized=True)
check('trivariate_weighted_square_linear_map_for_signed_zero_mass',weighted_identity)
def positive_depth():
    e,w=fixture([([F(1,3),F(2,5)],F(1))]);poly={(0,0):F(-1,7)};basis,_=p.multiplier_basis([0,0]);M,_=p.localizer_maps(e,poly,basis,w,emit);cert=m.congruence_certificate(M,emit)
    denominators=[(1+u*u)*(1+v*v)for u,v in product([F(-2),F(0),F(3)],repeat=2)];assert all(a>=1 for a in denominators)and cert['classification']=='exact_negative_mixed_square'
    return dict(passed=True,negative_depth_cannot_be_hidden_by_positive_finite_chart_denominator=True,source_image_inequality_does_not_replace_separate_depth_gate=True,minimum_test_denominator=c.rec(min(denominators)))
check('depth_support_is_separate_from_source_image_and_chart_denominator',positive_depth)
def strict():
    errors=[];e,w=fixture([([F(1,2),F(1,2)],F(1))])
    calls=[lambda:p.localizer_maps(e,{(4,0):F(1)},[(1,0)],w,emit),lambda:p.reorder({(0,0):F(1)},['a','b'],['a','a']),lambda:p.upper_polynomial([],[[F(1),F(0)]],emit)]
    for fn in calls:
        try:fn();errors.append(None)
        except ValueError as ex:errors.append(str(ex))
    return dict(passed=all(errors),errors=errors,unavailable_or_malformed_inputs_NOT_silently_filled=True)
check('unavailable_power_and_malformed_scope_fail_explicitly',strict)
j.close();out=dict(schema='v2d.pointwise-support-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/pointwise_support_moments_v1.py'),controls=controls,new_analytic_controls=len(controls),all_controls_passed=all(v['passed']for v in controls),new_solver_calls=0,episode_inputs_used=False,old_controls_or_producers_reexecuted=False,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={q.name:sha(q)for q in O.iterdir()if q.is_file()and q.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

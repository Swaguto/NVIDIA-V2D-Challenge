"""NEW continuous cone separation contracts before bounded episode algorithm."""
from pathlib import Path
from fractions import Fraction as F
from math import comb
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-moment-cone-contract-v1';sys.path.insert(0,str(S/'compute'))
import conditional_moment_cones_v1 as c
import bernstein_common_moments_v1 as m
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();j=(O/'all-new-continuous-cone-contracts-and-returns.jsonl').open('w');emit=lambda v:g.journal(j,v);controls=[];group=dict(id=0,scope=['a'],axis=0,conditioning_index=[],columns=list(range(5)))
def check(name,work):
    emit(dict(kind='BEFORE_NEW_continuous_cone_control',name=name));v=work();emit(dict(kind='AFTER_NEW_continuous_cone_control',name=name,result=v));controls.append(dict(name=name,passed=v['passed']));assert v['passed']
def bind():
    mass=F(7,13);t=F(3,7);u=F(5,8);w=[mass*(F(2,5)*comb(4,k)*t**k*(1-t)**(4-k)+F(3,5)*comb(4,k)*u**k*(1-u)**(4-k))for k in range(5)];y=c.moments(w);expected=[mass*(F(2,5)*t**r+F(3,5)*u**r)for r in range(5)]
    return dict(passed=y==expected and y[0]==mass,conditional_mass=c.rec(mass),moments=list(map(c.rec,y)))
check('unnormalized_conditioned_moment_linear_maps',bind)
def polynomial_identity(kind,vector):
    w=[F(1,7),F(-1,9),F(2,5),F(1,11),F(1,3)];y=c.moments(w);A=c.matrices(y)[kind];p,B=c.square_coefficients(vector,kind);expectation=sum(a*b for a,b in zip(B,w,strict=True));direct=sum(a*A[i][k]*b for i,a in enumerate(vector)for k,b in enumerate(vector));tests=[]
    for t in[F(i,19)for i in range(20)]:
        value=sum(v*t**k for k,v in p.items());actual=sum(v*t**k for k,v in enumerate(vector))**2
        if kind=='localizer':actual*=t*(1-t)
        tests.append(value==actual and actual>=0)
    return dict(passed=expectation==direct and all(tests),signed_input_preserved=True,exact_expectation=c.rec(expectation),coefficient_identity_values=len(tests))
check('quadratic_square_Hankel_cut_identity_for_signed_inputs',lambda:polynomial_identity('Hankel',[F(2,3),F(-5,7),F(3,11)]))
check('unit_interval_localizer_square_cut_identity',lambda:polynomial_identity('localizer',[F(3,4),F(-2,7)]))
def uniform():
    result=c.separate(group,[F(1,5)]*5,emit)
    return dict(passed=result['all_exact_principal_minors_PSD']and not result['cuts'],return_value=result)
check('full_rank_uniform_measure_continuous_PSD',uniform)
def negative_control(weights):
    result=c.separate(group,weights,emit)
    return dict(passed=not result['all_exact_principal_minors_PSD']and bool(result['cuts'])and all(c.dec(v['exact_square_expectation'])<0 for v in result['cuts']),result=result)
check('exact_eigenvector_square_separator',lambda:negative_control([F(0),F(1),F(0),F(0),F(0)]))
check('zero_mass_signed_conditioning_not_discarded',lambda:negative_control([F(1),F(-1),F(0),F(0),F(0)]))
check('negative_mass_not_clipped_or_normalized',lambda:negative_control([F(-1,3),F(0),F(0),F(0),F(0)]))
def signature():
    w=[F(0),F(1),F(0),F(0),F(0)];a=c.cut_from_vector(group,w,[F(1),F(-3),F(2)],'Hankel');b=c.cut_from_vector(group,w,[F(-2),F(6),F(-4)],'Hankel');zero=c.cut_from_vector(group,w,[F(0)]*3,'Hankel')
    return dict(passed=a['cut_sha256']==b['cut_sha256']and zero['classification']=='UNRESOLVED_zero_polynomial',scale_sign_invariant_sha256=a['cut_sha256'],zero_retained=zero)
check('exact_scale_invariant_plane_deduplication',signature)
def conic_phase():
    cuts=c.separate(group,[F(0),F(1),F(0),F(0),F(0)],emit)['cuts'];cut=cuts[0];model=dict(columns=[dict(scope=['a'],index=[i])for i in range(5)],equalities=[dict(name='fixed:'+str(i),coefficients=[dict(column=i,value=c.rec(1))],bound=c.rec(int(i==1)))for i in range(5)],inequalities=[dict(name='continuous_cone_plane',coefficients=cut['coefficients'],bound=c.rec(0))]);emit(dict(kind='BEFORE_NEW_analytic_conic_phase_control_LP',model=model));result=m.solve_phase_one(model,time_limit=10);emit(dict(kind='AFTER_NEW_analytic_conic_phase_control_LP_RETURN',result=result));dual=m.finite_box_dual(model,result);emit(dict(kind='AFTER_NEW_analytic_conic_phase_control_exact_dual_RETURN',certificate=dual))
    return dict(passed=result['success']and dual['classification']=='exact_saved_LP_model_INFEASIBILITY_certificate',objective=result['objective'],strict_dual_gap=dual.get('strict_gap'))
check('continuous_cone_cut_phase_I_and_exact_finite_box_dual',conic_phase)
def strict():
    errors=[]
    for call in[lambda:c.groups([dict(scope=['a'],index=[0])]),lambda:c.groups([dict(scope=['a','a'],index=[0,0])]),lambda:c.moments([F(0)]*4),lambda:c.square_coefficients([F(1)]*2,'Hankel'),lambda:c.moments([float('nan')]+[0]*4)]:
        try:call();errors.append(None)
        except (ValueError,OverflowError)as e:errors.append(type(e).__name__+':'+str(e))
    return dict(passed=all(errors),errors=errors)
check('strict_maps_shapes_and_nonfinite_contracts',strict)
j.close();out=dict(schema='v2d.continuous-moment-cone-contract-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/conditional_moment_cones_v1.py'),new_analytic_controls=controls,all_controls_passed=all(v['passed']for v in controls),new_tiny_analytic_conic_LPs=1,episode_LPs=0,no_old_common_moment_controls_or_source_jobs_repeated=True,local_PSD_necessity_NOT_global_measure_or_pose_sufficiency=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

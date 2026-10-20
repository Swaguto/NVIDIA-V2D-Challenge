"""NEW derivative/merit/damping/exact-stop and coupled construction contracts."""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,os,sys,time
import numpy as np
S=Path(__file__).resolve().parents[1];O=S/'e037-primal-restoration-contract-v1';sys.path.insert(0,str(S/'compute'))
import primal_inequality_restoration_v1 as p
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest();start=time.monotonic();J=(O/'all-NEW-primal-contract-and-analytic-iterate-returns.jsonl').open('w');emit=lambda v:g.journal(J,v);checks=[]
def save(n,v):
 with(O/n).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def record(name,scope,poly,sense='>='):return dict(id=name,scope=scope,coefficients=[dict(power=list(k),value=c.rec(v))for k,v in sorted(poly.items())if v],sense=sense)
def check(name,proposal,work):
 emit(dict(kind='BEFORE_NEW_primal_contract',name=name,proposal=proposal));result=work();emit(dict(kind='AFTER_NEW_primal_contract_RETURN',name=name,RETURN=result));checks.append(dict(name=name,RETURN=result))
def exact_value(row,x):
 return sum(c.dec(v['value'])*__import__('functools').reduce(lambda a,b:a*b,[x[n]**k for n,k in zip(row['scope'],v['power'])],F(1))for v in row['coefficients'])

def derivatives():
 row=record('mixed',['b','a'],{(4,3):F(2,7),(1,2):F(-5,11),(0,0):F(1,13),(2,4):F(1,1<<150)});system=p.PolynomialSystem([row],['a','b'],{'a':[F(-1),F(1)],'b':[F(-1),F(1)]},emit);point={'a':F(2,7),'b':F(-3,11)};h=F(1,17);cert=[]
 for name in['a','b']:
  vals=[]
  for k in[-2,-1,1,2]:
   shifted=dict(point);shifted[name]+=k*h;vals.append(exact_value(row,shifted))
  finite=(vals[0]-8*vals[1]+8*vals[2]-vals[3])/(12*h);mapped=F(0)
  for m in system.derivative_maps[0]:
   if m['variable']!=name:continue
   v=c.dec(row['coefficients'][m['original_term_index']]['value'])*m['multiplicity']
   for n,k in zip(system.names,m['derivative_global_power']):v*=point[n]**k
   mapped+=v
  cert.append(dict(variable=name,exact_four_point_derivative=c.rec(finite),index_map_derivative=c.rec(mapped)));emit(dict(kind='AFTER_NEW_exact_derivative_control_identity',RETURN=cert[-1]));assert finite==mapped
 return dict(exact_derivative_certificates=cert,structural_tiny_coefficient_preserved=True,axis_order_explicit=True)
check('exact_multivariate_derivative_maps_vs_degree4_four_point_identity',dict(no_source_polynomial_regenerated=True),derivatives)

def signs():
 r=record('less_than',['x'],{(1,):F(1),(0,):F(-1,3)},'<=');s=p.PolynomialSystem([r],['x'],{'x':[F(0),F(1)]},emit);a=s.exact_check([0.],emit);b=s.exact_check([.5],emit);assert a['all_exact_model_inequalities_passed']and not b['all_exact_model_inequalities_passed'];return dict(zero_passes=a,half_rejected=b,positive_normalization_does_NOT_change_sign=True)
check('sense_and_positive_static_normalization_bind_original_inequality',dict(no_candidate_tolerance=True),signs)

def strict():
 s=p.PolynomialSystem([record('strict_zero',['x'],{},'>')],['x'],{'x':[F(0),F(1)]},emit);value=s.evaluate([0.]);phi,res=p.merit(value);result=s.exact_check([0.],emit);assert phi<1e-18 and not result['all_exact_model_inequalities_passed'];return dict(near_zero_merit=phi,exact_strict_zero_rejected=result,merit_NEVER_qualification=True)
check('near_zero_merit_does_NOT_accept_zero_strict_depth',dict(no_normalization_of_missing_or_signed_mass=True),strict)

def SPD():
 J0=[[F(1),F(1)],[F(2),F(2)]];lam=F(1,10**6);H=[[sum(row[i]*row[j]for row in J0)+(lam if i==j else F(0))for j in range(2)]for i in range(2)];det=H[0][0]*H[1][1]-H[0][1]**2;emit(dict(kind='AFTER_NEW_exact_singular_Jacobian_damping_RETURN',matrix=[[c.rec(v)for v in row]for row in H],determinant=c.rec(det)));assert H[0][0]>0 and det>0;return dict(positive_first_minor=c.rec(H[0][0]),positive_determinant=c.rec(det),regularization_is_NOT_motor_or_source_weight=True)
check('constant_damping_makes_rank_deficient_normal_matrix_SPD',dict(damping='1e-6 algorithmic'),SPD)

def zero_safe():
 s=p.PolynomialSystem([record('zero_coordinate',['x','y'],{(1,0):F(3),(0,1):F(-4),(2,3):F(5)})],['x','y'],{'x':[F(-1),F(1)],'y':[F(-1),F(1)]},emit);v,j=s.evaluate([0.,0.],True);scale=s.scales[0];assert v[0]==0 and j[0,0]==3/scale and j[0,1]==-4/scale;return dict(value=v,Jacobian=j,zero_division_avoided=True)
check('zero_coordinate_derivatives_use_power_decrement_not_division',dict(poles_not_introduced=True),zero_safe)

def aliases():
 rows=[record('ab',['a','b'],{(1,1):F(1)}),record('ba',['b','a'],{(1,1):F(1)})];s=p.PolynomialSystem(rows,['a','b'],{'a':[F(0),F(1)],'b':[F(0),F(1)]},emit);v,j=s.evaluate([.25,.75],True);assert np.array_equal(j[0],j[1])and v[0]==v[1];return dict(shared_global_Jacobian=j,duplicate_axes_NOT_independent_observations=True)
check('shared_variables_keep_same_global_Jacobian_across_reordered_scopes',dict(no_product_mean_fill=True),aliases)

def invalid():
 messages=[]
 for rows,names,domains in[([record('bad',['x'],{(1,):F(1)})],['x','x'],{'x':[F(0),F(1)]}),([record('bad',['absent'],{(1,):F(1)})],['x'],{'x':[F(0),F(1)]})]:
  try:p.PolynomialSystem(rows,names,domains,emit)
  except ValueError as e:messages.append(str(e))
  else:raise AssertionError('invalid model accepted')
 s=p.PolynomialSystem([record('linear',['x'],{(1,):F(1)})],['x'],{'x':[F(0),F(1)]},emit)
 try:s.evaluate([float('nan')])
 except ValueError as e:messages.append(str(e))
 else:raise AssertionError('nonfinite point accepted')
 return dict(rejections=messages)
check('missing_scope_duplicate_variables_and_nonfinite_values_explicit',dict(no_silent_filling=True),invalid)

def outside():
 s=p.PolynomialSystem([record('box',['x'],{(1,):F(1)}),record('upper',['x'],{(0,):F(1),(1,):F(-1)})],['x'],{'x':[F(0),F(1)]},emit);initial=np.array([-.25]);copy=initial.copy();v=s.evaluate(initial);assert np.array_equal(initial,copy)and v[0]<0;return dict(actual_initial_vector=initial,normalized_violations=v,no_clipping_or_feasibility_claim=True)
check('out_of_domain_initial_F_analogue_is_retained_as_violation',dict(no_center_or_seed_change=True),outside)

def armijo():
 phi=F(1,2);slope=F(-1);alpha=F(1,2);threshold=phi+F(1,10000)*alpha*slope;bad=F(3,4);good=F(1,4);assert not bad<=threshold and good<=threshold;return dict(exact_Armijo_threshold=c.rec(threshold),rejected_merit=c.rec(bad),accepted_merit=c.rec(good),merit_is_NOT_physical_gate=True)
check('backtracking_descent_decision_keeps_rejected_trials',dict(no_weight_or_step_profile_sweep=True),armijo)

def coupled():
 rows=[record('xlo',['x'],{(1,):F(1)}),record('xhi',['x'],{(0,):F(1),(1,):F(-1)}),record('ylo',['y'],{(1,):F(1)}),record('yhi',['y'],{(0,):F(1),(1,):F(-1)}),record('curved',['x','y'],{(0,1):F(1),(2,0):F(-1)}),record('coupled_sum',['x','y'],{(0,0):F(2,5),(1,0):F(-1),(0,1):F(-1)}),record('xmin',['x'],{(1,):F(1),(0,):F(-1,10)}),record('ymin',['y'],{(1,):F(1),(0,):F(-1,10)}),record('strict_y',['y'],{(1,):F(1)},'>')];s=p.PolynomialSystem(rows,['x','y'],{'x':[F(0),F(1)],'y':[F(0),F(1)]},emit);initial=np.array([-.2,.8]);returned=p.restore(s,initial,lambda v:emit(dict(analytic_NEW_coupled_control=True,**v)));save('NEW-coupled-control-actual-TERMINAL-RETURN.json',returned);assert returned['status']=='PROPOSED_exact_saved_model_feasible';exact=s.exact_check(returned['actual_terminal_vector'],emit);assert exact['all_exact_model_inequalities_passed']and np.array_equal(initial,[-.2,.8]);return dict(terminal_return=returned,independent_exact_final_check=exact,analytic_values_NOT_episode_source_or_robot_anchors=True)
check('NEW_coupled_continuous_curved_feasibility_from_outside_domain',dict(no_old_control_or_episode_replay=True),coupled)
assert len(checks)==10;save('NEW-primal-contract-certificates.json',checks);J.close();report=dict(schema='v2d.primal-restoration-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_primal_contracts_passed=True,new_analytic_contracts=10,new_analytic_coupled_restoration_calls=1,new_episode_restoration_calls=0,helper_sha256=sha(S/'compute/primal_inequality_restoration_v1.py'),no_old_source_time_algebra_or_controls_replayed=True,no_original_candidate_gate_or_source_authority_change=True,new_native_position_checks=0,new_GPU_jobs=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',report);print(json.dumps({k:v for k,v in report.items()if k!='all_outputs_sha256'}))

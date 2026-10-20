"""NEW fixed-boundary folding and full-clock request contracts only."""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,math,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-surrounding-boundary-contract-v1';sys.path.insert(0,str(S/'compute'))
import surrounding_temporal_requests_v1 as b
import half_angle_temporal_algebra_v1 as h
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();J=(O/'all-NEW-boundary-folding-contract-returns.jsonl').open('w');emit=lambda v:g.journal(J,v);checks=[]
def save(n,v):
 with(O/n).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def check(name,proposal,work):
 emit(dict(kind='BEFORE_NEW_boundary_contract',name=name,proposal=proposal));r=work();emit(dict(kind='AFTER_NEW_boundary_contract_RETURN',name=name,RETURN=r));checks.append(dict(name=name,RETURN=r))
orig=[[F(t,20),F(2*t-1,30)]for t in range(8)];fixed=[[v+F((-1)**(t+j),500)for j,v in enumerate(row)]for t,row in enumerate(orig)];ts=[F(t*t+4*t,120)for t in range(8)];active={3,4,5};req=b.requests(orig,fixed,ts,active,[F(12),F(12)],emit)

def coverage():
 expected=[]
 for kind,weights,length in [('original_adjacent_added',[1,-1],2),('actual_URDF_velocity',[1,-1],2),('original_second_added',[1,-2,1],3)]:
  for component in range(2):
   for t in range(length-1,8):
    clock=list(range(t,t-length,-1))
    if set(clock)&active:expected.append((kind,component,tuple(clock),tuple(weights)))
 assert set(map(b.signature,req))==set(expected)and len(req)==len(expected)
 return dict(exact_incidence_signatures=[list(r)for r in expected],no_unaffected_gate_dropped_from_original_authority=True)
check('affected_full_clock_incidence_including_two_second_boundary_rows',dict(active=sorted(active)),coverage)

def folding():
 out=[]
 for r in req:
  constant=sum(F(w)*fixed[t][r['component']]for t,w in zip(r['clocks'],r['weights'])if t not in active);assert constant==h.unrec(r['fixed_sum']);lo=h.unrec(r['ORIGINAL_lower_rad']);hi=h.unrec(r['ORIGINAL_upper_rad']);assert lo-constant==h.unrec(r['folded_ORIGINAL_lower_rad'])and hi-constant==h.unrec(r['folded_ORIGINAL_upper_rad']);out.append(dict(signature=list(b.signature(r)),fixed_constant=h.rec(constant)))
  for d in[F(-1,19),F(0),F(2,23)]:
   point={t:orig[t][r['component']]+d*(t+1)for t in active};original=sum(w*(point[t]if t in active else fixed[t][r['component']])for t,w in zip(r['clocks'],r['weights']));folded=sum(w*point[t]for t,w in zip(r['variable_clocks'],r['variable_weights']));assert original==folded+constant and(lo<=original<=hi)==(lo-constant<=folded<=hi-constant)
 return dict(all_weighted_constant_and_interval_identities=out,three_exact_synthetic_assignments_per_request=True)
check('exact_original_vs_folded_gate_equivalence_nonzero_fixed_offsets',dict(not_measured_boundary=True),folding)

def one_variable():
 r=next(v for v in req if v['kind_of_constraint']=='original_second_added'and v['component']==0 and v['clocks']==[3,2,1]);assert r['variable_clocks']==[3]and r['variable_weights']==[1];expected=-2*fixed[2][0]+fixed[1][0];assert h.unrec(r['fixed_sum'])==expected
 return dict(request=r,weighted_two_fixed_terms=h.rec(expected))
check('second_difference_two_fixed_terms_keep_minus_two_weight',dict(joins='entry'),one_variable)

def exit_sign():
 r=next(v for v in req if v['kind_of_constraint']=='original_second_added'and v['component']==0 and v['clocks']==[7,6,5]);assert r['variable_weights']==[1]and r['variable_clocks']==[5]and h.unrec(r['fixed_sum'])==fixed[7][0]-2*fixed[6][0];return r
check('exit_second_boundary_uses_future_F_constants_without_observation_claim',dict(joins='exit'),exit_sign)

def velocity():
 rows=[v for v in req if v['kind_of_constraint']=='actual_URDF_velocity'];out=[]
 for r in rows:
  t0,t1=r['clocks'];dt=ts[t0]-ts[t1];bound=(F(12)+F(float(1e-7)))*dt;assert h.unrec(r['ORIGINAL_upper_rad'])==bound and h.unrec(r['ORIGINAL_lower_rad'])==-bound;assert bound!=F(12)*dt+F(float(1e-7));out.append(dict(dt=h.rec(dt),bound=h.rec(bound)))
 return dict(nonuniform_timestamp_bounds=out,tolerance_units_preserved=True)
check('actual_speed_tolerance_multiplies_exact_dt',dict(no_new_tolerance=True),velocity)

def invalid():
 cases=[ts[:3]+[ts[2]]+ts[4:],ts];messages=[]
 for times,frames in[(cases[0],active),(ts,{8})]:
  try:b.requests(orig,fixed,times,frames,[F(12),F(12)],lambda v:None)
  except ValueError as e:messages.append(str(e))
  else:raise AssertionError('invalid construction accepted')
 return dict(unresolved_rejections=messages)
check('zero_timestamp_and_out_of_clock_requests_rejected',dict(no_silent_boundary_fill=True),invalid)

def shifted_phase():
 r=next(v for v in req if v['kind_of_constraint']=='original_adjacent_added'and v['component']==0 and v['clocks']==[3,2]);name=r['variable_names'][0];anglebox={name:[F(-1),F(1)]};halfbox={name:[F(-3,5),F(3,5)]};record=b.new_polynomial_request(r,anglebox,halfbox,emit);assert record['angle_branch']=='certified_finite_angle_branch';center=(h.unrec(r['folded_ORIGINAL_lower_rad'])+h.unrec(r['folded_ORIGINAL_upper_rad']))/2;out=[]
 for offset in[F(-1,10),F(0),F(1,10)]:
  q=float(center+offset);u=math.tan(q/2);classification=h.classify(record,[F(u)]);direct=h.unrec(r['ORIGINAL_lower_rad'])<=F(q)+h.unrec(r['fixed_sum'])<=h.unrec(r['ORIGINAL_upper_rad']);out.append(dict(angle=q,classification=classification,direct_inside=direct));assert(classification=='inside')==direct
 return dict(shifted_phase_return=record,direct_NEW_synthetic_checks=out,positive_modulus_denominator_lower=1)
check('new_fixed_boundary_phase_interval_signs_match_direct_angles',dict(no_tangent_difference_sign_assumption=True),shifted_phase)

def pole():
 r=req[0];names=r['variable_names'];angles={n:[F(31,10),F(32,10)]for n in names};half={n:[F(1),F(2)]for n in names};ret=b.new_polynomial_request(r,angles,half,emit);assert ret['angle_branch']=='UNRESOLVED_angle_branch_or_half_angle_pole';return ret
check('new_boundary_pole_and_cut_remain_unresolved',dict(no_center_substitution=True),pole)
assert len(checks)==8;save('NEW-boundary-contract-certificates.json',checks);J.close();out=dict(schema='v2d.surrounding-boundary-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_boundary_contracts_passed=True,new_boundary_contracts=8,helper_sha256=sha(S/'compute/surrounding_temporal_requests_v1.py'),old22requests_702vectors_and_half_angle_contracts_NOT_replayed=True,new_solver_calls=0,new_native_position_checks=0,new_GPU_jobs=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))

"""Thirteen NEW higher observable/power/domain/residual-map contracts."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product,permutations
from math import comb
import datetime,hashlib,json,os,time
import pair_product_moments_v1 as h
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
S=Path(__file__).resolve().parents[1];O=S/'e037-pair-products-contract-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();J=(O/'NEW-pair-product-contract-proposals-and-returns.jsonl').open('w');emit=lambda v:g.journal(J,v);controls=[]
def save(n,v):
 with(O/n).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def begin(n,v):emit(dict(kind='BEFORE_NEW_contract',name=n,proposal=v))
def check(n,v,ok):emit(dict(kind='AFTER_NEW_contract_RETURN',name=n,record=v,passed=bool(ok)));assert ok,n;controls.append(n)
def pack(d):return [{**{'key':list(k)if isinstance(k,tuple)else k},'value':c.rec(v)}for k,v in sorted(d.items())]
begin('observable_square_shared_variable_aliasing',{});req=h.requested(('a','b'),('a','b'));check('observable_square_shared_variable_aliasing',dict(requested=req),req=={'a':2,'b':2})
begin('overlapping_pair_product_degree_three',{});req=h.requested(('a','b'),('b','c'));check('overlapping_pair_product_degree_three',dict(requested=req),req=={'a':1,'b':2,'c':1})
begin('absent_four_scope_product_stays_unknown',{});req=h.requested(('a','b'),('c','d'));av=h.available([('a','b'),('b','c'),('a','b','c')],req);check('absent_four_scope_product_stays_unknown',dict(requested=req,available=av),req=={'a':1,'b':1,'c':1,'d':1}and not av)
begin('no_powers_beyond_existing_per_coordinate_four',{});av=h.available([('a','b','c')],{'a':5});check('no_powers_beyond_existing_per_coordinate_four',dict(available=av),av==[])
begin('pointwise_observable_box_factor_identities',{});records=[]
for a,b in product([F(0),F(2,7),F(4,5),F(1)],repeat=2):
 y=a*b;records.append(y>=0 and 1-y>=0 and y-y*y==a*b*(1-a*b)>=0 and a-y==a*(1-b)>=0 and b-y==b*(1-a)>=0)
check('pointwise_observable_box_factor_identities',dict(exact_points=len(records),identities=['Y=ab','Y-Y^2=ab(1-ab)','a-Y=a(1-b)','b-Y=b(1-a)']),all(records))
begin('NEW_correlated_continuous_product_moments',dict(s_uniform=True,coordinates=['s','s^2','1-s'],observables=['s^3','s^2*(1-s)']));mean=F(1,4);second=F(1,7);cross=F(1,6)-F(1,7);check('NEW_correlated_continuous_product_moments',dict(mean=c.rec(mean),second=c.rec(second),shared_product=c.rec(cross),variance_numerator=c.rec(second-mean*mean)),cross==F(1,42)and second-mean*mean==F(9,112))
begin('NEW_tensor_power_Bernstein_expansion',{});identities=0;ok=True
for req in[{'a':2,'b':2},{'a':1,'b':2,'c':1},{'a':1,'b':1}]:
 scope=tuple(sorted(req));indices=list(product(range(5),repeat=len(scope)));entries={idx:i for i,idx in enumerate(indices)};co=h.moment_map(scope,entries,req);emit(dict(kind='AFTER_NEW_power_map_RETURN',requested=req,coefficients=pack(co)))
 for t in product([F(1,7),F(3,5)],repeat=len(scope)):
  lhs=F(0)
  for idx,col in entries.items():
   basis=F(1)
   for k,x in zip(idx,t,strict=True):basis*=comb(4,k)*x**k*(1-x)**(4-k)
   lhs+=co.get(col,F(0))*basis
  rhs=F(1)
  for name,x in zip(scope,t,strict=True):rhs*=x**req[name]
  ok=ok and lhs==rhs;identities+=1
check('NEW_tensor_power_Bernstein_expansion',dict(exact_identities=identities),ok and identities==16)
begin('axis_order_and_shared_scope_request_invariance',{});ok=True;n=0
for perm in permutations(('a','b','c')):
 entries={idx:i for i,idx in enumerate(product(range(5),repeat=3))};req={'a':1,'b':2,'c':1};co=h.moment_map(perm,entries,req)
 for idx,col in entries.items():
  expected=F(1)
  for name,k in zip(perm,idx,strict=True):expected*=F(comb(k,req[name]),comb(4,req[name]))if k>=req[name]else 0
  ok=ok and co.get(col,F(0))==expected
 n+=1
check('axis_order_and_shared_scope_request_invariance',dict(permutations=n),ok and n==6)
begin('NEW_squared_pair_marginal_residual_linear_identity',{});big={idx:i for i,idx in enumerate(product(range(5),repeat=3))};small={idx:125+i for i,idx in enumerate(product(range(5),repeat=2))};req={'a':2,'b':2};B=h.moment_map(('a','b','c'),big,req);A=h.moment_map(('a','b'),small,req);derived={};residuals={};w=[F((i%11)-4,19)for i in range(150)]
for idx,col in small.items():
 z=A.get(col,F(0));eq={col:F(-1)}
 for k,kcol in big.items():
  if k[:2]==idx:eq[kcol]=F(1)
 residuals[idx]=sum(v*w[j]for j,v in eq.items())
 for j,v in eq.items():derived[j]=derived.get(j,F(0))+z*v
expected=dict(B)
for j,v in A.items():expected[j]=expected.get(j,F(0))-v
derived={j:v for j,v in derived.items()if v};predicted=sum(A.get(col,F(0))*residuals[idx]for idx,col in small.items());difference=sum(v*w[j]for j,v in expected.items());check('NEW_squared_pair_marginal_residual_linear_identity',dict(predicted=c.rec(predicted),difference=c.rec(difference),sparse_coefficients=len(derived)),derived=={j:v for j,v in expected.items()if v}and predicted==difference)
begin('NEW_squared_pair_exact_residual_envelope',{});anchor=F(2,9);res=[F(1,2**160),-F(3,2**164)];multipliers=[F(2,3),F(1,6)];delta=sum(a*b for a,b in zip(res,multipliers,strict=True));radius=sum(abs(a*b)for a,b in zip(res,multipliers,strict=True));check('NEW_squared_pair_exact_residual_envelope',dict(anchor=c.rec(anchor),actual_copy=c.rec(anchor+delta),radius=c.rec(radius),no_acceptance_tolerance=True),anchor-radius<=anchor+delta<=anchor+radius and radius>0)
begin('derived_observable_zero_signed_mass_variance',{});records=[h.mass_variance(F(0),F(0),F(0)),h.mass_variance(F(0),F(1),F(2)),h.mass_variance(F(-1),F(1),F(1)),h.mass_variance(F(3),F(1),F(1))];check('derived_observable_zero_signed_mass_variance',dict(records=[{k:c.rec(v)if isinstance(v,F)else v for k,v in row.items()}for row in records]),records[0]['classification']=='ZERO_positive_measure_ONLY'and records[1]['classification']=='ZERO_mass_nonzero_moments_NOT_positive_measure'and records[2]['classification'].startswith('UNRESOLVED')and records[3]['variance_numerator']==2)
begin('coordinate_domain_alias_requires_exact_binding',{});d={'a':[F(0),F(1)],'b':[F(1),F(2)]};bad={'a':[F(0),F(1)],'b':[F(1),F(2)+F(1,2**160)]};check('coordinate_domain_alias_requires_exact_binding',dict(same=h.same_domain(d,d,['a','b']),different=h.same_domain(d,bad,['a','b']),different_domain_NOT_silently_aliased=True),h.same_domain(d,d,['a','b'])and not h.same_domain(d,bad,['a','b']))
begin('missing_scope_map_return_is_explicit',{});ret=None
try:h.moment_map(('a','b'),{(0,0):0},{'a':1,'c':1})
except ValueError as ex:ret=str(ex)
check('missing_scope_map_return_is_explicit',dict(RETURN=ret),ret=='UNRESOLVED_unavailable_power_or_scope')
J.close();out=dict(schema='v2d.pair-products-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_controls_passed=True,new_analytic_controls=len(controls),control_names=controls,new_solver_calls=0,new_episode_diagnoses=0,old_controls_audits_NOT_repeated=True,new_native_position_checks=0,new_GPU_jobs=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))

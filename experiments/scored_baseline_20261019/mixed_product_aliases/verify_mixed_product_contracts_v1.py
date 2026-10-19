"""Eleven NEW mixed-alias/power/zero-pivot/domain/pointer contracts."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product,combinations_with_replacement
from math import comb
import datetime,hashlib,json,os,time
import mixed_product_aliases_v1 as h
import pair_product_moments_v1 as old
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
S=Path(__file__).resolve().parents[1];O=S/'e037-mixed-products-contract-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();J=(O/'NEW-mixed-alias-contract-proposals-and-returns.jsonl').open('w');emit=lambda v:g.journal(J,v);controls=[]
def save(n,v):
 with(O/n).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def begin(n,v):emit(dict(kind='BEFORE_NEW_contract',name=n,proposal=v))
def check(n,v,ok):emit(dict(kind='AFTER_NEW_contract_RETURN',name=n,record=v,passed=bool(ok)));assert ok,n;controls.append(n)
begin('mixed_inside_pair_multiplicative_ideal',{});req=h.request('a',('a','b'));vals=[a*(a*b)==a*a*b for a,b in product([F(0),F(1,7),F(4,5),F(1)],repeat=2)];check('mixed_inside_pair_multiplicative_ideal',dict(requested=req,pointwise_identities=len(vals)),req=={'a':2,'b':1}and all(vals))
begin('mixed_outside_pair_triple_alias_ideal',{});first=h.request('a',('b','c'));second=h.request('c',('a','b'));vals=[a*(b*d)==d*(a*b)for a,b,d in product([F(0),F(2,7),F(1)],repeat=3)];check('mixed_outside_pair_triple_alias_ideal',dict(first=first,second=second,identities=len(vals)),first==second=={'a':1,'b':1,'c':1}and all(vals))
begin('NEW_mixed_power_tensor_expansions_ONLY',{});n=0;ok=True
for req in[{'a':2,'b':1},{'a':1,'b':1,'c':1}]:
 scope=tuple(sorted(req));entries={idx:i for i,idx in enumerate(product(range(5),repeat=len(scope)))};co=old.moment_map(scope,entries,req);emit(dict(kind='AFTER_NEW_mixed_power_map_RETURN',requested=req,coefficients=[dict(column=k,value=c.rec(v))for k,v in sorted(co.items())]))
 for t in product([F(1,5),F(4,7)],repeat=len(scope)):
  lhs=F(0)
  for idx,col in entries.items():
   basis=F(1)
   for k,x in zip(idx,t,strict=True):basis*=comb(4,k)*x**k*(1-x)**(4-k)
   lhs+=co.get(col,F(0))*basis
  rhs=F(1)
  for name,x in zip(scope,t,strict=True):rhs*=x**req[name]
  ok=ok and lhs==rhs;n+=1
check('NEW_mixed_power_tensor_expansions_ONLY',dict(identities=n),ok and n==12)
begin('NEW_continuous_correlated_mixed_moments',dict(s_uniform=True,coordinates=['s','s^3','1-s^2'],new_mixed_powers_ONLY=True));inside=F(1,6);outside=F(1,5)-F(1,7);check('NEW_continuous_correlated_mixed_moments',dict(inside=c.rec(inside),two_pointwise_aliases_have_same_outside_integral=c.rec(outside)),inside==F(1,6)and outside==F(2,35))
begin('mixed_alias_group_dedup_without_new_observations',{});requests=[h.request(v,pair)for v in['a','b','c']for pair in[('a','b'),('b','c')]];keys=[h.key(r)for r in requests];check('mixed_alias_group_dedup_without_new_observations',dict(requests=requests,groups=len(set(keys)),entries=len(keys)),len(keys)==6 and len(set(keys))==5 and keys.count(h.key({'a':1,'b':1,'c':1}))==2)
begin('unavailable_mixed_scope_remains_FREE',{});req=h.request('d',('a','b'));av=old.available([('a','b'),('a','b','c')],req);check('unavailable_mixed_scope_remains_FREE',dict(requested=req,available=av,no_product_of_means=True),av==[])
begin('NEW_two_one_marginal_row_identity',{});big={idx:i for i,idx in enumerate(product(range(5),repeat=3))};small={idx:125+i for i,idx in enumerate(product(range(5),repeat=2))};req={'a':2,'b':1};A=old.moment_map(('a','b'),small,req);B=old.moment_map(('a','b','c'),big,req);co={};w=[F((i%13)-5,23)for i in range(150)];predicted=F(0)
for idx,col in small.items():
 z=A.get(col,F(0));eq={col:F(-1)}
 for k,kcol in big.items():
  if k[:2]==idx:eq[kcol]=1
 predicted+=z*sum(v*w[j]for j,v in eq.items())
 for j,v in eq.items():co[j]=co.get(j,F(0))+z*v
expected=dict(B)
for j,v in A.items():expected[j]=expected.get(j,F(0))-v
co={j:v for j,v in co.items()if v};expected={j:v for j,v in expected.items()if v};difference=sum(z*w[j]for j,z in expected.items());check('NEW_two_one_marginal_row_identity',dict(predicted=c.rec(predicted),actual=c.rec(difference),nonzero_coefficients=len(co)),co==expected and predicted==difference)
begin('mixed_zero_variance_alias_NO_division',{});ret=h.mixed_zero_variance_case(F(2),F(4,5),F(8,25),F(1,2),F(1,5));check('mixed_zero_variance_alias_NO_division',{k:c.rec(v)if isinstance(v,F)else v for k,v in ret.items()},ret['classification']=='NECESSARY_zero_variance_mixed_alias_ONLY'and ret['mixed_residual']==0)
begin('zero_signed_mass_does_not_enable_mixed_alias_inference',{});rs=[h.mixed_zero_variance_case(m,F(1),F(1),F(1),F(0))for m in[F(0),F(-1)]];check('zero_signed_mass_does_not_enable_mixed_alias_inference',dict(returns=rs),all(r['classification']=='UNRESOLVED_zero_signed_mass_NO_normalization'for r in rs))
begin('mixed_affine_domain_mismatch_breaks_pointwise_alias',{});u=F(2,5);ta=u;other=(u-F(1,8))/(1-F(1,8));b=F(3,7);lhs=ta*(ta*b);rhs=ta*(other*b);check('mixed_affine_domain_mismatch_breaks_pointwise_alias',dict(first=c.rec(lhs),other_domain=c.rec(rhs),same_name_does_not_make_domains_equal=True),lhs!=rhs and not h.domains_equal({'a':[0,1]},{'a':[F(1,8),1]},['a']))
begin('new_partial24_pointer_embedding_preserves_old_blocks',{});cells={};overlap=0
for i,j in combinations_with_replacement(range(11),2):cells[i,j]='old11_pointer'
for i,j in combinations_with_replacement(range(14),2):
 key=(0 if i==0 else 10+i,0 if j==0 else 10+j)
 if key in cells:overlap+=1
 else:cells[key]='old14_pointer'
for i in range(10):
 for s in range(13):
  key=h.full_mixed_cell(i,s);assert key not in cells;cells[key]='NEW_mixed_pointer'
check('new_partial24_pointer_embedding_preserves_old_blocks',dict(cells=len(cells),only_constant_overlap=overlap,new_mixed=sum(v=='NEW_mixed_pointer'for v in cells.values()),no_values_or_completions_constructed=True),len(cells)==300 and overlap==1 and sum(v=='NEW_mixed_pointer'for v in cells.values())==130)
J.close();out=dict(schema='v2d.mixed-products-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_controls_passed=True,new_analytic_controls=len(controls),control_names=controls,old_pair_power_controls_NOT_repeated=True,old_controls_REUSED_BY_HASH={str(S/'e037-pair-products-contract-v1/report.json'):sha(S/'e037-pair-products-contract-v1/report.json')},new_solver_calls=0,new_episode_diagnoses=0,new_native_position_checks=0,new_GPU_jobs=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k not in['all_outputs_sha256','old_controls_REUSED_BY_HASH']}))

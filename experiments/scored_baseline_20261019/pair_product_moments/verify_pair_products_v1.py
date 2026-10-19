"""Independent higher-observable availability/falling-factorial/residual bindings."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product,combinations_with_replacement
from collections import Counter
import datetime,hashlib,json,os,time
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
S=Path(__file__).resolve().parents[1];O=S/'e037-pair-products-independent-v1';D=S/'e037-pair-products-diagnosis-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();J=(O/'independent-pair-product-proposals-and-returns.jsonl').open('w');emit=lambda v:g.journal(J,v)
def save(n,v):
 with(O/n).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def falling_weight(scope,idx,req):
 z=F(1)
 for name,k in zip(scope,idx,strict=True):
  for r in range(req.get(name,0)):z*=F(k-r,4-r)
 return z
def add(to,co,mult=F(1)):
 for k,v in co.items():to[k]=to.get(k,F(0))+mult*v
 return {k:v for k,v in to.items()if v}
er=read(D/'report.json');binding=read(D/'NEW-derived-observable-and-input-bindings.json');bindings=er['saved_input_bindings'];emit(dict(kind='BEFORE_NEW_independent_input_provenance',bindings=bindings));ok=all(sha(p)==v for p,v in bindings.items());emit(dict(kind='AFTER_NEW_independent_input_provenance_RETURN',passed=ok));assert ok
mp=S/'e037-bounded-moment-cone-v1/update-05-necessary-LP-model.json';model=read(mp);weights=list(map(F,read(S/'e037-image-conic-evaluation-v1/single-image-conic-solver-RETURN.json')['x']));scopes={}
for col,row in enumerate(model['columns']):scopes.setdefault(tuple(row['scope']),{})[tuple(row['index'])]=col
obs=sorted(s for s in scopes if len(s)==2);assert [r['variables']for r in binding['observables']]==[list(s)for s in obs]and len(obs)==13;domains=read(S/'e037-common-moment-model-v1/common-degree-factor-representations.json')['domains'];prior=read(S/'e037-global-cycle-diagnosis-v1/saved-global-moment-authority-bindings.json');assert domains==prior['domains_COPIED'];links={(tuple(r['big']),tuple(r['small'])):r['rows']for r in read(S/'e037-global-cycle-diagnosis-v1/saved-marginal-link-bindings.json')};jp=S/'e037-image-return-independent-v1/all-NEW-return-primal-matrices-and-conic-dual-certificates.jsonl';res={}
with jp.open()as f:
 for line in f:
  if'"kind":"AFTER_NEW_return_exact_scalar_constraint"'not in line:continue
  v=json.loads(line)
  if v['constraint_kind']=='equality':res[v['row']]=c.dec(v['residual'])
assert len(res)==549;known=unknown=copy_count=map_count=nonzero=0;maxdiff=F(0);certs=[];absent=[];entry_keys=[]
for i,j in combinations_with_replacement(range(14),2):
 emit(dict(kind='BEFORE_NEW_independent_entry',entry=[i,j]));row=read(D/f'partial-product-entry-{i:02d}-{j:02d}.json');req=dict(Counter((obs[i-1]if i else())+(obs[j-1]if j else())));assert row['requested']==req;entry_keys.append([i,j]);av=sorted([s for s in scopes if all(v in s and 0<=r<=4 for v,r in req.items())],key=lambda s:(len(s),s))if req else[]
 if i==j==0:
  passed=row['classification']=='SAVED_mass_target_and_copy_envelope_REUSED_BY_HASH'and row['saved_copy_envelope']==read(S/'e037-global-cycle-diagnosis-v1/original-unit-mass-and-actual-copy-envelope.json');known+=1
 elif not av:
  passed=row['classification']=='FREE_UNAVAILABLE_mixed_product'and row['missing_products_NOT_zero_or_mean_products'];unknown+=1;absent.append(dict(entry=[i,j],requested=req,all_existing_scopes_checked=len(scopes),classification='FREE_UNKNOWN_NOT_filled'))
 else:
  known+=1;anchor=av[0];amap={col:falling_weight(anchor,idx,req)for idx,col in scopes[anchor].items()};amap={k:v for k,v in amap.items()if v};anchorvalue=sum(v*weights[k]for k,v in amap.items());passed=row['anchor_scope']==list(anchor)and c.dec(row['anchor_value'])==anchorvalue and [r['scope']for r in row['copies']]==[list(s)for s in av];radius=F(0)
  for copy in row['copies']:
   scope=tuple(copy['scope']);emit(dict(kind='BEFORE_NEW_independent_copy_map',entry=[i,j],scope=list(scope),requested=req));co={col:falling_weight(scope,idx,req)for idx,col in scopes[scope].items()};co={k:v for k,v in co.items()if v};savedco={z['column']:c.dec(z['value'])for z in copy['coefficients']};actual=sum(v*weights[k]for k,v in co.items());combo={}
   for step in copy['path']:
    big,small=tuple(step['big']),tuple(step['small'])
    for link in links[big,small]:
     coeff=falling_weight(small,link['index'],req)
     if coeff:combo[link['equality_row']]=combo.get(link['equality_row'],F(0))+coeff
   savedcombo={z['row']:c.dec(z['value'])for z in copy['saved_equality_combination']};derived={};bound=F(0)
   for k,v in combo.items():
    eq=model['equalities'][k];derived=add(derived,{z['column']:c.dec(z['value'])for z in eq['coefficients']},v);bound+=v*c.dec(eq['bound'])
   difference=add(dict(co),amap,F(-1));predicted=sum(v*res[k]for k,v in combo.items());upper=sum(abs(v*res[k])for k,v in combo.items());diff=actual-anchorvalue;copyok=co==savedco and combo==savedcombo and derived==difference and bound==0 and predicted==diff==c.dec(copy['actual_copy_difference'])==c.dec(copy['predicted_saved_residual_combination'])and actual==c.dec(copy['actual_value'])and upper==c.dec(copy['copy_difference_absolute_upper'])and abs(diff)<=upper;ret=dict(entry=[i,j],scope=list(scope),requested=req,coefficient_count=len(co),residual_combination_rows=len(combo),exact_copy_difference=c.rec(diff),absolute_residual_bound=c.rec(upper),passed=copyok);emit(dict(kind='AFTER_NEW_independent_copy_map_RETURN',RETURN=ret));certs.append(ret);assert copyok;passed=passed and copyok;radius=max(radius,upper);copy_count+=1;map_count+=len(co);nonzero+=diff!=0;maxdiff=max(maxdiff,abs(diff))
  passed=passed and list(map(c.dec,row['diagnostic_interval']))==[anchorvalue-radius,anchorvalue+radius]and c.dec(row['exact_residual_radius'])==radius
 emit(dict(kind='AFTER_NEW_independent_entry_RETURN',entry=[i,j],available=bool(av),passed=passed));assert passed
assert known==33 and unknown==72 and copy_count==56 and nonzero==24 and map_count==2113 and maxdiff==c.dec(er['max_exact_copy_difference']);events=[json.loads(line)for line in(D/'ONE-new-pair-product-proposals-and-returns.jsonl').read_text().splitlines()];selected=[r for r in events if r['kind']in['BEFORE_NEW_partial_product_entry','AFTER_NEW_partial_product_entry_RETURN']];ordered=len(selected)==210 and all(selected[2*i]['matrix_entry']==selected[2*i+1]['RETURN']['matrix_entry']==entry_keys[i]for i in range(105));emit(dict(kind='AFTER_NEW_independent_journal_RETURN',all105entry_pairs_ordered=ordered));assert ordered and all(sha(p)==v for p,v in bindings.items())
save('all-56-independent-map-copy-residual-certificates.json',certs);save('all-72-independent-unavailable-product-certificates.json',absent);J.close();out=dict(schema='v2d.pair-products-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_independent_checks_passed=True,derived_observables=13,partial_Gram_dimension=14,upper_entry_availability_checks=105,known_entries=known,free_unavailable_entries=unknown,new_nonconstant_moment_groups=known-1,independent_copies=copy_count,independent_falling_factorial_map_coefficients=map_count,nonzero_copy_differences=nonzero,max_exact_copy_difference=c.rec(maxdiff),all105entry_BEFORE_AFTER_pairs_verified=True,all549saved_residual_VALUES_REUSED_by_hash_not_reaudited=True,no_unknown_moment_filled_or_raw_weight_repair=True,classification=er['classification'],no_global_completion_common_supported_measure_or_trajectory_qualified=True,new_solver_calls=0,new_PSD_audits_or_completions=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))

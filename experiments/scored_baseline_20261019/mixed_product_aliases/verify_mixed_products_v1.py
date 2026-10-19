"""Independent130mixed maps/aliases; old blocks only hash-pointer checks."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from itertools import product,combinations_with_replacement
import datetime,hashlib,json,os,time
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
S=Path(__file__).resolve().parents[1];O=S/'e037-mixed-products-independent-v1';D=S/'e037-mixed-products-diagnosis-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();J=(O/'independent-mixed-product-proposals-and-returns.jsonl').open('w');emit=lambda v:g.journal(J,v)
def save(n,v):
 with(O/n).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def fw(scope,idx,req):
 z=F(1)
 for name,k in zip(scope,idx,strict=True):
  for r in range(req.get(name,0)):z*=F(k-r,4-r)
 return z
def add(out,co,z=F(1)):
 for k,v in co.items():out[k]=out.get(k,F(0))+z*v
 return {k:v for k,v in out.items()if v}
er=read(D/'report.json');bindings=er['saved_input_bindings'];emit(dict(kind='BEFORE_NEW_independent_input_binding',bindings=bindings));passed=all(sha(p)==v for p,v in bindings.items());emit(dict(kind='AFTER_NEW_independent_input_binding_RETURN',passed=passed));assert passed;model=read(S/'e037-bounded-moment-cone-v1/update-05-necessary-LP-model.json');variables=model['variables'];weights=list(map(F,read(S/'e037-image-conic-evaluation-v1/single-image-conic-solver-RETURN.json')['x']));scopes={}
for col,r in enumerate(model['columns']):scopes.setdefault(tuple(r['scope']),{})[tuple(r['index'])]=col
obs=[tuple(r['variables'])for r in read(S/'e037-pair-products-diagnosis-v1/NEW-derived-observable-and-input-bindings.json')['observables']];domains=read(S/'e037-common-moment-model-v1/common-degree-factor-representations.json')['domains'];assert domains==read(S/'e037-global-cycle-diagnosis-v1/saved-global-moment-authority-bindings.json')['domains_COPIED'];links={(tuple(r['big']),tuple(r['small'])):r['rows']for r in read(S/'e037-global-cycle-diagnosis-v1/saved-marginal-link-bindings.json')};res={};jp=S/'e037-image-return-independent-v1/all-NEW-return-primal-matrices-and-conic-dual-certificates.jsonl'
with jp.open()as f:
 for line in f:
  if'"kind":"AFTER_NEW_return_exact_scalar_constraint"'not in line:continue
  r=json.loads(line)
  if r['constraint_kind']=='equality':res[r['row']]=c.dec(r['residual'])
assert len(res)==549;entries=read(D/'all-130-mixed-entry-alias-pointers.json');assert len(entries)==130;unique={};known=unknown=copies=mapcount=nonzero=0;maxdiff=F(0);certs=[];aliasgroups={}
for cursor,(a,s)in enumerate(product(range(10),range(13))):
 entry=entries[cursor];req=dict(Counter((variables[a],)+obs[s]));key=tuple(sorted(req.items()));cell=[a+1,s+11];emit(dict(kind='BEFORE_NEW_independent_mixed_entry',matrix_entry=cell,requested=req));available=sorted([scope for scope in scopes if set(req)<=set(scope)and all(0<=v<=4 for v in req.values())],key=lambda t:(len(t),t));p=D/entry['group_file'];group=read(p);entryok=entry['matrix_entry']==cell and entry['requested']==req and group['requested']==req and sha(p)==entry['group_sha256'];aliasgroups.setdefault(key,[]).append(cell)
 if available:known+=1
 else:unknown+=1
 if key in unique:entryok=entryok and unique[key]==entry['group_file']
 else:
  unique[key]=entry['group_file']
  if not available:entryok=entryok and group['classification']=='FREE_UNAVAILABLE_scope_product'and group['copies']==[]
  else:
   anchor=available[0];A={col:fw(anchor,idx,req)for idx,col in scopes[anchor].items()};A={k:v for k,v in A.items()if v};anchorvalue=sum(v*weights[k]for k,v in A.items());radius=F(0);entryok=entryok and group['anchor_scope']==list(anchor)and c.dec(group['anchor_value'])==anchorvalue and [r['scope']for r in group['copies']]==[list(t)for t in available]
   for copy in group['copies']:
    scope=tuple(copy['scope']);emit(dict(kind='BEFORE_NEW_independent_mixed_copy',requested=req,scope=list(scope)));co={col:fw(scope,idx,req)for idx,col in scopes[scope].items()};co={k:v for k,v in co.items()if v};value=sum(v*weights[k]for k,v in co.items());combo={}
    for step in copy['path']:
     big,small=tuple(step['big']),tuple(step['small'])
     for link in links[big,small]:
      z=fw(small,link['index'],req)
      if z:combo[link['equality_row']]=combo.get(link['equality_row'],F(0))+z
    derived={};bound=F(0)
    for k,z in combo.items():
     eq=model['equalities'][k];derived=add(derived,{r['column']:c.dec(r['value'])for r in eq['coefficients']},z);bound+=z*c.dec(eq['bound'])
    predicted=sum(z*res[k]for k,z in combo.items());upper=sum(abs(z*res[k])for k,z in combo.items());diff=value-anchorvalue;ok=co=={r['column']:c.dec(r['value'])for r in copy['coefficients']}and combo=={r['row']:c.dec(r['value'])for r in copy['saved_residual_combination']}and derived==add(dict(co),A,F(-1))and bound==0 and predicted==diff==c.dec(copy['actual_difference'])==c.dec(copy['predicted_difference'])and value==c.dec(copy['actual_value'])and upper==c.dec(copy['absolute_residual_bound'])and abs(diff)<=upper;ret=dict(requested=req,scope=list(scope),coefficient_count=len(co),actual_difference=c.rec(diff),exact_residual_bound=c.rec(upper),passed=ok);emit(dict(kind='AFTER_NEW_independent_mixed_copy_RETURN',RETURN=ret));certs.append(ret);assert ok;entryok=entryok and ok;copies+=1;mapcount+=len(co);nonzero+=diff!=0;maxdiff=max(maxdiff,abs(diff));radius=max(radius,upper)
   entryok=entryok and list(map(c.dec,group['diagnostic_interval']))==[anchorvalue-radius,anchorvalue+radius]and c.dec(group['radius'])==radius
 emit(dict(kind='AFTER_NEW_independent_mixed_entry_RETURN',matrix_entry=cell,group_file=entry['group_file'],available=bool(available),passed=entryok));assert entryok
# Check metadata/hash pointers, never reevaluate a completed old block.
old=read(D/'old11-and-old14-block-pointers-BY-HASH.json');combined=read(D/'partial24Gram-pointer-inventory.json');oldcells={tuple(r['matrix_entry']):r for r in old};cells={tuple(r['matrix_entry']):r for r in combined['cells']};emit(dict(kind='BEFORE_NEW_independent_old_pointer_embedding',old_block_entries=len(old),full_cells=len(cells)));pointerok=len(oldcells)==170 and len(cells)==300 and set(cells)==set(combinations_with_replacement(range(24),2))and all(cells[k]==v for k,v in oldcells.items());hashchecks=0
for row in old:
 if row['source']is not None:pointerok=pointerok and sha(row['source'])==row['sha256'];hashchecks+=1
for entry in entries:
 record=cells[tuple(entry['matrix_entry'])];pointerok=pointerok and record==dict(entry,block='NEW130mixed_entries')
emit(dict(kind='AFTER_NEW_independent_old_pointer_embedding_RETURN',old_numerical_blocks_NOT_evaluated=True,pointer_hashes_checked=hashchecks,passed=pointerok));assert pointerok;specified=sum(r['classification'].startswith('SPECIFIED')for r in cells.values());free=300-specified;availableunique=sum(read(D/p)['classification'].startswith('AVAILABLE')for p in unique.values());aliases=[dict(requested=dict(k),entries=v)for k,v in aliasgroups.items()if len(v)>1];assert known==38 and unknown==92 and len(unique)==108 and availableunique==32 and copies==56 and mapcount==2136 and nonzero==24 and maxdiff==c.dec(er['max_exact_copy_difference'])and specified==110 and free==190
journal=[json.loads(line)for line in(D/'ONE-mixed-product-proposals-and-returns.jsonl').read_text().splitlines()];pairs=[r for r in journal if r['kind']in['BEFORE_NEW_mixed_entry','AFTER_NEW_mixed_entry_RETURN']];ordered=len(pairs)==260 and all(list(pairs[2*i]['matrix_entry'])==pairs[2*i+1]['RETURN']['matrix_entry']==entries[i]['matrix_entry']for i in range(130));emit(dict(kind='AFTER_NEW_independent_journal_RETURN',all130pairs_ordered=ordered));assert ordered and all(sha(p)==v for p,v in bindings.items())
save('independent-56-new-map-copy-residual-certificates.json',certs);save('independent-22-duplicate-monomial-alias-groups.json',aliases);J.close();out=dict(schema='v2d.mixed-products-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_independent_checks_passed=True,mixed_entry_checks=130,available_mixed_entries=known,FREE_mixed_entries=unknown,unique_monomial_groups=len(unique),available_unique_groups=availableunique,new_copies=copies,new_falling_factorial_map_coefficients=mapcount,nonzero_copy_differences=nonzero,max_exact_copy_difference=c.rec(maxdiff),duplicate_alias_groups=len(aliases),available_alias_duplicate_entries=6,old_upper_entries_REUSED_BY_HASH=170,old_pointer_hash_bindings=hashchecks,partial_Gram_dimension=24,total_upper_entries=300,specified_upper_entries=specified,FREE_upper_entries=free,all130durable_entry_pairs_verified=True,all549old_residual_VALUES_REUSED_BY_HASH=True,no_old_numerical_block_replayed=True,classification=er['classification'],no_global_completion_measure_source_truth_or_trajectory_qualified=True,new_solver_calls=0,new_PSD_audits_or_completions=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))

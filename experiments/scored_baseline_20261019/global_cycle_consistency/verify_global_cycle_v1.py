"""Independent NEW scope/tree/copy-row/Schur certificates; no producer replay."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations,product
import datetime,hashlib,json,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-global-cycle-independent-v1';P=S/'e037-global-cycle-diagnosis-v1';sys.path.insert(0,str(S/'compute'))
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());j=(O/'independent-NEW-global-cycle-tree-row-and-Schur-certificates.jsonl').open('w');emit=lambda v:g.journal(j,v);start=time.monotonic()
def decI(a):return list(map(c.dec,a))
def plus(a,b):return [a[0]+b[0],a[1]+b[1]]
def minus(a,b):return [a[0]-b[1],a[1]-b[0]]
def times(a,b):
 vals=[x*y for x in a for y in b];return [min(vals),max(vals)]
def sq(a):return [F(0)if a[0]<=0<=a[1]else min(x*x for x in a),max(x*x for x in a)]
def quotient(a,b):
 assert b[0]>0;return times(a,[1/b[1],1/b[0]])
def co(v):return {z['column']:c.dec(z['value'])for z in v}
def add(a,b,scale=F(1)):
 out=dict(a)
 for k,v in b.items():out[k]=out.get(k,F(0))+scale*v
 return {k:v for k,v in out.items()if v}
def falling(k,p):
 n=d=1
 for i in range(p):n*=k-i;d*=4-i
 return F(n,d)
def mono(scope,entries,requested):
 out={}
 for idx,column in entries.items():
  value=F(1)
  for name,k in zip(scope,idx):value*=falling(k,requested.get(name,0))
  if value:out[column]=value
 return out
report=read(P/'report.json')
for n,d in report['all_outputs_sha256'].items():assert sha(P/n)==d
model=read(S/'e037-bounded-moment-cone-v1/update-05-necessary-LP-model.json');weights=list(map(F,read(S/'e037-image-conic-evaluation-v1/single-image-conic-solver-RETURN.json')['x']));variables=model['variables'];tables={}
for col,v in enumerate(model['columns']):tables.setdefault(tuple(v['scope']),{})[tuple(v['index'])]=col
oldjournal=S/'e037-image-return-independent-v1/all-NEW-return-primal-matrices-and-conic-dual-certificates.jsonl';assert sha(oldjournal)==report['binding']['saved_equality_journal_sha256'];residuals={}
with oldjournal.open()as f:
 for line in f:
  if '"kind":"AFTER_NEW_return_exact_scalar_constraint"'in line:
   r=json.loads(line)
   if r['constraint_kind']=='equality':residuals[r['row']]=c.dec(r['residual'])
structure=read(P/'factor-scope-incidence-and-unavailable-global-products.json');maximal=list(map(tuple,structure['maximal_scopes']));assert all(s in tables for s in maximal)and set(maximal)=={s for s in tables if not any(set(s)<set(t)for t in tables)};edges=structure['all_scope_intersections'];tree=structure['maximum_weight_tree'];assert len(tree)==len(maximal)-1;edge_by={tuple(sorted((r['i'],r['j']))):r for r in edges};assert len(edge_by)==len(maximal)*(len(maximal)-1)//2;adj={i:[]for i in range(len(maximal))}
for r in edges:assert r['intersection']==sorted(set(maximal[r['i']])&set(maximal[r['j']]))and r['weight']==len(r['intersection'])
for r in tree:
 i,k=r['i'],r['j'];assert r==edge_by[tuple(sorted((i,k)))];adj[i].append((k,r['weight']));adj[k].append((i,r['weight']))
def path(a,b):
 queue=[(a,[])];seen={a}
 for i,values in queue:
  if i==b:return values
  for k,w in adj[i]:
   if k not in seen:seen.add(k);queue.append((k,values+[w]))
 raise AssertionError('disconnected tree')
checks=0
for (i,k),r in edge_by.items():assert r['weight']<=min(path(i,k));checks+=1
required=sum(sum(v in s for s in maximal)-1 for v in variables);actual=sum(r['weight']for r in tree);assert actual==structure['maximum_intersection_weight']and required==structure['running_intersection_required_weight']and required-actual==structure['running_intersection_deficit']and(actual==required)==structure['standard_junction_tree_exists'];emit(dict(kind='AFTER_NEW_independent_maximum_tree_certificate',tree_edges=len(tree),all_cycle_property_checks=checks,weight=actual,required=required,deficit=required-actual,no_global_measure_exclusion_from_RIP_failure=True))
known={tuple(sorted((a,b)))for scope in tables for a,b in combinations(scope,2)};unknown=set(combinations(sorted(variables),2))-known;assert known==set(map(tuple,structure['known_variable_pairs']))and unknown==set(map(tuple,structure['unknown_variable_pairs']))
def canonical(cycle):
 seq=[]
 for z in [tuple(cycle),tuple(reversed(cycle))]:
  for i in range(4):seq.append(z[i:]+z[:i])
 return min(seq)
expected_cycles=set()
for (a,d),(b,e)in combinations(sorted(unknown),2):
 if len({a,b,d,e})==4 and all(tuple(sorted((x,y)))in known for x in [a,d]for y in [b,e]):expected_cycles.add(canonical((a,b,d,e)))
assert expected_cycles==set(map(tuple,structure['induced_four_variable_cycles']));emit(dict(kind='AFTER_NEW_independent_cycle_availability_RETURN',known_pairs=len(known),missing_pairs=len(unknown),cycles=len(expected_cycles),opposite_pair_enumeration_NOT_old_candidate_graph=True))
masses=read(P/'unnormalized-actual-mass-bindings.json');max_mass_res=F(0)
for r in masses:
 scope=tuple(r['scope']);expected={col:F(1)for col in tables[scope].values()};row=model['equalities'][r['saved_normalization_row']];value=sum(weights[k]for k in expected);assert co(r['coefficients'])==co(row['coefficients'])==expected and c.dec(row['bound'])==1 and value==c.dec(r['actual_mass'])and value-1==c.dec(r['actual_residual'])==c.dec(r['saved_residual'])==residuals[r['saved_normalization_row']];max_mass_res=max(max_mass_res,abs(value-1))
mass_record=read(P/'original-unit-mass-and-actual-copy-envelope.json');mass=decI(mass_record['diagnostic_common_mass_interval']);assert mass==[1-max_mass_res,1+max_mass_res]and c.dec(mass_record['residual_radius'])==max_mass_res;copy_count=coef_count=nonzero_differences=0;max_difference=F(0);moment_intervals={}
availability=read(P/'partial-global-first-second-moment-availability.json');assert availability['known_upper_triangle_entries']==40 and availability['total_upper_triangle_entries']==66 and set(map(tuple,availability['unknown_mixed_pairs']))==unknown
for pointer in availability['moment_records']:
 r=read(P/pointer['file']);assert sha(P/pointer['file'])==pointer['sha256'];requested=r['requested'];anchor=tuple(r['anchor_scope']);expected_anchor=mono(anchor,tables[anchor],requested);anchor_value=sum(v*weights[k]for k,v in expected_anchor.items());assert expected_anchor==co(r['anchor_coefficients'])and anchor_value==c.dec(r['anchor_value']);radius=F(0)
 for z in r['copies']:
  scope=tuple(z['scope']);mapped=mono(scope,tables[scope],requested);assert mapped==co(z['coefficients']);value=sum(v*weights[k]for k,v in mapped.items());derived={};bound=F(0);predicted=F(0);absolute=F(0)
  for term in z['saved_equality_combination']:
   row=model['equalities'][term['row']];w=c.dec(term['value']);derived=add(derived,co(row['coefficients']),w);bound+=w*c.dec(row['bound']);predicted+=w*residuals[term['row']];absolute+=abs(w*residuals[term['row']])
  assert derived==add(mapped,expected_anchor,F(-1))and bound==0 and value==c.dec(z['actual_value'])and predicted==value-anchor_value==c.dec(z['actual_difference'])==c.dec(z['predicted_saved_residual_combination'])and absolute==c.dec(z['copy_difference_absolute_upper'])and abs(predicted)<=absolute;radius=max(radius,absolute);copy_count+=1;coef_count+=len(mapped);nonzero_differences+=predicted!=0;max_difference=max(max_difference,abs(predicted))
 assert radius==c.dec(r['exact_residual_envelope_radius'])and decI(r['diagnostic_interval'])==[anchor_value-radius,anchor_value+radius];moment_intervals[tuple(sorted(requested.items()))]=decI(r['diagnostic_interval']);emit(dict(kind='AFTER_NEW_independent_global_copy_map_RETURN',moment=r['index'],copies=len(r['copies']),radius=c.rec(radius),anchor_value=c.rec(anchor_value),no_weight_or_residual_repair=True))
# Exact generic symbolic identity B*det3=(AB-P^2)(BC-Q^2)-(BR-PQ)^2.
def symmul(p,q):
 out={}
 for a,v in p.items():
  for b,w in q.items():
   k=tuple(x+y for x,y in zip(a,b));out[k]=out.get(k,0)+v*w
 return {k:v for k,v in out.items()if v}
def symadd(a,b,z=1):
 out=dict(a)
 for k,v in b.items():out[k]=out.get(k,0)+z*v
 return {k:v for k,v in out.items()if v}
A,B,C,Pv,Q,R=[{tuple(int(i==k)for i in range(6)):1}for k in range(6)];AB=symmul(A,B);BC=symmul(B,C);PQ=symmul(Pv,Q);right=symadd(symmul(symadd(AB,symmul(Pv,Pv),-1),symadd(BC,symmul(Q,Q),-1)),symmul(symadd(symmul(B,R),PQ,-1),symadd(symmul(B,R),PQ,-1)),-1);det=symadd(symadd(symadd(symmul(symmul(A,B),C),symmul(A,symmul(Q,Q)),-1),symmul(C,symmul(Pv,Pv)),-1),symmul(symmul(Pv,Q),R),2);det=symadd(det,symmul(B,symmul(R,R)),-1);assert right==symmul(B,det);emit(dict(kind='AFTER_NEW_exact_generic_Schur_identity_RETURN',coefficient_identity=[dict(exponent=k,value=v)for k,v in sorted(right.items())]))
root_checks=triangle_checks=cycle_count=0;classes={}
for row in read(P/'all-four-variable-cycle-results.json'):
 r=read(P/row['record_file']);cycle=r['cycle'];first=[moment_intervals[((v,1),)]for v in cycle];second=[moment_intervals[((v,2),)]for v in cycle];cross=[moment_intervals[tuple(sorted([(cycle[i],1),(cycle[(i+1)%4],1)]))]for i in range(4)];assert first==list(map(decI,r['first_intervals']))and second==list(map(decI,r['second_intervals']))and cross==list(map(decI,r['cross_intervals']))and mass==decI(r['mass_interval']);V=[minus(times(mass,s),sq(f))for f,s in zip(first,second)];E=[minus(times(mass,cross[i]),times(first[i],first[(i+1)%4]))for i in range(4)];cov=r['covariance'];assert cov['classification']=='positive_unnormalized_covariance_necessities'and mass[0]>0 and V==list(map(decI,cov['variance_intervals']))and E==list(map(decI,cov['edge_intervals']));result=r['result'];intervals=[]
 for saved,vals in [(result['first_triangle'],[V[0],V[1],V[2],E[0],E[1]]),(result['second_triangle'],[V[0],V[3],V[2],E[3],E[2]])]:
  assert [decI(saved[k])for k in ['A','B','C','P','Q']]==vals;av,bv,cv,pv,qv=vals;assert bv[0]>0;center=quotient(times(pv,qv),bv);left=minus(av,quotient(sq(pv),bv));rr=minus(cv,quotient(sq(qv),bv));rad=max(F(0),left[1])*max(F(0),rr[1]);lo,hi=decI(saved['sqrt_bounds']);assert 0<=lo<=hi and lo*lo<=rad<=hi*hi and hi-lo<=F(1,2**96)and rad==c.dec(saved['radicand'])and center==decI(saved['center'])and left==decI(saved['left'])and rr==decI(saved['right']);interval=[center[0]-hi,center[1]+hi];assert interval==decI(saved['interval']);intervals.append(interval);root_checks+=1;triangle_checks+=1
 intersection=[max(z[0]for z in intervals),min(z[1]for z in intervals)];gap=max(F(0),intersection[0]-intersection[1]);assert intersection==decI(result['intersection'])and gap==c.dec(result['strict_separation']);expected='EXCLUDED_cycle_covariance_envelope'if gap>0 else'NECESSARY_cycle_bounds_NOT_EXCLUDED';assert result['classification']==row['classification']==expected;classes[expected]=classes.get(expected,0)+1;cycle_count+=1;emit(dict(kind='AFTER_NEW_independent_cycle_Schur_certificate_RETURN',cycle=cycle,intersection=[c.rec(z)for z in intersection],gap=c.rec(gap),classification=expected,both_missing_entries_remain_free=True))
j.close();out=dict(schema='v2d.global-cycle-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),all_independent_checks_passed=True,existing_scopes=len(tables),maximal_scopes=len(maximal),tree_edges=len(tree),maximum_tree_optimality_cycle_checks=checks,max_tree_weight=actual,running_intersection_target=required,running_intersection_deficit=required-actual,known_global_upper_entries=40,missing_global_mixed_entries=26,mass_row_identities=len(masses),moment_groups=len(moment_intervals),actual_moment_copy_identities=copy_count,exact_copy_map_coefficients=coef_count,nonzero_moment_copy_differences=nonzero_differences,max_actual_copy_difference=c.rec(max_difference),symbolic_Schur_identity_verified=True,cycle_count=cycle_count,triangle_certificates=triangle_checks,rational_sqrt_certificates=root_checks,cycle_classifications=classes,raw_AlmostSolved_status_weights_and_shared_residuals_UNCHANGED=True,diagnostic_copy_envelopes_NOT_candidate_tolerances_or_family_bounds=True,global_missing_moments_longer_cycles_and_supported_measure_UNRESOLVED=True,classification='seven_cycle_necessities_clear_but_no_global_measure_or_robot_trajectory_qualified',new_solver_calls=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
with (O/'report.json').open('w')as f:f.write(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))

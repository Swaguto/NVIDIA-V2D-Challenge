"""Independent raw auxiliary RETURN, separate exact completion and dual proof.

No solver/source/old table audit rerun. The exact certificate is an existential
diagnostic Gram matrix, never an alteration of any actual returned vector.
"""
from pathlib import Path
from fractions import Fraction as F
from math import sqrt
import datetime,hashlib,json,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-global-completion-independent-v1';P=S/'e037-global-completion-model-v1';E=S/'e037-global-completion-evaluation-v1';sys.path.insert(0,str(S/'compute'))
import mixed_joint_moments_v1 as m
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());j=(O/'all-NEW-independent-completion-return-exact-matrix-and-dual-identities.jsonl').open('w');emit=lambda v:g.journal(j,v);start=time.monotonic()
def co(v):return {z['column']:c.dec(z['value'])for z in v}
def matrix(x):return [[x[index[min(i,k),max(i,k)]]for k in range(11)]for i in range(11)]
def verify_positive(M,cert,label):
 assert cert['classification']=='exact_mixed_matrix_PSD';B=[list(map(c.dec,z))for z in cert['basis_columns']];D=list(map(c.dec,cert['diagonal']));assert len(B)==len(D)==11 and all(z>=0 for z in D)
 for i in range(11):assert B[i][i]==1 and all(z==0 for z in B[i][i+1:])
 for i in range(11):
  for k in range(11):assert sum(B[i][a]*M[a][b]*B[k][b]for a in range(11)for b in range(11))==(D[i]if i==k else 0)
 emit(dict(kind='AFTER_NEW_independent_completion_positive_congruence',label=label,exact_entries=121,rank=sum(z>0 for z in D),diagonal=cert['diagonal']));return sum(z>0 for z in D)
def constraints(x,label):
 eq=[];ine=[];bad=[]
 for kind,rows in [('equality',model['equalities']),('inequality',model['inequalities'])]:
  for i,row in enumerate(rows):
   residual=sum(c.dec(v['value'])*x[v['column']]for v in row['coefficients'])-c.dec(row['bound']);emit(dict(kind='AFTER_NEW_independent_completion_constraint',label=label,constraint_kind=kind,row=i,residual=c.rec(residual)))
   (eq if kind=='equality'else ine).append(residual)
   if(residual!=0 if kind=='equality'else residual>0):bad.append(dict(kind=kind,row=i,residual=c.rec(residual)))
 boxes=[dict(column=i,value=c.rec(v))for i,v in enumerate(x)if not 0<=v<=1];return dict(exact_equality_residual=c.rec(max(map(abs,eq))),exact_inequality_violation=c.rec(max([F(0),*ine])),violated_constraints=bad,unitbox_failures=boxes)
report=read(E/'report.json')
for n,d in report['all_outputs_sha256'].items():assert sha(E/n)==d
binding=read(P/'completion-model-binding.json');assert sha(P/'completion-model-binding.json')==report['model_binding_sha256'];model=read(P/'separate-global-completion-model.json');assert sha(P/'separate-global-completion-model.json')==binding['separate_model_sha256'];assert sha(S/'e037-bounded-moment-cone-v1/update-05-necessary-LP-model.json')==binding['original_source_model_sha256']and sha(S/'e037-image-conic-evaluation-v1/single-image-conic-solver-RETURN.json')==binding['original1125_RETURN_sha256'];layout=[(i,k)for k in range(11)for i in range(k+1)];index={p:i for i,p in enumerate(layout)};raw=read(E/'single-global-completion-solver-RETURN.json');x=list(map(F,raw['x']));z=list(map(F,raw['z']));assert len(x)==66 and len(z)==432;emit(dict(kind='BEFORE_NEW_independent_completion_RETURN_audit',actual_status=raw['status'],actual_RETURN_sha256=sha(E/'single-global-completion-solver-RETURN.json'),no_values_clipped_normalized_or_repaired=True));raw_constraints=constraints(x,'ACTUAL_auxiliary_RETURN');rawM=matrix(x);rawcert=m.congruence_certificate(rawM,emit);rawrank=None
if rawcert['classification']=='exact_mixed_matrix_PSD':rawrank=verify_positive(rawM,rawcert,'ACTUAL_auxiliary_RETURN')
else:assert sum(a*rawM[i][k]*b for i,a in enumerate(map(c.dec,rawcert['vector']))for k,b in enumerate(map(c.dec,rawcert['vector'])))==c.dec(rawcert['expectation'])<0
proposal=read(E/'ONE-separate-existential-exact-completion-proposal.json');values=list(map(c.dec,proposal['values']));assert len(values)==66 and values[index[0,0]]==1;known={tuple(v['entry']):v for v in model['known_intervals']};free_entries=0;known_entries=0
for pair,col in index.items():
 if pair==(0,0):continue
 if pair in known:
  r=known[pair];assert c.dec(r['interval'][0])<=values[col]<=c.dec(r['interval'][1])and values[col]==c.dec(r['anchor']);known_entries+=1
 else:assert values[col]==x[col]and model['columns'][col]['role']=='free_existential_mixed_completion';free_entries+=1
assert known_entries==39 and free_entries==26;proposal_constraints=constraints(values,'SEPARATE_exact_diagnostic_certificate');M=matrix(values);savedM=read(E/'separate-existential-completion-matrix.json');assert M==[list(map(c.dec,row))for row in savedM['matrix']];cert=read(E/'separate-exact-completion-PSD-certificate.json');proposal_rank=None
if cert['classification']=='exact_mixed_matrix_PSD':proposal_rank=verify_positive(M,cert,'SEPARATE_exact_diagnostic_certificate')
else:assert sum(a*M[i][k]*b for i,a in enumerate(map(c.dec,cert['vector']))for k,b in enumerate(map(c.dec,cert['vector'])))==c.dec(cert['expectation'])<0
exact_proposal=not proposal_constraints['violated_constraints']and not proposal_constraints['unitbox_failures']and proposal_rank is not None;assert exact_proposal==(report['single_separate_exact_proposal_status']=='EXACT_existential_Gram_completion_of_DIAGNOSTIC_envelope_ONLY');emit(dict(kind='AFTER_NEW_independent_exact_completion_VERDICT',qualified_ONLY_as_existential_Gram_completion_of_diagnostic_envelope=exact_proposal,rank=proposal_rank,known_interval_entries=39,free_unobserved_entries=26,actual_values_NOT_repaired=True))
wrapped=read(E/'completion-dual-certificate-scope.json');mathematical=read(E/'raw-generic-mathematical-dual-RETURN.json');assert sha(E/'raw-generic-mathematical-dual-RETURN.json')==wrapped['mathematical_RETURN_sha256']and wrapped['source_time_family_excluded']is False;dual_verified=False;gap=None;Gram_entries=0
if mathematical['classification']in['EXACT_frozen_supported_model_exclusion','UNRESOLVED_no_strict_supported_conic_gap']:
 alpha=list(map(c.dec,mathematical['alpha']));beta=list(map(c.dec,mathematical['beta']));assert beta==z[:1]and alpha==z[1:234]and all(v>=0 for v in alpha);residual=[F(0)]*66;rhs=F(0)
 for rows,multipliers in [(model['equalities'],beta),(model['inequalities'],alpha)]:
  for row,w in zip(rows,multipliers,strict=True):
   rhs+=w*c.dec(row['bound'])
   for v in row['coefficients']:residual[v['column']]+=w*c.dec(v['value'])
 certs=mathematical['PSD_Gram_certificates'];assert len(certs)==1;d=certs[0];B=[list(map(c.dec,row))for row in d['Gram_factor']];G=[list(map(c.dec,row))for row in d['certificate_matrix']];recover=[[F(0)for _ in range(11)]for _ in range(11)]
 for value,(i,k)in zip(raw['z'][366:432],layout,strict=True):recover[i][k]=recover[k][i]=F(float(value)/(1 if i==k else sqrt(2)))
 assert recover==[list(map(c.dec,row))for row in d['raw_dual_matrix']]
 for i in range(11):
  for k in range(11):assert G[i][k]==sum(B[i][a]*B[k][a]for a in range(11));Gram_entries+=1;residual[index[min(i,k),max(i,k)]]-=G[i][k]
 minimum=sum(min(F(0),v)for v in residual);gap=minimum-rhs;assert residual==list(map(c.dec,mathematical['exact_residual']))and minimum==c.dec(mathematical['finite_unit_box_minimum'])and rhs==c.dec(mathematical['RHS'])and gap==c.dec(mathematical['strict_gap'])==c.dec(wrapped['strict_gap']);assert(gap>0)==(wrapped['classification']=='EXACT_diagnostic_envelope_exclusion');dual_verified=True;emit(dict(kind='AFTER_NEW_independent_diagnostic_dual_RETURN',Gram_entries=121,RHS=c.rec(rhs),finite_unitbox_minimum=c.rec(minimum),strict_gap=c.rec(gap),applies_ONLY_diagnostic_envelope_NOT_original_source_family=True))
else:assert mathematical['classification']=='UNRESOLVED_negative_scalar_dual'and any(v<0 for v in z[1:234])
assert sha(S/'e037-image-conic-evaluation-v1/single-image-conic-solver-RETURN.json')==binding['original1125_RETURN_sha256'];j.close();out=dict(schema='v2d.global-completion-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),all_independent_checks_passed=True,actual_solver_status=raw['status'],actual_solver_iterations=raw['iterations'],actual_auxiliary_constraints=raw_constraints,actual_auxiliary_Gram_rank=rawrank,separate_exact_certificate_qualified=exact_proposal,separate_exact_PSD_rank=proposal_rank,separate_exact_constraints=proposal_constraints,known_interval_bindings_verified=39,free_unobserved_entries_sole_raw_RETURN=26,separate_certificate_entry_differences_from_raw_auxiliary_RETURN=sum(a!=b for a,b in zip(values,x)),raw_positive_congruence_entries=121 if rawrank is not None else 0,separate_positive_congruence_entries=121 if proposal_rank is not None else 0,independent_dual_Gram_entries=Gram_entries,independent_residual_corrected_dual_verified=dual_verified,independent_strict_diagnostic_gap=c.rec(gap)if gap is not None else None,diagnostic_envelope_excluded=dual_verified and gap>0,original1125table_actual_returns_source_time_authority_UNCHANGED=True,no_common_supported_measure_source_truth_deterministic_robot_path_or_physical_candidate=True,classification='EXACT_first_second_Gram_completion_of_DIAGNOSTIC_envelope_ONLY'if exact_proposal else'UNRESOLVED_no_exact_completion_certificate',new_solver_calls=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
with (O/'report.json').open('w')as f:f.write(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
print(json.dumps({k:v for k,v in out.items()if k not in['all_outputs_sha256','actual_auxiliary_constraints','separate_exact_constraints','independent_strict_diagnostic_gap']}))

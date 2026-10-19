"""Independent factored corner signs and rising-product moment certificates.
No old audits, new density ansatz, solver or completion is executed.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product,combinations
import datetime,hashlib,json,os,time
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
S=Path(__file__).resolve().parents[1];O=S/'e037-product-density-independent-v1';B=S/'e037-product-density-binding-v1';E=S/'e037-product-density-evaluation-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();J=(O/'independent-density-proposals-and-returns.jsonl').open('w');emit=lambda v:g.journal(J,v)
def save(name,v):
 with(O/name).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def rising(a,r):
 z=F(1)
 for k in range(r):z*=a+k
 return z
model=read(B/'fixed-density-model.json');er=read(E/'report.json');bindings=er['all_saved_input_bindings'];emit(dict(kind='BEFORE_NEW_independent_input_binding',bindings=bindings));ok=all(sha(p)==v for p,v in bindings.items());emit(dict(kind='AFTER_NEW_independent_input_binding_RETURN',passed=ok));assert ok;M=[[c.dec(z)for z in row]for row in read(S/'e037-global-completion-evaluation-v1/separate-existential-completion-matrix.json')['matrix']];ps=[{k:c.dec(z)for k,z in p.items()}for p in model['parameters']];P={tuple(row['powers']):c.dec(row['coefficient'])for row in model['polynomial']};corrections=[];parameter_checks=[]
for i,p in enumerate(ps,1):
 emit(dict(kind='BEFORE_NEW_independent_parameter_identity',index=i));mu=M[0][i];v=M[i][i]-mu**2;k=mu*(1-mu)/v-1;a=mu*k;b=(1-mu)*k;passed=0<mu<1 and 0<v<mu*(1-mu)and p==dict(mu=mu,variance=v,kappa=k,a=a,b=b)and a>0 and b>0;ret=dict(index=i,passed=passed,mu=c.rec(mu),variance=c.rec(v),a=c.rec(a),b=c.rec(b));emit(dict(kind='AFTER_NEW_independent_parameter_identity_RETURN',RETURN=ret));parameter_checks.append(ret);assert passed
for i,j in combinations(range(10),2):
 cov=M[i+1][j+1]-ps[i]['mu']*ps[j]['mu'];lam=cov/(ps[i]['variance']*ps[j]['variance']);corrections.append((i,j,lam))
assert [dict(indices=[i,j],covariance=c.rec(M[i+1][j+1]-ps[i]['mu']*ps[j]['mu']),coefficient=c.rec(lam))for i,j,lam in corrections]==model['corrections']
# Derive coefficients independently from corner-factor identities, without helper.
independent={(0,)*10:F(1)}
for i,j,lam in corrections:
 terms=[((0,)*10,lam*ps[i]['mu']*ps[j]['mu']),(tuple(int(k==i)for k in range(10)),-lam*ps[j]['mu']),(tuple(int(k==j)for k in range(10)),-lam*ps[i]['mu']),(tuple(int(k in[i,j])for k in range(10)),lam)]
 for powers,z in terms:independent[powers]=independent.get(powers,F(0))+z
independent={k:v for k,v in independent.items()if v};emit(dict(kind='AFTER_NEW_independent_coefficient_binding_RETURN',coefficients=len(independent),passed=independent==P));assert independent==P
saved=read(E/'all-1024-fixed-density-corner-returns.json');corner_cert=[];negative=[]
for index,b in enumerate(product((0,1),repeat=10)):
 emit(dict(kind='BEFORE_NEW_independent_factored_corner',index=index,corner=b));value=F(1)+sum(lam*(F(b[i])-ps[i]['mu'])*(F(b[j])-ps[j]['mu'])for i,j,lam in corrections);row=saved[index];passed=row['index']==index and row['corner']==list(b)and value==c.dec(row['density'])and row['sign']==(-1 if value<0 else 1 if value>0 else 0);ret=dict(index=index,corner=list(b),density=c.rec(value),negative=value<0,passed=passed,strict_negative_continuity_witness_for_THIS_ansatz_ONLY=value<0);emit(dict(kind='AFTER_NEW_independent_factored_corner_RETURN',RETURN=ret));corner_cert.append(ret)
 if value<0:negative.append(ret)
 assert passed
# Independent finite Beta moments use rising products, not producer recurrence.
tables=[[rising(p['a'],r)/rising(p['a']+p['b'],r)for r in range(4)]for p in ps];emit(dict(kind='AFTER_NEW_independent_Beta_rising_product_RETURN',moments=[[c.rec(v)for v in row]for row in tables],only_used_as_proposed_density_certificate_NOT_measured_tensor_values=True));moments=[];savedmom=read(E/'all-66-exact-normalization-first-second-cross-identities.json');cursor=0
for i in range(11):
 for j in range(i,11):
  powers=[int(i==k)+int(j==k)for k in range(1,11)];emit(dict(kind='BEFORE_NEW_independent_centered_moment_identity',indices=[i,j],powers=powers));base=F(1)
  for k,r in enumerate(powers):base*=tables[k][r]
  total=base
  for a,b,lam in corrections:
   term=lam*(tables[a][powers[a]+1]-ps[a]['mu']*tables[a][powers[a]])*(tables[b][powers[b]+1]-ps[b]['mu']*tables[b][powers[b]])
   for k,r in enumerate(powers):
    if k not in[a,b]:term*=tables[k][r]
   total+=term
  row=savedmom[cursor];cursor+=1;passed=row['indices']==[i,j]and row['powers']==powers and total==M[i][j]==c.dec(row['value'])==c.dec(row['target']);ret=dict(indices=[i,j],moment=c.rec(total),passed=passed,algebraic_closure_does_NOT_establish_positive_density=True);emit(dict(kind='AFTER_NEW_independent_centered_moment_RETURN',RETURN=ret));moments.append(ret);assert passed
lo=min(corner_cert,key=lambda r:c.dec(r['density']));hi=max(corner_cert,key=lambda r:c.dec(r['density']));negative_keys={r['index']for r in read(E/'all-negative-density-corner-certificates.json')};assert negative_keys=={r['index']for r in negative}and len(negative)==er['negative_corners']and lo['density']==er['minimum_density']and hi['density']==er['maximum_density'];events=[json.loads(line)for line in(E/'ONE-density-proposals-and-returns.jsonl').read_text().splitlines()];selected=[v for v in events if v['kind']in['BEFORE_NEW_density_corner','AFTER_NEW_density_corner_RETURN']];cornersordered=len(selected)==2048 and all(selected[2*i]['kind']=='BEFORE_NEW_density_corner'and selected[2*i+1]['kind']=='AFTER_NEW_density_corner_RETURN'and selected[2*i]['index']==selected[2*i+1]['RETURN']['index']==i for i in range(1024));ms=[v for v in events if v['kind']in['BEFORE_NEW_formal_density_moment','AFTER_NEW_formal_density_moment_RETURN']];momordered=len(ms)==132 and all(ms[2*i]['kind']=='BEFORE_NEW_formal_density_moment'and ms[2*i+1]['kind']=='AFTER_NEW_formal_density_moment_RETURN'and ms[2*i]['indices']==ms[2*i+1]['RETURN']['indices']for i in range(66));emit(dict(kind='AFTER_NEW_independent_journal_binding_RETURN',corners1024ordered=cornersordered,moments66ordered=momordered));assert cornersordered and momordered and all(sha(p)==v for p,v in bindings.items())
save('independent-10-parameter-bindings.json',parameter_checks);save('independent-1024-factored-corner-certificates.json',corner_cert);save('independent-all-negative-ansatz-witnesses.json',negative);save('independent-66-centered-moment-certificates.json',moments);J.close();out=dict(schema='v2d.product-density-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_independent_checks_passed=True,independent_positive_parameter_bindings=10,independent_fixed_cross_bindings=45,independent_multiaffine_coefficients=len(P),independent_factored_corner_signs=1024,negative_corners=len(negative),minimum_density=lo['density'],maximum_density=hi['density'],independent_finite_Beta_rising_product_moments=40,independent_normalization_first_second_cross_identities=66,all1090durable_BEFORE_AFTER_pairs_verified=True,classification=er['classification'],THIS_fixed_density_ansatz_rejected=bool(negative),common_cube_measure_for_THIS_matrix_NOT_excluded=True,other_completions_envelope_source_time_family_and_task_NOT_excluded=True,all_actual_returns_entries_AND_original_source_authority_UNCHANGED=True,new_solver_calls=0,new_density_ansatzes=0,new_source_URDF_native_points=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k not in['all_outputs_sha256','minimum_density','maximum_density']},allow_nan=False))

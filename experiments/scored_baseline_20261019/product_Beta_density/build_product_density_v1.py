"""Bind one fixed product-Beta density to the SAME saved exact diagnostic matrix."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
import datetime,hashlib,json,os,time
import product_beta_density_v1 as h
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
S=Path(__file__).resolve().parents[1];O=S/'e037-product-density-binding-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();J=(O/'model-proposals-intermediate-returns-and-bindings.jsonl').open('w');emit=lambda v:g.journal(J,v)
def save(name,v):
 with(O/name).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
mp=S/'e037-global-completion-evaluation-v1/separate-existential-completion-matrix.json';vp=S/'e037-global-completion-model-v1/separate-global-completion-model.json';contract=S/'e037-product-density-contract-v1/report.json';bindings={str(p):sha(p)for p in [mp,vp,contract]};emit(dict(kind='BEFORE_NEW_input_binding',bindings=bindings));M=[[c.dec(v)for v in row]for row in read(mp)['matrix']];variables=read(vp)['variables'];cr=read(contract);emit(dict(kind='AFTER_NEW_input_binding_RETURN',dimension=len(M),variables=variables,all_NEW_contracts_passed=cr['all_NEW_controls_passed'],contracts=cr['new_analytic_controls']));assert cr['all_NEW_controls_passed']and cr['new_analytic_controls']==12 and len(M)==11 and len(variables)==10
emit(dict(kind='BEFORE_NEW_parameter_construction',scope='THIS_ONE_separate_exact_diagnostic_matrix_ONLY'));ret=h.parameters(M,emit);packed=dict(ret)
if'parameters'in ret:packed['parameters']=h.recparams(ret['parameters'])
save('parameter-constructor-RETURN.json',packed);emit(dict(kind='AFTER_NEW_parameter_constructor_RETURN',RETURN=packed));valid=ret['classification']=='VALID_positive_Beta_parameters_ONLY';counts={};allok=False
if valid:
 ps=ret['parameters'];emit(dict(kind='BEFORE_NEW_fixed_density_construction',all45corrections_fixed_from_saved_cross_entries=True,no_new_completion_values=True));poly,corrections=h.density(M,ps,emit);result=dict(parameters=h.recparams(ps),polynomial=h.recpoly(poly),corrections=corrections);save('density-constructor-RETURN.json',result);emit(dict(kind='AFTER_NEW_density_constructor_RETURN',RETURN=result))
 # Independent binding from original exact matrix and analytic Beta moments.
 checks=[];independent={};zero=(0,)*10;independent[zero]=F(1)
 for i,p in enumerate(ps,1):
  emit(dict(kind='BEFORE_NEW_independent_parameter_binding',index=i));mu=M[0][i];var=M[i][i]-mu**2;k=mu*(1-mu)/var-1;a=mu*k;b=(1-mu)*k;first=a/(a+b);second=a*(a+1)/((a+b)*(a+b+1));ok=p==dict(mu=mu,variance=var,kappa=k,a=a,b=b)and a>0 and b>0 and first==M[0][i]and second==M[i][i];r=dict(index=i,first=c.rec(first),second=c.rec(second),variance=c.rec(var),passed=ok);emit(dict(kind='AFTER_NEW_independent_parameter_binding_RETURN',RETURN=r));checks.append(ok)
 for row in corrections:
  i,j=row['indices'];emit(dict(kind='BEFORE_NEW_independent_fixed_pair_binding',indices=[i,j]));vi=M[i+1][i+1]-M[0][i+1]**2;vj=M[j+1][j+1]-M[0][j+1]**2;cov=M[i+1][j+1]-M[0][i+1]*M[0][j+1];lam=cov/(vi*vj);ei=tuple(int(k==i)for k in range(10));ej=tuple(int(k==j)for k in range(10));eij=tuple(int(k in[i,j])for k in range(10))
  for key,value in [(zero,lam*M[0][i+1]*M[0][j+1]),(ei,-lam*M[0][j+1]),(ej,-lam*M[0][i+1]),(eij,lam)]:independent[key]=independent.get(key,F(0))+value
  ok=lam==c.dec(row['coefficient'])and cov==c.dec(row['covariance']);emit(dict(kind='AFTER_NEW_independent_fixed_pair_binding_RETURN',indices=[i,j],lambda_coefficient=c.rec(lam),passed=ok));checks.append(ok)
 independent={k:v for k,v in independent.items()if v};allok=all(checks)and independent==poly and len(corrections)==45 and all(max(k)<=1 for k in poly);br=dict(all_independent_bindings_passed=allok,parameter_bindings=10,cross_correction_bindings=45,multiaffine_coefficients=len(poly),polynomial_identical_to_independent_matrix_expansion=independent==poly);save('independent-pre-evaluation-bindings.json',br);emit(dict(kind='AFTER_NEW_all_bindings_RETURN',RETURN=br));assert allok
 model=dict(schema='v2d.fixed-product-Beta-density.v1',variables=variables,bindings=bindings,**result,unit_mass_original=1,all26completed_entries_remain_existential=True,no_original_auxiliary_or_separate_certificate_entries_changed=True,declared_corner_count=1024,declared_normalization_first_second_cross_identities=66,positivity_UNASSESSED=True,even_if_positive_ONLY_first_second_diagnostic_moments_NOT_original_higher_tensor_or_source_time_support=True);save('fixed-density-model.json',model);counts=br
assert all(sha(p)==d for p,d in bindings.items());J.close();out=dict(schema='v2d.product-density-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),valid_parameter_domains=valid,all_independent_bindings_passed=allok,parameter_constructor_status=ret['classification'],**{k:v for k,v in counts.items()if k!='all_independent_bindings_passed'},new_solver_calls=0,new_episode_density_evaluations=0,all_saved_input_bindings=bindings,all_inputs_UNCHANGED=True,new_source_URDF_native_points=0,new_GPU_jobs=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k not in['all_outputs_sha256','all_saved_input_bindings']}))

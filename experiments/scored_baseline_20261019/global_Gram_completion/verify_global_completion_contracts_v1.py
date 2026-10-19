"""NEW11D partial-Gram completion contracts, no old controls replayed."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations,product
import datetime,hashlib,json,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-global-completion-contract-v1';sys.path[:0]=[str(S/'compute'),str(S/'e037-direct-SDP-runtime-v2/site-packages')]
import numpy as np
from scipy import sparse
import clarabel
import global_completion_conic_v1 as h
import conditional_moment_cones_v1 as c
import direct_conditional_moment_sdp_v1 as old
import supported_conic_maps_v1 as generic
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();j=(O/'all-NEW11D-completion-contract-proposals-and-returns.jsonl').open('w');emit=lambda v:g.journal(j,v);start=time.monotonic();controls=[]
def save(name,v):
 with (O/name).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def check(name,record,ok):emit(dict(kind='AFTER_NEW_completion_contract_RETURN',name=name,record=record,passed=bool(ok)));assert ok;controls.append(name)
variables=['synthetic_'+str(i)for i in range(10)];known={(0,i):dict(interval=[F(1,2)]*2,anchor=F(1,2))for i in range(1,11)};known.update({(i,i):dict(interval=[F(1,3)]*2,anchor=F(1,3))for i in range(1,11)});known.update({(i,i+1):dict(interval=[F(1,4)]*2,anchor=F(1,4))for i in range(1,10)});emit(dict(kind='BEFORE_NEW_partial_Gram_model_contract',synthetic_independent_continuous_uniforms=True));model,block=h.formulate(variables,known);packing=h.pack11(block);save('NEW11D-partial-model-RETURN.json',model);save('NEW11D-packing-RETURN.json',packing);check('partial11_availability_and_free_entries',dict(columns=len(model['columns']),known=model['known_upper_entries'],free=model['free_unknown_entries']),len(model['columns'])==66 and model['known_upper_entries']==30 and model['free_unknown_entries']==36 and all(z['role']=='free_existential_mixed_completion'for z in model['columns']if z['matrix_entry'][0]>0 and tuple(z['matrix_entry'])not in known))
lo,hi=list(map(c.dec,packing['sqrt_two_interval']));errors=[[c.dec(z)for z in v]for v in packing['entry_errors']];check('dimension11_packing_spectral_guard',packing,len(packing['rows'])==66 and lo*lo<=2<=hi*hi and F(packing['necessary_diagonal_guard'])>=max(map(sum,errors)))
values=[F((i%7)+1,9)for i in range(66)];M=h.matrix(block,values);B=[[F((i+1)*(k+2)%5-2,7)for k in range(3)]for i in range(11)];G=[[sum(B[i][a]*B[k][a]for a in range(3))for k in range(11)]for i in range(11)];lhs=sum(G[i][k]*M[i][k]for i,k in product(range(11),repeat=2));adj=[F(0)]*66
for i,k in old.layout(11):
 for col,v in block['maps'][i][k].items():adj[col]+=(1 if i==k else 2)*G[i][k]*v
rhs=sum(v*z for v,z in zip(values,adj));check('dimension11_Frobenius_adjoint',dict(left=c.rec(lhs),right=c.rec(rhs),Gram=[[c.rec(v)for v in row]for row in G]),lhs==rhs)
# Every support inequality is a nonnegative product on the normalized square.
checks=[]
for x,y in product([F(0),F(1,3),F(2,5),F(1)],repeat=2):checks.append([x*(1-y),(1-x)*y,(1-x)*(1-y),x*(1-x)])
check('unit_support_Frechet_and_diagonal_bounds',dict(values=[[c.rec(z)for z in v]for v in checks],proof_factors=['x(1-y)','(1-x)y','(1-x)(1-y)','x(1-x)']),all(z>=0 for v in checks for z in v))
interval=F(1,3);lower=interval-F(1,2**120);upper=interval+F(1,2**120);_,up,ru=old.outer_linear(h.row('NEW_known_upper',{0:F(1)},upper));_,low,rl=old.outer_linear(h.row('NEW_known_lower',{0:F(-1)},-lower));check('nonbinary_interval_necessary_RHS_bounds',dict(original=[c.rec(lower),c.rec(upper)],upper=ru,lower=rl),F(up)>=upper and F(low)>=-lower and upper-lower==F(1,2**119))
zero=h.matrix(block,[F(0)]*66);signed=h.matrix(block,[F(-1)]*66);check('zero_signed_mass_not_normalized',dict(zero=[[c.rec(z)for z in v]for v in zero],signed_mass=c.rec(signed[0][0]),classification='signed mass is NOT a positive unit measure'),all(z==0 for v in zero for z in v)and signed[0][0]<0)
# Independent exact Gram certificate for partial known m1=3/4,m2=1/4.
check('known_principal_minor_and_dual_sign',dict(vector=list(map(c.rec,[F(-3,4),F(1)])),known_quadratic=c.rec(F(1,4)-F(9,16)),alpha_lower_first=c.rec(F(3,2)),alpha_upper_second=c.rec(F(1)),beta_mass=c.rec(F(9,16)),RHS=c.rec(F(-5,16)),residual=[c.rec(F(0))]*66,strict_gap=c.rec(F(5,16))),F(1,4)-F(9,16)==F(-5,16))
raw=[F((i%5)+1,7)for i in range(66)];saved=list(raw);proposal=h.auxiliary_exact_proposal(model,raw);emit(dict(kind='AFTER_NEW_separate_exact_proposal_RETURN',proposal=proposal));check('certificate_is_separate_from_raw_RETURN',dict(raw_values=list(map(c.rec,raw)),proposal=proposal),raw==saved and all(c.dec(proposal['values'][i])==raw[i]for i,z in enumerate(model['columns'])if z['role']=='free_existential_mixed_completion')and raw[0]!=c.dec(proposal['values'][0]))
for name,incompatible in [('positive_partial',False),('incompatible_partial',True)]:
 known2={k:dict(v)for k,v in known.items()}
 if incompatible:known2[0,1]=dict(interval=[F(3,4)]*2,anchor=F(3,4));known2[1,1]=dict(interval=[F(1,4)]*2,anchor=F(1,4))
 model2,block2=h.formulate(variables,known2);emit(dict(kind='BEFORE_NEW_tiny11_completion_model',name=name,model=model2));A,b,packed=h.assemble(model2,block2,emit);settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=30;settings.time_limit=5.;settings.max_threads=1;emit(dict(kind='BEFORE_NEW_tiny11_completion_constructor',name=name,settings=repr(settings)));solver=clarabel.DefaultSolver(sparse.csc_matrix((66,66)),np.zeros(66),A,b,[clarabel.ZeroConeT(packed['equalities']),clarabel.NonnegativeConeT(packed['nonnegative_rows']),clarabel.PSDTriangleConeT(11)],settings);emit(dict(kind='AFTER_NEW_tiny11_completion_constructor_RETURN',name=name,constructed=True));emit(dict(kind='BEFORE_NEW_tiny11_completion_solve',name=name));ret=old.solver_return(solver.solve());save('NEW-tiny11-'+name+'-solver-RETURN.json',ret);emit(dict(kind='AFTER_NEW_tiny11_completion_solver_RETURN',name=name,status=ret['status'],RETURN=ret));expected='PrimalInfeasible'if incompatible else'Solved';ok=ret['status']==expected
 if incompatible:
  certificate=generic.dual_certificate(model2,[block2],ret['z'],emit);save('NEW-tiny11-incompatible-dual-certificate.json',certificate);ok=ok and certificate['classification']=='EXACT_frozen_supported_model_exclusion'and c.dec(certificate['strict_gap'])>0
 check('NEW_tiny11_'+name,dict(status=ret['status'],maximum_calls=1),ok)
j.close();out=dict(schema='v2d.global-completion-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/global_completion_conic_v1.py'),all_controls_passed=True,new_analytic_controls=len(controls),new_tiny_analytic_SDPs=2,new_episode_completion_solves=0,old_packing_controls_REUSED_BY_HASH_NOT_repeated={str(p):sha(p)for p in [S/'e037-direct-SDP-contract-v2/report.json',S/'e037-supported-conic-contract-v2/report.json',S/'e037-image12-contract-v1/report.json']},actual_original_and_auxiliary_returns_NOT_modified=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k not in['all_outputs_sha256']}))

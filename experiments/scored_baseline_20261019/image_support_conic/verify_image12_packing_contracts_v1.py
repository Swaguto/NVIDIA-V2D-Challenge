"""Eight NEW12D packing/adjoint/spectral/dual contracts, two tiny NEW SDPs."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import datetime,hashlib,json,math,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-image12-contract-v1';sys.path[:0]=[str(S/'compute'),str(S/'e037-direct-SDP-runtime-v2/site-packages')]
import numpy as np
from scipy import sparse
import clarabel
import image_conic_packing_v1 as q
import supported_conic_maps_v1 as h
import direct_conditional_moment_sdp_v1 as old
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();j=(O/'all-new12D-contract-proposals-and-returns.jsonl').open('w');emit=lambda v:g.journal(j,v);start=time.monotonic();controls=[]
def save(name,v):
 with (O/name).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def check(name,record,valid):emit(dict(kind='AFTER_NEW12_contract_RETURN',name=name,record=record,passed=bool(valid)));assert valid;controls.append(name)
# New synthetic dimension12 dense matrix map; all tiny terms remain exact.
n=12;maps=[[{k:F((i+1)*(r+1)*(k+2)+1,(i+r+k+3)*7)for k in range(4)}for r in range(n)]for i in range(n)]
for i in range(n):
 for r in range(i,n):maps[r][i]=maps[i][r]
block=dict(id='NEW12_analytic',dimension=12,kind='synthetic',maps=maps);emit(dict(kind='BEFORE_NEW12_packing',block=h.block_record(block)));packing=q.pack12(block);save('new12-synthetic-packing-RETURN.json',packing);lo,hi=list(map(c.dec,packing['sqrt_two_interval']));errors=[[c.dec(v)for v in row]for row in packing['entry_error_bounds']];radius=max(map(sum,errors));check('dimension12_layout_and_spectral_guard',packing,len(packing['rows'])==78 and lo*lo<=2<=hi*hi and c.dec(packing['spectral_error_bound'])==radius and F(packing['outward_diagonal_guard'])>=radius)
failures=[]
for corner in product([F(0),F(1)],repeat=4):
 for row in packing['rows']:
  i,k=row['i'],row['j'];exact=sum(v*corner[col]for col,v in maps[i][k].items());floating=sum(F(z['value'])*corner[z['column']]for z in row['packed_coefficients']);error=abs(floating-exact)if i==k else max(abs(floating/lo-exact),abs(floating/hi-exact))
  if error>errors[i][k]:failures.append((corner,i,k))
check('sixteen_unitbox_corner_errors',dict(corners=16,rows_per_corner=78,failure_count=len(failures)),not failures)
B=[[F((i+2)*(k+1)%7-3,5)for k in range(4)]for i in range(n)];G=[[sum(B[i][r]*B[k][r]for r in range(4))for k in range(n)]for i in range(n)];x=[F(1,7),F(2,7),F(3,7),F(4,7)];M=[[sum(z*x[col]for col,z in co.items())for co in row]for row in maps];lhs=sum(G[i][k]*M[i][k]for i,k in product(range(n),repeat=2));adjoint=[sum((1 if i==k else 2)*G[i][k]*maps[i][k][col]for i,k in old.layout(n))for col in range(4)];rhs=sum(v*z for v,z in zip(x,adjoint));check('dimension12_Frobenius_adjoint',dict(Gram_factor=[[c.rec(v)for v in row]for row in B],left=c.rec(lhs),right=c.rec(rhs),adjoint=list(map(c.rec,adjoint))),lhs==rhs)
zero=[[sum(z*F(0)for z in co.values())for co in row]for row in maps];signed=[[sum(z*(F(-1)if col%2 else F(1))for col,z in co.items())for co in row]for row in maps];negative=[[sum(z*(F(1)if col%2 else F(-1))for col,z in co.items())for co in row]for row in maps];check('dimension12_zero_and_signed_linearity',dict(zero=[[c.rec(v)for v in row]for row in zero],signed=[[c.rec(v)for v in row]for row in signed]),all(v==0 for row in zero for v in row)and all(signed[i][k]==-negative[i][k]for i,k in product(range(n),repeat=2)))
asymmetric=[[dict(co)for co in row]for row in maps];asymmetric[0][11][0]+=F(1,2**500);rejected=False;emit(dict(kind='BEFORE_NEW12_tiny_asymmetric_map',block=h.block_record(dict(block,maps=asymmetric))))
try:q.pack12(dict(block,maps=asymmetric))
except ValueError as e:emit(dict(kind='AFTER_NEW12_asymmetry_RETURN',error=str(e)));rejected=True
check('dimension12_tiny_asymmetry_rejected',dict(rejected=rejected),rejected)
# Original-map finite-box certificate has gap1/2 for x>=3/4,x<=1/4.
residual=[F(-3,4)+F(3,4),F(1)-F(1)];rhs=F(1,4)-F(3,4);gap=sum(min(F(0),v)for v in residual)-rhs;check('dimension12_exact_Gram_finitebox_dual_sign',dict(Gram_diagonal=c.rec(F(1,12)),alpha=c.rec(F(1)),beta=c.rec(F(-3,4)),residual=list(map(c.rec,residual)),RHS=c.rec(rhs),strict_gap=c.rec(gap)),gap==F(1,2))
for name,threshold,upper,expected in [('feasible',F(1,4),F(3,4),'Solved'),('infeasible',F(3,4),F(1,4),'PrimalInfeasible')]:
 maps2=[[({0:-threshold,1:F(1)}if i==k else{})for k in range(n)]for i in range(n)];newblock=dict(id='tiny12:'+name,dimension=12,kind='NEW12_tiny',maps=maps2);A0=sparse.csc_matrix([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.],[1.,0.],[0.,1.]]);b0=np.array([1.,float(upper),0.,0.,1.,1.]);A,b,packed=q.append12(A0,b0,[newblock],emit);settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=30;settings.time_limit=5.;settings.max_threads=1;emit(dict(kind='BEFORE_NEW_tiny12_constructor',name=name,settings=repr(settings),block=h.block_record(newblock)));solver=clarabel.DefaultSolver(sparse.csc_matrix((2,2)),np.zeros(2),A,b,[clarabel.ZeroConeT(1),clarabel.NonnegativeConeT(5),clarabel.PSDTriangleConeT(12)],settings);emit(dict(kind='AFTER_NEW_tiny12_constructor_RETURN',name=name,constructed=True));emit(dict(kind='BEFORE_NEW_tiny12_solve',name=name));ret=old.solver_return(solver.solve());save('NEW-tiny12-'+name+'-solver-RETURN.json',ret);emit(dict(kind='AFTER_NEW_tiny12_solver_RETURN',name=name,status=ret['status'],RETURN=ret));valid=ret['status']==expected
 if name=='infeasible':
  model=dict(columns=[{},{}],equalities=[dict(bound=c.rec(F(1)),coefficients=[dict(column=0,value=c.rec(F(1)))])],inequalities=[dict(bound=c.rec(upper),coefficients=[dict(column=1,value=c.rec(F(1)))])]);cert=h.dual_certificate(model,[newblock],ret['z'],emit);save('NEW-tiny12-infeasible-certificate-RETURN.json',cert);valid=valid and cert['classification']=='EXACT_frozen_supported_model_exclusion'and c.dec(cert['strict_gap'])>0
 check('NEW_tiny12_'+name,dict(status=ret['status'],maximum_calls=1),valid)
j.close();out=dict(schema='v2d.image12-conic-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/image_conic_packing_v1.py'),all_controls_passed=True,new_analytic_controls=len(controls),new_tiny_dimension12_SDPs=2,new_episode_SDPs=0,old_dimension3_contract_report_sha256=sha(S/'e037-direct-SDP-contract-v2/report.json'),old_dimension4_6_8_10_contract_report_sha256=sha(S/'e037-supported-conic-contract-v2/report.json'),old_controls_reused_BY_HASH_NOT_rerun=True,all_actual_returns_preserved=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))

"""Twelve NEW24Dpacking/auxiliary/congruence/dual contracts;2tiny NEWSDPs."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import datetime,hashlib,json,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-global24-contract-v1';sys.path.insert(0,str(S/'e037-direct-SDP-runtime-v2/site-packages'))
import numpy as np
from scipy import sparse
import clarabel
import global_conic24_v1 as q
import supported_conic_maps_v1 as h
import direct_conditional_moment_sdp_v1 as old
import mixed_joint_moments_v1 as m
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();J=(O/'NEW24packing-congruence-dual-proposals-and-returns.jsonl').open('w');emit=lambda v:g.journal(J,v);controls=[]
def save(n,v):
 with(O/n).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def begin(n,v):emit(dict(kind='BEFORE_NEW24contract',name=n,proposal=v))
def check(n,v,ok):emit(dict(kind='AFTER_NEW24contract_RETURN',name=n,record=v,passed=bool(ok)));assert ok,n;controls.append(n)
n=24;maps=[[{}for _ in range(n)]for _ in range(n)]
for i,j in old.layout(n):maps[i][j]=maps[j][i]={k:F((i+1)*(j+2)*(k+1)+1,(i+j+k+3)*11)for k in range(4)}
block=dict(id='NEW24_auxiliary_analytic',dimension=24,kind='synthetic',maps=maps);begin('new24layout_and_full_spectral_guard',dict(block=h.block_record(block)));packed=q.pack24(block);save('NEW24-packing-RETURN.json',packed);lo,hi=map(c.dec,packed['sqrt_two_interval']);errors=[[c.dec(z)for z in row]for row in packed['entry_error_bounds']];radius=max(map(sum,errors));check('new24layout_and_full_spectral_guard',dict(rows=len(packed['rows']),radius=c.rec(radius)),len(packed['rows'])==300 and lo*lo<=2<=hi*hi and radius==c.dec(packed['spectral_error_bound'])and F(packed['outward_diagonal_guard'])>=radius)
begin('new24sixteen_original_plus_auxiliary_corner_errors',{});ok=True
for corner in product((F(0),F(1)),repeat=4):
 for row in packed['rows']:
  i,j=row['i'],row['j'];expected=sum(v*corner[k]for k,v in maps[i][j].items());numeric=sum(F(z['value'])*corner[z['column']]for z in row['packed_coefficients']);err=abs(numeric-expected)if i==j else max(abs(numeric/lo-expected),abs(numeric/hi-expected));ok=ok and err<=errors[i][j]
check('new24sixteen_original_plus_auxiliary_corner_errors',dict(corners=16,entry_error_checks=4800),ok)
begin('new24exact_Frobenius_original_and_auxiliary_adjoint',{});B=[[F((i+1)*(k+2)%9-4,7)for k in range(3)]for i in range(n)];G=[[sum(B[i][r]*B[j][r]for r in range(3))for j in range(n)]for i in range(n)];x=[F(1,5),F(2,5),F(3,5),F(4,5)];M=[[sum(v*x[k]for k,v in co.items())for co in row]for row in maps];left=sum(G[i][j]*M[i][j]for i,j in product(range(n),repeat=2));adj=[sum((1 if i==j else 2)*G[i][j]*maps[i][j][col]for i,j in old.layout(n))for col in range(4)];right=sum(a*b for a,b in zip(x,adj,strict=True));check('new24exact_Frobenius_original_and_auxiliary_adjoint',dict(left=c.rec(left),right=c.rec(right),adjoint=list(map(c.rec,adj))),left==right)
begin('new24tiny_asymmetric_alias_map_rejected',{});asym=[[dict(co)for co in row]for row in maps];asym[0][23][3]+=F(1,2**400);ret=None
try:q.pack24(dict(block,maps=asym))
except ValueError as ex:ret=str(ex)
check('new24tiny_asymmetric_alias_map_rejected',dict(RETURN=ret),ret=='asymmetric24maps_NOT_averaged')
for name,kind in[('new24zero_pivot_PSD','PSD'),('new24zero_pivot_nonzero_cross','cross'),('new24negative_pivot','negative')]:
 begin(name,dict(dimension=24));A=[[F(int(i==j))for j in range(n)]for i in range(n)];A[0][0]=F(0)if kind!='negative'else F(-1,2**130)
 if kind=='cross':A[0][23]=A[23][0]=F(1,3)
 emit(dict(kind='BEFORE_NEW24exact_congruence',name=name,matrix=[[c.rec(z)for z in row]for row in A]));ret=m.congruence_certificate(A,lambda v:emit(dict(contract=name,**v)));save(name+'-congruence-RETURN.json',ret);emit(dict(kind='AFTER_NEW24exact_congruence_RETURN',name=name,RETURN=ret));ok=ret['classification']=='exact_mixed_matrix_PSD'and ret['rank']==23 if kind=='PSD'else ret['classification']=='exact_negative_mixed_square'and c.dec(ret['expectation'])<0;check(name,dict(classification=ret['classification'],rank=ret.get('rank'),expectation=ret.get('expectation')),ok)
begin('new24prefix_extension_and_appended_auxiliary_boxes',{});A0=sparse.csc_matrix([[1.,0.],[0.,1.]]);b0=np.array([1.,.75]);toy=[[({2:F(1),0:F(-1,5)}if i==j else{})for j in range(n)]for i in range(n)];tb=dict(id='NEW24_prefix',dimension=24,kind='synthetic',maps=toy);A,b,p=q.append_global(A0,b0,tb,3,[2],emit);check('new24prefix_extension_and_appended_auxiliary_boxes',dict(shape=list(A.shape),PSD_start=p['row_start'],new_unit_boxes=[0,1]),A.shape==(304,3)and(A[:2,:2]!=A0).nnz==0 and A[:2,2:].nnz==0 and b[:2].tobytes()==b0.tobytes()and A[2,2]==-1 and A[3,2]==1 and b[2]==0 and b[3]==1 and p['row_start']==4)
begin('new24auxiliary_error_contribution_is_NOT_omitted',{});onlyaux=[[{}for _ in range(n)]for _ in range(n)];onlyaux[0][23]=onlyaux[23][0]={3:F(1,3)};p=q.pack24(dict(block,maps=onlyaux));error=c.dec(p['entry_error_bounds'][0][23]);check('new24auxiliary_error_contribution_is_NOT_omitted',dict(auxiliary_entry_error=c.rec(error),spectral=c.rec(c.dec(p['spectral_error_bound']))),error>0 and c.dec(p['spectral_error_bound'])>=error and F(p['outward_diagonal_guard'])>=error)
begin('new24alias_coupled_finite_box_dual_sign',dict(mass=1,auxiliary_u_equals_original_x=True,original_x_upper='1/3',PSD_auxiliary_lower='2/3'));# beta mass=-2/3,alias=-1;alpha upper=1;Z=I/24.
residual=[F(0)]*3;rhs=F(1,3)-F(2,3);gap=sum(min(F(0),z)for z in residual)-rhs;check('new24alias_coupled_finite_box_dual_sign',dict(residual=list(map(c.rec,residual)),RHS=c.rec(rhs),strict_gap=c.rec(gap),all3unitbox_variables_in_residual_bound=True),gap==F(1,3))
for name,threshold,upper,expected in[('feasible',F(1,5),F(4,5),'Solved'),('infeasible',F(2,3),F(1,3),'PrimalInfeasible')]:
 begin('NEW_tiny24_alias_coupled_'+name,dict(maximum_solver_calls=1));M=[[({0:-threshold,2:F(1)}if i==j else{})for j in range(n)]for i in range(n)];block2=dict(id='NEW_tiny24:'+name,dimension=24,kind='synthetic:auxiliary_alias',maps=M);A0=sparse.csc_matrix([[1.,0.,0.],[0.,-1.,1.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.],[1.,0.,0.],[0.,1.,0.]]);b0=np.array([1.,0.,float(upper),0.,0.,1.,1.]);A,b,pk=q.append_global(A0,b0,block2,3,[2],emit);settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=40;settings.time_limit=5.;settings.max_threads=1;emit(dict(kind='BEFORE_NEW_tiny24_constructor',name=name,settings=repr(settings)));solver=clarabel.DefaultSolver(sparse.csc_matrix((3,3)),np.zeros(3),A,b,[clarabel.ZeroConeT(2),clarabel.NonnegativeConeT(5),clarabel.NonnegativeConeT(2),clarabel.PSDTriangleConeT(24)],settings);emit(dict(kind='AFTER_NEW_tiny24_constructor_RETURN',name=name,constructed=True));emit(dict(kind='BEFORE_NEW_tiny24_solve',name=name));ret=old.solver_return(solver.solve());save('NEW-tiny24-'+name+'-solver-RETURN.json',ret);emit(dict(kind='AFTER_NEW_tiny24_solver_RETURN',name=name,RETURN=ret));valid=ret['status']==expected
 if name=='infeasible':
  model=dict(columns=[{}, {}, {}],equalities=[dict(bound=c.rec(F(1)),coefficients=h.co_rec({0:F(1)})),dict(bound=c.rec(F(0)),coefficients=h.co_rec({1:F(-1),2:F(1)}))],inequalities=[dict(bound=c.rec(upper),coefficients=h.co_rec({1:F(1)}))]);cert=h.dual_certificate(model,[block2],ret['z'],emit);save('NEW-tiny24-infeasible-exact-certificate-RETURN.json',cert);valid=valid and c.dec(cert['strict_gap'])>0
 check('NEW_tiny24_alias_coupled_'+name,dict(status=ret['status'],no_retry=True),valid)
J.close();out=dict(schema='v2d.global24-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_controls_passed=True,new_analytic_controls=len(controls),control_names=controls,new_tiny24_analytic_SDPs=2,new_episode_solver_calls=0,private_solver_version=clarabel.__version__,old_packing_controls_REUSED_BY_HASH_NOT_repeated={str(p):sha(p)for p in[S/'e037-direct-SDP-contract-v2/report.json',S/'e037-supported-conic-contract-v2/report.json',S/'e037-image12-contract-v1/report.json',S/'e037-global-completion-contract-v1/report.json']},new_native_position_checks=0,new_GPU_jobs=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k not in['all_outputs_sha256','old_packing_controls_REUSED_BY_HASH_NOT_repeated']}))

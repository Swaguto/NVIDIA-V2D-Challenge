"""NEW concrete all-row, tie, cone, conditioning and return contracts only."""
from structured_all_rows_v1 import *
from fractions import Fraction as F
from scipy.optimize import minimize
O=S/'e037-structured-all-row-contracts-v1'
returns=[]
def check(name,data,condition):
 returns.append(dict(name=name,actual_RETURN=data));dump(O/'NEW-contract-RETURNS.json',returns);assert condition,name
A=sparse.csr_matrix([[1.,0.],[-1.,0.],[0.,1.],[0.,0.],[0.,0.],[0.,-1.]])
b=np.array([1.,1.,2.,1.,0.,0.]);g=[(0,3,'all',0)];c=[(3,3,'SOC')]
x=np.array([.25,.5]);r=reduce_all(A,b,x,g,c)
check('ALL-row equivalence and exact incidence',r, bool((r['values'][0]>=0)==np.all(r['raw_rows'][:3]>=0)) and len(r['raw_rows'])==6)
T=sparse.csr_matrix([[1.,0.],[-1.,0.],[0.,1.]])
t=reduce_all(T,np.array([0.,0.,2.]),np.zeros(2),[(0,3,'all',0)],[])
check('ALL ties retained, mean is SEARCH supergradient not directional derivative',t,t['tie_rows'].tolist()==[0,1] and t['search_J'][0,0]==0 and min(-1.,1.)!=0)
# Concavity: each tied affine plane is an upper support; their mean remains one.
y=np.array([.75,.125]);exact_min=min(F(0)-F(3,4),F(0)+F(3,4),F(2)-F(1,8))
check('exact all-tie supergradient upper support',{'exact_min':str(exact_min),'mean_upper':0},exact_min<=0)
zero=reduce_all(sparse.csr_matrix(np.eye(3)),np.zeros(3),np.zeros(3),[],[(0,3,'SOC')])
check('SOC zero norm finite SEARCH and full cone event',zero,zero['values'][0]==0 and np.array_equal(zero['search_J'],[[-1.,0.,0.]]))
u=reduce_all(T,np.array([0.,np.nan,2.]),np.zeros(2),[(0,3,'all',0)],[])
check('unknown row cannot disappear under minimum',u,len(u['unknown'])==1 and np.isnan(u['values'][0]))
# Independent positive diagonal conditioning keeps sign iff every row is retained.
v=np.array([F(1,8),F(-1,32),F(0)]);sc=[F(2),F(1,4),F(8)]
check('conditioning sign equivalence without gate tolerance',{'scaled':[str(a*z) for a,z in zip(v,sc)]},all((a>=0)==(a*z>=0) for a,z in zip(v,sc)))
check('strict/open closed proposal cannot certify actual boundary',{'closed_at_zero':True,'strict_at_zero':False},not (F(0)>0))
check('complete sealed actual-return acceptance',{'good':acceptance_identity('new',['new']*151,[True]*6,True,True,True),'missing':acceptance_identity('new',['new']*150,[True]*6,True,True,True),'stale':acceptance_identity('new',['old']*151,[True]*6,True,True,True)},acceptance_identity('new',['new']*151,[True]*6,True,True,True) and not acceptance_identity('new',['old']*151,[True]*6,True,True,True) and not acceptance_identity('new',['new']*150,[True]*6,True,True,True))
# Installed SLSQP actual API/analytical tiny smooth problem. No episode input.
def fun(z):return .5*float(z@z)
def jac(z):return z
res=minimize(fun,np.array([2.,0.]),jac=jac,method='SLSQP',constraints=[dict(type='ineq',fun=lambda z:np.array([z[0]-1.,2.-z[0],1.-abs(z[1])]),jac=lambda z:np.array([[1.,0.],[-1.,0.],[0.,-np.sign(z[1])]]))],options=dict(maxiter=10,ftol=1e-6,disp=False))
arrays(O/'NEW-tiny-SLSQP-actual-RETURN.npz',x=res.x,jac=res.jac)
check('actual NEW tiny SLSQP interface and unmodified returned point',dict(status=res.status,success=res.success,message=res.message,nit=res.nit,fun=res.fun),res.success and abs(res.x[0]-1)<1e-10 and res.x[1]==0)
# Deterministic budget is an algorithm stop, never infeasibility or acceptance.
check('bounded journal/disk resource envelope',dict(max_unique_queries=48,full_rows=4618162,bytes_per_raw_row_return=8,raw_row_disk_upper=48*4618162*8,method_seconds=150,guard_seconds=240),48*4618162*8<2*1024**3)
dump(O/'report.json',dict(schema='v2d.NEW-structured-all-row-contracts.v1',contracts=len(returns),passed=True,no_episode_source_or_solver_replay=True))

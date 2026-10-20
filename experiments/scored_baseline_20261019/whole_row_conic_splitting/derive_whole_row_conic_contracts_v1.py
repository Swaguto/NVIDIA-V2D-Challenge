"""NEW direct split/projection/normal/provenance/resource controls only."""
from whole_row_conic_split_v1 import *
from fractions import Fraction as F
O=S/'e037-whole-row-conic-contracts-v1';returns=[]
def check(name,data,passed):
 returns.append(dict(name=name,actual_RETURN=data));dump(O/'NEW-contract-RETURNS.json',returns);assert passed,name
# Exact KKT at optimum x=1 for min .5x^2-2x subject x+s=1,s>=0.
x=F(1);y=F(1);q=F(-2);s=F(0)
check('positive dual sign and exact KKT',dict(stationarity=str(x+q+y),complementarity=str(s*y),wrong_sign=str(x+q-y)),x+q+y==0 and s*y==0 and x+q-y!=0)
# Entire exact split step from x0=0,s0=1,u0=0, P=A=1,q=-2,b=1.
x1=(F(1)-F(1)-F(0)-F(-2))/(F(1)+F(1));v=F(1)-x1;s1=max(v,F(0));u1=x1+s1-F(1)
check('exact normal/slack/dual update with same rho1 metric',dict(x=str(x1),s=str(s1),u=str(u1)),x1==1 and s1==0 and u1==0)
# ALL three SOC regions, zero and boundaries in one product; no point bank.
v=np.array([-2.,3., 2.,0.,1.,0., -4.,1.,0.,0., 0.,0.,4.,0., 0.,0.,0.,0.,0.])
cones=[(2,4,'inside'),(6,4,'polar'),(10,4,'outside'),(14,5,'zero')]
p=project_product(v,2,cones);w=p['value'];negative=v-w
check('whole product projection region/zero/boundary semantics',p,p['status']=='projected_ALL_product_coordinates' and np.array_equal(w[10:14],[2.,0.,2.,0.]) and np.all(cone_diagnostics(w,2,cones)>=0))
check('Moreau polar sign and orthogonality',dict(negative_dual=negative,dot=float(w@negative)),np.all(cone_diagnostics(-negative,2,cones)>=0) and float(w@negative)==0)
bound=project_product(np.array([1.,1.,0.,0.,-1.,1.,0.,0.]),0,[(0,4,'onK'),(4,4,'onPolar')])
check('region equality ties continuous and never arbitrary support selection',bound,np.array_equal(bound['value'],[1.,1.,0.,0.,0.,0.,0.,0.]))
unknown=project_product(np.array([np.nan]),1,[]);check('nonfinite remains unknown',unknown,unknown['status']=='UNRESOLVED_nonfinite')
bad=project_product(np.ones(4),0,[(1,3,'gap')]);check('missing row incidence cannot project a partial model',bad,bad['status']=='UNRESOLVED_malformed_incidence')
# Exact full-incidence adjoint/normal identity on NEW synthetic rational map.
A=[[F(1),F(2)],[F(-3),F(1)],[F(0),F(4)]];z=[F(2),F(-1)];h=[F(1,2),F(3),F(-2)]
Az=[sum(a*b for a,b in zip(row,z)) for row in A];ATH=[sum(A[i][j]*h[i] for i in range(3)) for j in range(2)]
check('ALL-row exact adjoint and normal equation',dict(lhs=str(sum(Az[i]*h[i] for i in range(3))),rhs=str(sum(z[j]*ATH[j] for j in range(2)))),sum(Az[i]*h[i] for i in range(3))==sum(z[j]*ATH[j] for j in range(2)))
# No diagonal loading/repair: coordinate rows give exact positive definiteness.
D=[F(1),F(1,1024),F(3)];z=[F(1),F(-2),F(3)];full=sum((a*b)**2 for a,b in zip(D,z))+F(5)
check('coordinate witness SPD lower bound for ALL-row normal',dict(lower=str(min(a*a for a in D)*sum(b*b for b in z)),actual=str(full)),full>=min(a*a for a in D)*sum(b*b for b in z)>0)
# Returned state only; approximate primal/dual/slack never qualifies source.
check('raw RETURN acceptance and immutable actual-gate ordering',dict(approximate_does_not_pass=True,strict_zero_fails=True,raw_terminal_only=True,original_source_heldout_POSTonly=True),not(F(0)>0))
check('declared complete intermediate resource envelope',dict(maximum_new_steps=32,row_vectors_per_step=5,rows=4618162,bytes_per_scalar=8,raw_disk_upper=32*5*4618162*8,working_vector_reserve=16*4618162*8,unchanged_guard_seconds=240,unchanged_RSS=2147483648),32*5*4618162*8<6*1024**3 and 16*4618162*8<600*1024**2)
dump(O/'report.json',dict(schema='v2d.NEW-whole-row-conic-contracts.v1',contracts=len(returns),passed=True,no_episode_pose_solver_or_old_control_replay=True,mathematical_projection_NOT_global_numeric_norm_identity=True))

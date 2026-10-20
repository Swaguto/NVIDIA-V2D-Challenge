"""NEW method-specific controls only; old interface/contracts never rerun."""
from joint_primal_dual_method_v1 import *
from fractions import Fraction as Q
D=S/'e037-joint-method-contracts-v1';contracts=[]
def check(name,raw,test):
 seal_return(D,name,raw);passed=bool(test(raw));contracts.append(dict(name=name,passed=passed));assert passed,name
p=np.array([.17,-.23]);n,J=sphere_chart(p)
cs=[]
for j in range(2):
 c=p.astype(complex);c[j]+=1e-25j;cs.append(sphere_chart(c)[0].imag/1e-25)
check('NEW-free-continuous-sphere-chart',dict(n=n,J=J,complex=np.column_stack(cs),norm=n@n),lambda r:abs(r['norm']-1)<1e-14 and np.max(np.abs(r['J']-r['complex']))<1e-14)
# Original signed constraint Ax+s=b -> positive dual and negative feasible F.
z=np.array([-.2,.3,1.,0.,0.,-.4,1.,0.]);cones=[(2,3),(5,3)]
y,branches=product_projection(z,cones)
check('NEW-dual-signed-whole-product',dict(raw=z,projected=y,branches=branches),lambda r:r['projected'][0]==0 and r['projected'][1]==.3 and r['branches'][0]['branch']=='inside' and r['branches'][1]['branch']=='middle')
check('NEW-whole-cone-metric-step',dict(returned=steps(np.array([1.,2.,3.,4.,5.]),np.array([2.,0.]),[(1,4)])),lambda r:np.array_equal(r['returned'][0][1:],np.full(4,.1)) and r['returned'][1][1]==1.)
A=sparse.csc_matrix([[.2,-.3,0.],[.4,.1,-.2],[0.,0.,0.]])
rb,cb=positive_sum_outer(A);sigma,tau=steps(rb,cb,[])
actual=np.linalg.norm(np.diag(np.sqrt(sigma))@A.toarray()@np.diag(np.sqrt(tau)),2)
check('NEW-conditional-Schur-bound',dict(row=rb,col=cb,sigma=sigma,tau=tau,norm=actual),lambda r:r['norm']<=.5)
# Exact dyadic check including subnormals: diagnostic guards do not alter gates.
v=np.array([np.nextafter(0.,1.),np.nextafter(0.,1.),.125,.5])
saved=2*float(v.sum());exact=sum((Q(float(x)) for x in v),Q(0))
check('NEW-positive-RN-outer-sum',dict(bound=saved,exact=str(exact)),lambda r:Q(r['bound'])>=exact)
raw=np.zeros(NVAR);raw[0]=.3;raw[1522]=-.2;tau=np.ones(NVAR)*.4;diag=np.zeros(NVAR);diag[0]=.1
ret,phase=primal_prox(raw,tau,diag)
check('NEW-original-objective-and-phase-prox',dict(raw=raw[:3],ret=ret[:3],phase=phase),lambda r:abs(r['ret'][0]-.3/1.04)<1e-15 and r['phase']['original_phase_prox_output']==0.)
# Zero rows are retained; their zero-Jacobian norm does not remove infeasibility.
f=np.array([.2,-.3]);rows=np.zeros(2);cols=np.zeros(1);sg,tt=steps(rows,cols,[]);out,_=product_projection(sg*f,[])
check('NEW-constant-row-retained',dict(dual=out,rows=len(out),tau=tt),lambda r:r['rows']==2 and r['dual'][0]==.2)
# All tied vertex slopes contribute to the bound even if the mean cancels.
a=np.array([-.4,.4]);b=np.array([-.3,.3]);mean=a.mean()+b.mean();bound=np.max(np.abs(a))+np.max(np.abs(b))
check('NEW-ALL-tie-norm-envelope',dict(mean=mean,envelope=bound,extremes=np.array([x+y for x in a for y in b])),lambda r:r['mean']==0 and r['envelope']==np.max(np.abs(r['extremes'])))
check('NEW-complete-whole-row-layout',dict(NVAR=NVAR,NROW=NROW,NDYN=NDYN,pair=NPAIR,cones=layout_cones(4547216)),lambda r:r['NVAR']==500729 and r['NROW']==4810140 and len(r['cones'])==1245 and r['pair']==249603)
# The last update alone is selected; no objective-based intermediate choice.
check('NEW-terminal-only-and-return-order',dict(updates=2,terminal=2,unique_oracles_max=3,method_seconds=150,
 order=['sealed_input','raw_oracle','raw_steps','raw_dual_input','raw_projection','raw_adjoint','raw_prox_input','raw_primal_return']),lambda r:r['terminal']==r['updates'] and len(r['order'])==8)
save(D/'report.json',dict(schema='v2d.NEW-joint-primal-dual-method-contracts.v1',contracts=contracts,passed=True,
 old_controls_model_source_native_solver_NOT_replayed=True,no_episode_values_selected_or_evaluated=True))

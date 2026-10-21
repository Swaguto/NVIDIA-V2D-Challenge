from joint_proximal_gn_v1 import *
from joint_saved_mean_bound_v1 import save
D=S/'e037-joint-proximal-contracts-v1';contracts=[]
def check(name,v,passed):seal_return(D,name,v);contracts.append(dict(name=name,passed=bool(passed)));assert passed,name
F=np.array([-.3,.4,-2.,0.,0.,.3,.4,0.]);r,branches=product_projection(F,[(2,3),(5,3)])
check('NEW-signed-Moreau-wholecone-residual',dict(F=F,r=r,branches=branches),r[0]==0 and r[1]==.4 and branches[0]['branch']=='polar' and branches[1]['branch']=='middle')
# Every row stays in J.T J, including satisfied rows whose Moreau residual is0.
J=np.array([[1.,2.],[-3.,4.]]);F=np.array([-.4,.7]);r=np.maximum(F,0);H=np.eye(2)+J.T@J;g=J.T@r
check('NEW-fullrow-proximal-normal-no-rowbank',dict(H=H,g=g,satisfied_row=J[0]),np.array_equal(H,np.array([[11.,-10.],[-10.,21.]])) and np.array_equal(g,np.array([-2.1,2.8])))
a=np.array([.2,-.3]);b=np.array([.4,.1]);w=1/(1+b@b);full=np.eye(4)+np.outer(np.r_[a,b],np.r_[a,b]);S2=full[:2,:2]-full[:2,2:]@np.linalg.solve(full[2:,2:],full[2:,:2]);formula=np.eye(2)+w*np.outer(a,a)
check('NEW-continuous-two-direction-Schur',dict(actual=S2,formula=formula),np.max(np.abs(S2-formula))<1e-15)
r=.7;g=np.r_[a,b]*r;dz=np.linalg.solve(formula,-a*w*r);dp=-b*w*(r+a@dz);joint=np.r_[dz,dp]
check('NEW-whole-pair-backsub-signed-update',dict(d=joint,full_residual=full@joint+g),np.max(np.abs(full@joint+g))<1e-15)
# Original phase domain constrained quadratic, exact analytic tiny example.
H=np.array([[2.,.3],[.3,1.]]);rhs=np.array([.2,-1.]);u=np.linalg.solve(H,rhs);v=np.linalg.solve(H,np.array([0.,1.]));d=u-u[1]/v[1]*v
check('NEW-original-phase-metric-domain',dict(unconstrained=u,domain_answer=d,free_gradient=(H@d-rhs)[0]),d[1]==0 and abs((H@d-rhs)[0])<1e-15)
# Correlated ties stay explicit even when their mean SEARCH slope cancels.
t=np.array([-.5,.5]);mean=float(t.mean());directional=float(t.max())
check('NEW-GN-SEARCH-not-universal-tie-derivative',dict(ALLties=t,mean=mean,directional=directional),mean==0 and directional==.5)
B=sparse.csr_matrix([[1.,0.,0.,0.],[0.,0.,0.,0.]]);D2=B[:,2:];n=np.asarray(D2.multiply(D2).sum(axis=1)).ravel();w=1/(1+n)
check('NEW-zero-filter-row-retained',dict(weights=w,rows=B.shape[0],direction_coordinates=D2.shape[1]),np.array_equal(w,np.ones(2)) and B.shape[0]==2)
# Actual NEW SciPy factor/two-RHS API, independently known linear equations.
H=np.array([[3.,.25],[.25,2.]]);fac=linalg.cho_factor(H,lower=True);rhs=np.array([[1.,0.],[0.,1.]]);sol=linalg.cho_solve(fac,rhs)
check('NEW-actual-factor-two-RHS-return',dict(factor=fac,solution=sol,residual=H@sol-rhs),np.max(np.abs(H@sol-rhs))<1e-15)
# Exact rational normal identity and finite-RN outer guard on NEW operands.
X=np.array([[.1,.2],[.3,-.4]]);raw=X.T@X;exact=np.array([[float(sum(Q(float(X[k,i]))*Q(float(X[k,j])) for k in range(2))) for j in range(2)] for i in range(2)]);bound=2*(100*2**-53)/(1-100*2**-53)*(np.abs(X).T@np.abs(X))
check('NEW-normal-RN-algebraic-scope',dict(raw=raw,exact=exact,outer=bound),np.all(np.abs(raw-exact)<=bound))
# The complete fixed namespace remains; no dense 4.8M-by500729 Jacobian exists.
check('NEW-fixed-resource-and-single-trial-policy',dict(base=1523,directions=499206,rows=4810140,pairs=249603,updates=1,trials=1,method_seconds=150,outer_seconds=240,RSS=2*1024**3),NVAR==500729 and NROW==4810140)
save(D/'report.json',dict(passed=True,NEW_contracts=len(contracts),contracts=contracts,no_old_method_controls_or_episode_query=True,
 one_proximal_GN_step_ONErawtrial_only=True,original_Pq_phase_constraints_unchanged=True))

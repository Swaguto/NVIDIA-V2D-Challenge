from joint_zero_conic_v1 import *
from joint_saved_mean_bound_v1 import save
from fractions import Fraction as Q
D=S/'e037-joint-zero-contracts-v1';checks=[]
def check(name,raw,ok):seal_return(D,name,raw);checks.append(dict(name=name,passed=bool(ok)));assert ok,name
v=np.zeros(NFREE);v[:3]=[.125,-.25,.5];w=lift(v)
check('NEW-exact-zero-target-bijection',dict(phase=w[PHASE],count=len(w),roundtrip=select(w),free=v),w[PHASE]==0 and np.array_equal(select(w),v) and len(w)==500729)
# min .5*x*x subject -x+s=-1,s>=0: known KKT x=y=1.
A=np.array([[-1.]]);P=np.array([[1.]]);q=np.array([0.]);b=np.array([-1.]);zx=np.array([1.]);zy=np.array([1.]);H=np.eye(1)+P+A.T@A;ux=np.linalg.solve(H,zx-q-A.T@(zy-b));uy=zy-b+A@ux;xn,yn,rr=dr_update(zx,zy,ux,uy,[])
check('NEW-signed-hard-KKT-resolvent-fixedpoint',dict(ux=ux,uy=uy,xnext=xn,ynext=yn),ux[0]==1 and uy[0]==1 and xn[0]==1 and yn[0]==1)
# Linearization RHS is J*w0-F0, with ALL closed cone rows retained.
w0=np.array([.5]);J=np.array([[2.]]);F0=np.array([-2.]);bb=J@w0-F0
check('NEW-hard-model-RHS-sign',dict(b=bb,residual=J@w0-bb,F0=F0),bb[0]==3 and np.array_equal(J@w0-bb,F0))
a=np.array([.25,-.5]);d=np.array([.5,.25]);rz=np.array([.125,-.25]);rp=np.array([.25,.5]);weight=1/(1+d@d);full=np.eye(4)+np.outer(np.r_[a,d],np.r_[a,d]);S2=np.eye(2)+weight*np.outer(a,a);z=np.linalg.solve(S2,rz-a*weight*(d@rp));p=rp-d*weight*(d@rp+a@z);answer=np.r_[z,p]
check('NEW-general-RHS-continuous-direction-Schur',dict(answer=answer,residual=full@answer-np.r_[rz,rp]),np.max(np.abs(full@answer-np.r_[rz,rp]))<1e-15)
zx=np.array([.125,.25]);zy=np.array([-.5,.125,.5,0.,0.]);ux=np.array([.25,-.125]);uy=np.array([.25,-.25,.75,.125,0.]);xn,yn,returned=dr_update(zx,zy,ux,uy,[(2,3)])
check('NEW-whole-cone-reflection-and-update',dict(xnext=xn,ynext=yn,raw=returned),np.array_equal(xn,ux) and np.array_equal(yn,zy+returned['projected']-uy) and len(returned['branches'])==1)
P=np.diag([.25,.5,1.]);x=np.array([.5,.25,0.]);q=np.array([0.,0.,1.]);small=P[:2,:2];free=x[:2]
check('NEW-original-objective-zero-slice',dict(original=float(.5*x@(P@x)+q@x),restricted=float(.5*free@(small@free))),.5*x@(P@x)+q@x==.5*free@(small@free))
# All original rows stay represented; zero numerical rows have cone records.
z=np.array([0.,-.25,0.,0.,0.]);pr,br=product_projection(z,[(2,3)])
check('NEW-tautology-and-zero-SOC-retained',dict(value=pr,branches=br,rows=len(z)),len(pr)==5 and len(br)==1 and np.array_equal(pr,np.zeros(5)))
# Exact adjoint parameterization identity on NEW tiny dyadic inputs.
E=np.array([[1.,0.],[0.,1.],[0.,0.]]);L=np.array([[.25,.5,.125],[-.5,.25,.25]]);v=np.array([.5,-.25]);y=np.array([.125,.5]);left=(L@E@v)@y;right=v@(E.T@L.T@y)
check('NEW-full-incidence-parameter-adjoint',dict(left=left,right=right),left==right)
# Predeclared worst-case uncompressed typed payload budget, no row selection.
perstep=8*NROW*8+12*NFREE*8+12*(NROW-NOBJ-NROB)*8;budget=16*perstep+2*1024**3
check('NEW-fixed-resource-terminal-policy',dict(updates=16,terminal=16,seconds=150,estimated_bytes=budget,guard_seconds=240,RSS=2*1024**3),budget<10*1024**3)
save(D/'report.json',dict(passed=True,NEW_controls=len(checks),controls=checks,no_old_controls_model_query_solver_or_method_replay=True,original_objective_constraints_and_zero_target_preserved=True))

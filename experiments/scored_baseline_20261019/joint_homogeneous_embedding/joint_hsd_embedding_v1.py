"""NEW whole-cone homogeneous embedding, square-root-free original objective."""
from joint_zero_conic_v1 import *
from joint_zero_return_guard_v1 import linear_return_guard
from joint_saved_mean_bound_v1 import save
NF=NFREE;NR=NROW
def epigraph_action(x,idx,p):
 t=x[NF:];out=np.empty((len(idx),3));out[:,0]=-t;out[:,1]=-t;out[:,2]=-p*x[idx]
 return out.ravel()
def epigraph_adjoint(y,idx,p):
 y=y.reshape(-1,3);out=np.zeros(NF+len(idx));out[idx]=-p*y[:,2];out[NF:]=-y[:,0]-y[:,1]
 return out
def action(A,R,J,x,idx,p,record,tag):
 orig=full_action(A,R,J,x[:NF],record,tag+'-original')
 epi=epigraph_action(x,idx,p);record(tag+'-epigraph-action-RETURN',dict(value=epi))
 out=np.r_[orig,epi];record(tag+'-ALL-embedded-action-RETURN',dict(value=out))
 return out
def adj(A,R,J,y,idx,p,record,tag):
 orig=full_adjoint(A,R,J,y[:NR],record,tag+'-original')
 epi=epigraph_adjoint(y[NR:],idx,p);record(tag+'-epigraph-adjoint-RETURN',dict(value=epi))
 out=epi.copy();out[:NF]+=orig;record(tag+'-ALL-embedded-adjoint-RETURN',dict(value=out))
 return out
def solve(B,D,w,factor,rhs,record,tag):
 original=resolvent_solve(B,D,w,factor,rhs[:NF],record,tag)
 epi=rhs[NF:]/3.;record(tag+'-epigraph-normal-solve-RETURN',dict(value=epi))
 out=np.r_[original,epi];record(tag+'-ALL-embedded-normal-solve-RETURN',dict(value=out))
 return out
def square_dot(value,record,tag):
 out=float(np.dot(value,value));record(tag+'-positive-square-dot-RETURN',dict(value=out,count=len(value),raw_NOT_repaired=True))
 return out
def split_projection(x,y,tau,cones):
 out,branches=product_projection(y,cones)
 return x.copy(),out,max(0.,float(tau)),branches

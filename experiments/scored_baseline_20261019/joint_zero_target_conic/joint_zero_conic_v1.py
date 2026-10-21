"""NEW hard joint zero-target conic KKT construction.

Douglas-Rachford resolves a NEW joint affine KKT inclusion. Operator splitting is
shared with prior ADMM theory, but no prior problem/iterate/budget is continued.
All original rows/Pq remain, plus the original required zero-phase target.
"""
from joint_primal_dual_method_v1 import NVAR,NROW,NOBJ,NROB,product_projection,method_oracle,finite_or_unknown
from joint_constraint_derivatives_v1 import JointExecutableInterface
from joint_method_pair_columns_v1 import MethodRecorder
from joint_saved_mean_bound_v1 import S,C,OLD,typed,sha,dynamic_file
from joint_lossless_returns_v1 import seal_return
from scipy import sparse,linalg
from pathlib import Path
import numpy as np,json,time,hashlib
BASE=1523;PHASE=1522;KEEP=np.arange(NVAR)!=PHASE;BASEKEEP=np.arange(BASE)!=PHASE;NFREE=NVAR-1

def lift(v):
 out=np.empty(NVAR);out[KEEP]=v;out[PHASE]=0.
 return out

def select(v):return np.asarray(v)[KEEP]

def full_action(A,R,J,v,record,tag):
 x=lift(v)
 a=A@x[465:1523];record(tag+'-object-action-RETURN',dict(value=a))
 r=R@x[:465];record(tag+'-robot-action-RETURN',dict(value=r))
 j=J@x;record(tag+'-dynamic-action-RETURN',dict(value=j))
 out=np.r_[a,r,j];record(tag+'-ALL4810140-action-RETURN',dict(value=out,formal_phase_equality_value=0.))
 return out

def full_adjoint(A,R,J,y,record,tag):
 a=A.T@y[:NOBJ];record(tag+'-object-adjoint-RETURN',dict(value=a))
 r=R.T@y[NOBJ:NOBJ+NROB];record(tag+'-robot-adjoint-RETURN',dict(value=r))
 j=J.T@y[NOBJ+NROB:];record(tag+'-dynamic-adjoint-RETURN',dict(value=j))
 x=np.zeros(NVAR);x[465:1523]=a;x[:465]=r;x+=j;out=select(x)
 record(tag+'-ALL500728-adjoint-RETURN',dict(value=out,formal_phase_component=j[PHASE]+a[-1]))
 return out

def resolvent_solve(B,D,weight,factor,rhs,record,tag):
 rz=rhs[:1522];rp=rhs[1522:]
 dr=D@rp;record(tag+'-D-rp-RETURN',dict(value=dr))
 wr=weight*dr;record(tag+'-weighted-D-rp-RETURN',dict(value=wr))
 correction=B.T@wr;record(tag+'-Schur-RHS-correction-RETURN',dict(value=correction))
 schur_rhs=rz-correction;record(tag+'-Schur-RHS-RETURN',dict(value=schur_rhs))
 z=linalg.cho_solve(factor,schur_rhs,check_finite=True);record(tag+'-Schur-solve-RETURN',dict(value=z))
 bz=B@z;record(tag+'-B-z-RETURN',dict(value=bz))
 back=weight*(dr+bz);record(tag+'-backsub-input-RETURN',dict(value=back))
 dp=D.T@back;record(tag+'-D-adjoint-backsub-RETURN',dict(value=dp))
 out=np.r_[z,rp-dp];record(tag+'-ALL-free-linear-resolvent-RETURN',dict(value=out))
 return out

def dr_update(zx,zy,ux,uy,cones):
 reflected=2*uy-zy;projected,branches=product_projection(reflected,cones)
 return ux.copy(),zy+projected-uy,dict(reflected=reflected,projected=projected,branches=branches)

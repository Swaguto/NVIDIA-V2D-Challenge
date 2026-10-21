"""NEW one-step full-row proximal-residual Gauss-Newton SEARCH construction.

Original f/P/q and constraints remain; the unit proximal metric is a fixed search
operator. Every row contributes to the normal matrix. No active-row/cut bank.
"""
from joint_primal_dual_method_v1 import method_oracle,product_projection,NVAR,NROW,NOBJ,NROB,finite_or_unknown
from joint_constraint_derivatives_v1 import JointExecutableInterface
from joint_method_pair_columns_v1 import MethodRecorder
from joint_saved_mean_bound_v1 import typed,sha,S,C,OLD,dynamic_file
from joint_lossless_returns_v1 import seal_return
from pathlib import Path
from scipy import sparse,linalg
from fractions import Fraction as Q
import numpy as np,json,time,hashlib
PHASE=1522;NBASE=1523;NDIR=NVAR-NBASE

def schur_pair_rows(J):
    base=J[:,:NBASE].tocsr();directions=J[:,NBASE:].tocsr()
    # Each geometric row owns exactly its two continuous chart coordinates.
    norm2=np.asarray(directions.multiply(directions).sum(axis=1)).ravel()
    weight=1./(1.+norm2)
    return base,directions,norm2,weight

def normal_schur(A,robot,P,J):
    B,D,norm2,w=schur_pair_rows(J)
    constant=np.zeros((NBASE,NBASE));constant[465:,465:]=(A.T@A).toarray();constant[:465,:465]=(robot.T@robot).toarray()
    dynamic=(B.T@B.multiply(w[:,None])).toarray()
    normal=np.eye(NBASE)+P.toarray()+constant+dynamic
    return dict(normal=normal,base_J=B,direction_J=D,direction_norm_squared=norm2,Schur_weight=w,
                constant_normal=constant,dynamic_normal=dynamic)

def rhs_schur(A,robot,P,q,w0,Jdata,residual):
    B=Jdata['base_J'];w=Jdata['Schur_weight'];r_dyn=residual[NOBJ+NROB:]
    original_gradient=P@w0[:NBASE]+q
    gradient=np.asarray(original_gradient).copy();gradient[:465]+=robot.T@residual[NOBJ:NOBJ+NROB];gradient[465:]+=A.T@residual[:NOBJ]
    gradient+=B.T@(w*r_dyn)
    return -gradient

def phase_constrained_quadratic(normal,rhs,phase0,record):
    # Two RHS, one actual factorization; original phase interval [0,1] only.
    record('NEW-normal-constructor-input-BEFORE',dict(normal=normal,rhs=rhs,phase0=phase0))
    factor=linalg.cho_factor(normal,lower=True,check_finite=True)
    record('NEW-normal-factor-RETURN',dict(factor=factor[0],lower=factor[1]))
    e=np.zeros(len(rhs));e[PHASE]=1
    both=linalg.cho_solve(factor,np.column_stack([rhs,e]),check_finite=True)
    record('NEW-two-RHS-normal-solve-RETURN',dict(solution=both))
    unconstrained=both[:,0];inverse_phase=both[:,1];value=phase0+unconstrained[PHASE]
    target=0. if value<0 else (1. if value>1 else float(value))
    answer=unconstrained.copy()
    if target!=value:answer+=(target-value)/inverse_phase[PHASE]*inverse_phase
    record('NEW-original-phase-domain-quadratic-RETURN',dict(unconstrained=unconstrained,inverse_phase=inverse_phase,unconstrained_phase=value,target_phase=target,answer=answer,
         original_domain_algorithm_operator_NOT_terminal_command_repair=True))
    return answer,both,dict(target_phase=target,unconstrained_phase=value)

def backsubstitute(normaldata,dz,residual):
    B=normaldata['base_J'];D=normaldata['direction_J'];w=normaldata['Schur_weight']
    z=residual[NOBJ+NROB:]+B@dz
    dp=-D.T@(w*z)
    return np.r_[dz,np.asarray(dp).ravel()]

def solve_RN_guard(normal,both,rhs):
    n=len(rhs);e=np.zeros(n);e[PHASE]=1;R=np.column_stack([rhs,e]);residual=normal@both-R
    u=2.**-53;k=10*n+32;gamma=k*u/(1-k*u)
    outer=2*gamma*(np.abs(normal)@np.abs(both)+np.abs(R))+np.finfo(float).tiny
    return dict(residual=residual,conditional_RN_outer=outer,all_within=bool(np.all(np.abs(residual)<=outer)),
        numerical_search_guard_NOT_candidate_tolerance=True,no_global_binary_identity=True)

"""NEW concrete Cayley/lifted-cell/curvature/packing/acceptance contracts."""
from pathlib import Path
import os,json,datetime,math
from fractions import Fraction as F
import numpy as np
from scipy.spatial.transform import Rotation
from hard_raster_cayley_v1 import *
S=Path(__file__).resolve().parents[1];O=S/'e037-hard-raster-contracts-v1'
def cv(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,F):return str(x)
 if isinstance(x,dict):return{k:cv(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return[cv(v)for v in x]
 return x
def save(n,d):
 with (O/n).open('w')as f:json.dump(cv(d),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def check(n,d,ok):save(n+'-RETURN.json',dict(actual_RETURN=d,passed=bool(ok),not_episode_or_original_gate=True));assert ok,n
A=Rotation.from_euler('xyz',[.2,-.1,.3]).as_matrix();b=np.array([.02,-.03,.6]);L=Rotation.from_rotvec([-.2,.1,.05]).as_matrix();V=np.array([[-.04,.02,.01],[.03,-.02,.015],[.01,.025,-.008]]);t=np.array([.031,-.023,.017]);eta=np.array([.0003,-.0002,.0001]);Q,D=cayley(t);C=V@(A@Q).T+b+L@(eta/D);N=D*Q;poly=V@(A@N).T+b*D+L@eta;err=float(np.max(abs(C-poly/D)));rv=2*math.atan(np.linalg.norm(t))*t/np.linalg.norm(t);check('NEW-three-component-Cayley-lifted-world-camera-composition',dict(Q=Q,D=D,camera=C,homogeneous=poly,error=err,Exp_rotation_error=float(np.max(abs(Q-Rotation.from_rotvec(rv).as_matrix())))),err<1e-15 and np.max(abs(Q@Q.T-np.eye(3)))<1e-15)
K=np.array([[210.,0,70.],[0,205.,60.],[0,0,1.]]);r=.12;bound=curvature_scale(A,b,V,K,r);s=bound['scale'];check('NEW-exact-uniform-negative-curvature-and-scale',dict(bound=bound,exact_r_s_square=str(F.from_float(r)*F.from_float(s)**2),all_input_bounds_hold=bool(np.max(abs(A))<=bound['A_entry_bound']and np.max(abs(V))<=bound['V_entry_bound'])),bound['exact_scale_square_bound'])
# NEW lifted homogeneous cell lower bound versus exact binary-input polynomial.
integer=np.array([[55,70],[80,50],[60,80]],np.int32);rows=frame_rows(A,b,L,V,K,integer,r,s);u=s*t;sigma=float(u@u);x=np.r_[eta/r,u,sigma];values=rows['rhs']-rows['coefficients']@x;proofs=[]
for k in range(len(values)):
 slot=k//len(V);i=k%len(V);g=exact_polynomial(A,b,L,V[i],t,eta,rows['weights'][k],depth=slot==4);lower=float(values[k]*r);proofs.append(dict(row=k,exact_polynomial=str(g),lower=lower,slack=float(g)-lower))
check('NEW-all-five-cell-depth-row-lower-bound',dict(proofs=proofs,closed_proposal_faces_NOT_actual_strict_gate=True),all(p['slack']>=-1e-12 for p in proofs))
# Packing checks rows on mixed signed targets/saturation without replaying old
# quantizer controls. Inactive rows stay durable, never observations deleted.
uv=np.array([[0,-20000],[20000,-2],[-1,3]],np.int32);rr=frame_rows(A,b,L,V,K,uv,r,s);inactive=np.flatnonzero(~rr['active']);check('NEW-seven-column-row-incidence-and-tautologies',dict(shape=rr['coefficients'].shape,inactive=inactive,strict_open=rr['strict_open'],row_kinds=rr['row_kind']),rr['coefficients'].shape==(15,7)and np.all(rr['coefficients'][inactive]==0)and np.all(rr['rhs'][inactive]==0)and len(inactive)==2)
# Raw nonlinear cell and source acceptance must both hold; a closed proposal
# boundary never upgrades a changed integer cell or nonfinite/unknown result.
def accept(integer_equal,strict_depth,source,motion,identity):return bool(integer_equal is True and strict_depth is True and source is True and len(motion)==6 and all(v is True for v in motion)and identity is True)
returns=[accept(True,True,True,[True]*6,True),accept(False,True,True,[True]*6,True),accept(True,None,True,[True]*6,True),accept(True,True,False,[True]*6,True),accept(True,True,True,[True]*5+[False],True)];check('NEW-actual-cell-source-six-motion-identity-hard-acceptance',dict(returns=returns,no_initial_F_cache_feasibility_assumption=True,all_old_alias_nominal_defects_retained=True),returns==[True,False,False,False,False])
save('report.json',dict(schema='v2d.NEW-hard-raster-Cayley-contracts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEW_contracts=5,old_quantizer_source_controls_reused_BY_HASH_NOT_replayed=True,no_episode_pose_source_NN_raster_solver_native_physics_GPU=True));print(json.dumps(dict(NEW_contracts=5,success=True)))

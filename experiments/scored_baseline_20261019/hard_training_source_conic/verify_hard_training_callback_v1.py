"""ONE NEW analytic callback-durability control, not an episode solve."""
from pathlib import Path
import datetime,hashlib,json,os,sys
S=Path(__file__).resolve().parents[1];O=S/'e037-hard-training-callback-contract-v1';sys.path.insert(0,str(S/'e037-direct-SDP-runtime-v2/site-packages'))
import clarabel,numpy as np
from scipy import sparse
from hard_training_conic_returns_v1 import snapshot
from thumb_conic_geometry_v1 import plain,journal
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,v)
 declaration=dict(schema='v2d.NEW-callback-durability-analytic-contract.v1',
  problem='min t; x=1/8; (t,x+15/8,3x+5/8,0) in SOC4; analytic optimum t=sqrt(5)',
  callback='Snapshot every actual returned info before continuing; no reentrant solver calls or reconstructed primal iterates',
  no_episode_source_or_old_control_input=True,iterations=50,seconds=10,threads=1,default_tolerances=True)
 save('ONE-declared-callback-control.json',declaration);emit(dict(kind='BEFORE_NEW_callback_control',declaration=declaration))
 A=sparse.csc_matrix([[1.,0],[0,-1],[-1,0],[-3,0],[0,0]]);b=np.array([.125,0,1.875,.625,0]);q=np.array([0.,1.]);P=sparse.csc_matrix((2,2));cones=[clarabel.ZeroConeT(1),clarabel.SecondOrderConeT(4)];creturn=[snapshot(c)for c in cones];save('actual-cone-constructor-RETURN.json',creturn)
 settings=clarabel.DefaultSettings();settings.max_iter=50;settings.time_limit=10.;settings.max_threads=1;settings.verbose=False
 emit(dict(kind='BEFORE_NEW_callback_toy_constructor'));solver=clarabel.DefaultSolver(P,q,A,b,cones,settings);save('actual-solver-constructor-RETURN.json',dict(type=type(solver).__name__,repr=repr(solver),initial_info=snapshot(solver.get_info())))
 returns=[]
 def callback(info):
  ret=snapshot(info);save(f'callback-{len(returns):03d}-actual-RETURN.json',ret);emit(dict(kind='AFTER_NEW_actual_callback_RETURN',actual_RETURN=ret));returns.append(ret);return False
 registration=solver.set_termination_callback(callback);save('actual-callback-registration-RETURN.json',dict(actual_RETURN=registration));emit(dict(kind='BEFORE_ONE_NEW_callback_toy_solve'))
 sol=solver.solve();vectors=O/'actual-toy-primal-dual-slack-RETURN.npz';np.savez_compressed(vectors,x=np.asarray(sol.x),z=np.asarray(sol.z),s=np.asarray(sol.s));vectors.chmod(0o600)
 with vectors.open('rb')as f:os.fsync(f.fileno())
 final=snapshot(sol,exclude=('x','z','s'));save('actual-toy-solver-RETURN.json',dict(**final,array_path=str(vectors),array_sha256=sha(vectors)));emit(dict(kind='AFTER_NEW_toy_solver_RETURN',actual_RETURN=final,arrays_sha256=sha(vectors)))
 x=np.asarray(sol.x);diagnostic=dict(status=str(sol.status),callback_count=len(returns),actual_x=x,analytic_x=.125,analytic_t=np.sqrt(5.),x_residual=float(x[0]-.125),squared_t_residual=float(x[1]*x[1]-5.))
 save('NEW-analytic-diagnostic-RETURN.json',diagnostic);assert str(sol.status)=='Solved'and len(returns)>0 and abs(x[0]-.125)<1e-8 and abs(x[1]*x[1]-5)<1e-6
 report=dict(schema='v2d.NEW-hard-training-callback-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEWanalyticcontracts=1,actual_callbacks=len(returns),every_callback_RETURN_durable_BEFORE_continue=True,no_reentrant_solver_or_reconstructed_internal_iterates=True,actual_cone_dimensions=[c.dim for c in cones],no_episode_source_old_control_or_candidate_gate_change=True)
 save('report.json',report);emit(dict(kind='AFTER_NEW_callback_contract_RETURN',actual_RETURN=report));print(json.dumps(report))

"""ONE NEW fixed direct all-row conic ADMM method; no adaptive parameters.

The actual returned terminal state is step32 ONLY. On a time/arithmetic/resource
stop, all intermediates remain unqualified and no alternate point is selected.
Every method intermediate RETURN is saved before subsequent work/assertions.
No implicit termination value is a source/native/physical certificate.
"""
from whole_row_conic_split_v1 import *
from scipy.linalg import cho_solve
from whole_row_conic_runtime_v1 import guarded_projection
import time
O=S/'e037-whole-row-conic-evaluation-v1';started=time.monotonic();journal=O/'NEW-method-journal.jsonl'
def emit(x):
 with journal.open('a') as f:f.write(json.dumps(normalized(dict(utc=now(),**x)),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
class MethodStop(Exception):pass
for p in [S/'e037-whole-row-conic-contracts-v1/report.json',S/'e037-whole-row-conic-binding-v1/report.json']:assert p.exists()
A,b,q,P,meta=load_frozen();nn=4613807;cones=meta['cones']
with np.load(N/'NEW-ALL-row-normal-RETURN.npz') as z:G=z['normal']
with np.load(N/'NEW-Cholesky-constructor-RETURN.npz') as z:factor=z['factor'];lower=bool(z['lower'])
declaration=dict(schema='v2d.ONE-NEW-whole-row-conic-method.v1',created_utc=now(),method='direct all-row conic ADMM',rho=1,relaxation=1,fixed_iterations=32,method_seconds=120,threads=1,outer_wall_seconds=240,outer_RSS_bytes=2147483648,disk_reserve_bytes=1610612736,full_rows=4618162,stored_coefficients=32306054,variables=1058,orthant_rows=nn,SOC4=900,SOC5=151,no_warmstart_restart_adaptation_tolerance_weight_step_sweep=True,start='F lifted zero coordinates/rho-r1; s=projection(b-Ax), dual zero; no source query',objective_variables_conditioning_UNCHANGED=True,normal_rho1='P + A.T A',normal_input_sha256=sha(N/'NEW-ALL-row-normal-RETURN.npz'),factor_input_sha256=sha(N/'NEW-Cholesky-constructor-RETURN.npz'),linear_solve_relative_guard=1e-9,linear_guard_NOT_original_candidate_tolerance=True,only_terminal_step32_eligible_for_raw_screen=True,actual_gate_protocol='raw terminal saved before screening; no approximate gate tolerance; actual integer/depth/all6motion/151training then ORIGINALfullpixelheldoutPOST and whole robot/native/self/swept/motor/grasp REQUIRED')
dump(O/'ONE-NEW-method-declaration-BEFORE.json',declaration)
# Actual constructor/input returns are distinct from old unavailable constructor.
dump(O/'NEW-actual-interface-constructor-RETURN.json',dict(rows=A.shape[0],columns=A.shape[1],stored_entries=A.nnz,cones=len(cones),normal_shape=list(G.shape),factor_shape=list(factor.shape),factor_lower=lower,normal_factor_saved=True,old_materialized_constructor_NOT_replayed=True))
x=np.zeros(1058);x[-1]=1.;save(O/'initial-NEW-x-BEFORE.npz',x=x)
Ax=A@x;save(O/'initial-NEW-Ax-RETURN.npz',Ax=Ax)
w=b-Ax;save(O/'initial-NEW-projection-input-RETURN.npz',w=w)
ret=guarded_projection(w,nn,cones);save(O/'initial-NEW-projection-RETURN.npz',s=ret['value']);dump(O/'initial-NEW-projection-meta-RETURN.json',{k:v for k,v in ret.items() if k!='value'})
u=np.zeros_like(b);save(O/'initial-NEW-dual-RETURN.npz',u=u)
assert ret['status']=='projected_ALL_product_coordinates';s=ret['value'];summaries=[]
try:
 for step in range(32):
  if time.monotonic()-started>120:raise MethodStop('declared120second_boundary; no terminal point')
  prefix=O/f'step-{step:03d}'
  emit(dict(kind='BEFORE_NEW_ALL_row_split_step',step=step,ALL_rows=A.shape[0]))
  term=b-s-u;save(str(prefix)+'-NEW-adjoint-input-RETURN.npz',term=term)
  at=A.T@term;save(str(prefix)+'-NEW-adjoint-RETURN.npz',ATterm=at)
  rhs=at-q;save(str(prefix)+'-NEW-normal-rhs-BEFORE.npz',rhs=rhs)
  x=cho_solve((factor,lower),rhs,overwrite_b=False,check_finite=True);save(str(prefix)+'-NEW-actual-linear-solve-RETURN.npz',x=x)
  lr=G@x-rhs;ls=abs(G)@abs(x)+abs(rhs);save(str(prefix)+'-NEW-linear-residual-RETURN.npz',residual=lr,scale=ls)
  linerr=float(np.max(abs(lr)/np.maximum(1.,ls)));dump(str(prefix)+'-NEW-linear-decision-RETURN.json',dict(relative_error=linerr,LOCALguard=1e-9,NOT_candidate_tolerance=True))
  if not np.isfinite(x).all() or linerr>1e-9:raise MethodStop('NEW raw linear solve arithmetic guard; no repair')
  Ax=A@x;save(str(prefix)+'-NEW-ALL-row-Ax-RETURN.npz',Ax=Ax)
  w=b-Ax-u;save(str(prefix)+'-NEW-ALL-cone-input-RETURN.npz',w=w)
  ret=guarded_projection(w,nn,cones);snew=ret['value'];save(str(prefix)+'-NEW-ALL-cone-projection-RETURN.npz',s=snew);dump(str(prefix)+'-NEW-ALL-cone-meta-RETURN.json',{k:v for k,v in ret.items() if k!='value'})
  if ret['status']!='projected_ALL_product_coordinates':raise MethodStop('NEW projection unknown; no partial model')
  unew=u+Ax+snew-b;save(str(prefix)+'-NEW-ALL-row-dual-RETURN.npz',u=unew)
  primal=float(np.max(abs(unew-u)));dualcones=cone_diagnostics(unew,nn,cones);save(str(prefix)+'-NEW-dual-cone-diagnostic-RETURN.npz',values=dualcones)
  obj=float(np.dot(q,x)+.5*np.dot(x,P@x));summary=dict(step=step,objective=obj,max_primal_residual=primal,min_dual_cone_diagnostic=float(dualcones.min()),linear_relative_error=linerr,wall_seconds=time.monotonic()-started)
  dump(str(prefix)+'-NEW-step-summary-RETURN.json',summary);emit(dict(kind='AFTER_NEW_complete_split_step_RETURN',**summary))
  assert np.isfinite(Ax).all() and np.isfinite(snew).all() and np.isfinite(unew).all() and np.isfinite(obj)
  summaries.append(summary);u=unew;s=snew
 save(O/'ONE-actual-raw-terminal-RETURN.npz',x=x,s=s,u=u)
 dump(O/'ONE-actual-method-RETURN.json',dict(status='UNRESOLVED_fixed32step_cap',steps=32,wall_seconds=time.monotonic()-started,no_convergence_or_optimality_claim=True,terminal_only=True))
 # Closed-model raw screen uses NO clipping/repair/original gate tolerance.
 raw=b-Ax;save(O/'ONE-raw-terminal-ALL-model-residual-RETURN.npz',residual=raw)
 conev=cone_diagnostics(raw,nn,cones);save(O/'ONE-raw-terminal-model-cone-RETURN.npz',values=conev)
 screen=bool(np.isfinite(raw).all() and np.all(conev>=0))
 report=dict(schema='v2d.NEW-whole-row-conic-evaluation.v1',scope_status='UNRESOLVED_fixed32step_cap',steps=32,wall_seconds=time.monotonic()-started,raw_model_screen=screen,raw_model_min_affine=float(raw[:nn].min()),raw_model_min_cone=float(conev[1:].min()),raw_terminal_objective=summaries[-1]['objective'],raw_primal_residual=summaries[-1]['max_primal_residual'],actual_source_queries=0,no_reference_or_physical_qualification=True)
 dump(O/'NEW-raw-terminal-screen-RETURN.json',report)
 if screen:
  from postcheck_whole_row_conic_candidate_v1 import postcheck
  actual=postcheck(O,x,emit);dump(O/'ONE-NEW-actual-POST-RETURN.json',actual);report['actual_post']=actual;report['actual_source_queries']=actual['actual_training_queries']
 else:report['candidate_status']='REJECTED_raw_numeric_closed_model; no actual pose/source query'
 dump(O/'report.json',report)
except MethodStop as error:
 emit(dict(kind='NEW_method_stop_RETURN',reason=str(error)))
 dump(O/'report.json',dict(schema='v2d.NEW-whole-row-conic-evaluation.v1',scope_status='UNRESOLVED_declared_method_boundary',reason=str(error),completed_steps=len(summaries),wall_seconds=time.monotonic()-started,no_terminal_selected=True,actual_source_queries=0,no_qualification=True))

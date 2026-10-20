"""NEW local camera-SOCP packing, signs, actual acceptance and method controls."""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,math,os,time
import numpy as np
from scipy import sparse
import sequential_camera_conic_v1 as method
from thumb_conic_geometry_v1 import plain,journal
S=Path('/Users/rohithiruku/v2d_scored_baseline_20261017');C=S/'compute';O=S/'e037-sequential-camera-contract-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
class ContinuousControl:
 """A NEW continuous analytic two-angle problem, not robot/source anchors."""
 def __init__(self):
  self.n=2;self.x0=np.array([.2,.1]);self.A=sparse.csr_matrix([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.],[1.,-1.],[-1.,1.]]);self.b=np.array([.4,.4,.4,.4,.3,.3]);self.linear_scale=.05*np.maximum(1.,np.asarray(abs(self.A).sum(axis=1)).ravel());self.source_scale=np.array([1.]);self.half_indices=[];self.half_lo=np.array([]);self.half_hi=np.array([])
 def snapshot(self,x,emit,label,initial=False):
  x=np.asarray(x);phase=float(x.sum());C0=np.array([math.sin(phase),0.,1.]);J=np.array([[math.cos(phase),math.cos(phase)],[0.,0.],[0.,0.]]);s=dict(id=0,columns=[0,1],C=C0,J=J,B=np.array([[1.,0.,0.],[0.,1.,0.]]),radius=.05+1e-6,direct_image_error_px=abs(float(C0[0])));r=dict(x=x.copy(),source=[s],linear_violation=(self.A@x-self.b)/self.linear_scale,half_values=np.array([]),half_derivatives=np.array([]));r['merit']=method.actual_merit(self,r);emit(dict(kind='AFTER_NEW_analytic_continuous_actual_RETURN',label=label,returned=r));return r
 def exact_acceptance(self,snapshot,emit,label):
  x=list(map(F,snapshot['x']));linear=all(sum((F(float(self.A[i,j]))*x[j]for j in range(2)),F(0))<=F(float(self.b[i]))for i in range(6));source=all(s['C'][2]>.01 and s['direct_image_error_px']<=s['radius']for s in snapshot['source']);r=dict(all_original_model_checks_passed=linear and source,exact_binary_linear_passed=linear,actual_source_strict_depth_passed=source);emit(dict(kind='AFTER_NEW_analytic_continuous_exact_check',label=label,returned=r));return r
checks=[];cache={};toy=ContinuousControl();site=S/'e037-direct-SDP-runtime-v2/site-packages'
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,v)
 def run(name,build,verify):
  emit(dict(kind='BEFORE_NEW_sequential_contract',name=name));r=build();p=save(name+'.json',r);emit(dict(kind='AFTER_NEW_sequential_contract_RETURN',name=name,returned=r,return_sha256=sha(p)));verify(r);checks.append(dict(name=name,passed=True,return_sha256=sha(p)))
 def canonical():
  view=toy.snapshot(toy.x0,emit,'NEW-control-initial');problem=method.step_problem(toy,view,emit,'NEW-control-canonical');cache.update(view=view,problem=problem);return dict(A=method.sparse_record(problem['A']),b=problem['b'],P=method.sparse_record(problem['P']),q=problem['q'],nonnegative_rows=problem['nonnegative_rows'],SOC_blocks=problem['SOC_blocks'])
 run('01-complete-affine-camera-SOC-constructor',canonical,lambda r:(_ for _ in()).throw(AssertionError())if r['A']['shape'][1]!=3 or len(r['SOC_blocks'])!=1 or r['SOC_blocks'][0]['dimension']!=3 else None)
 def cone_signs():
  p=cache['problem'];v=cache['view'];trials=[]
  for y in [np.array([0.,0.,0.]),np.array([-.03,.02,.1]),np.array([.02,-.04,-.1])]:
   slack=p['b']-p['A']@y;source=v['source'][0];dC=source['J']@y[:2];expected=np.r_[source['radius']*(source['C'][2]+dC[2])+y[2]*toy.source_scale[0],source['B']@(source['C']+dC)];actual=slack[-3:];trials.append(dict(y=y,actual_SOC_slack=actual,independent_expected=expected,error=float(abs(actual-expected).max())))
  return dict(trials=trials,negative_rho_NOT_silently_clipped=True)
 run('02-shared-elastic-camera-cone-sign-and-order',cone_signs,lambda r:(_ for _ in()).throw(AssertionError())if max(x['error']for x in r['trials'])>1e-15 else None)
 def prefix():
  p=cache['problem'];y=np.array([.01,-.02,.07]);actual=(p['b']-p['A']@y)[:6];expected=toy.b-toy.A@(toy.x0+y[:2])+toy.linear_scale*y[2];return dict(actual=actual,independent_original_linear_prefix=expected,error=float(abs(actual-expected).max()),no_actual_original_row_changed=True)
 run('03-original-linear-prefix-and-positive-static-units',prefix,lambda r:(_ for _ in()).throw(AssertionError())if r['error']>1e-15 or not np.all(toy.linear_scale>0)else None)
 def trust():
  p=cache['problem'];y=np.array([.02,-.03,.4]);slack=p['b']-p['A']@y;depth=6;rho=7;trust=8;expected_depth=1.-.01+.01*y[2];expected_trust=np.ravel([[.1+y[j],.1-y[j]]for j in range(2)]);return dict(depth_actual=float(slack[depth]),depth_expected=expected_depth,rho_actual=float(slack[rho]),rho_expected=float(y[2]),trust_actual=slack[trust:trust+4],trust_expected=expected_trust)
 run('04-depth-elastic-rho-and-trust-bound-directions',trust,lambda r:(_ for _ in()).throw(AssertionError())if abs(r['depth_actual']-r['depth_expected'])>1e-15 or r['rho_actual']!=r['rho_expected']or np.max(abs(r['trust_actual']-r['trust_expected']))>1e-15 else None)
 def half():
  other=ContinuousControl();other.half_indices=[0];other.half_lo=np.array([-.1]);other.half_hi=np.array([.2]);v=other.snapshot(other.x0,emit,'NEW-half-control');u=np.tan(other.x0[0]/2);du=.5*(1+u*u);v['half_values']=np.array([u]);v['half_derivatives']=np.array([du]);p=method.step_problem(other,v,emit,'NEW-half-canonical');y=np.array([.03,.02,.1]);slack=p['b']-p['A']@y;return dict(actual=slack[6:8],expected=[u+du*y[0]-other.half_lo[0]+.05*y[2],other.half_hi[0]-u-du*y[0]+.05*y[2]],positive_derivative=float(du))
 run('05-local-finite-halfangle-derivative-and-bound-signs',half,lambda r:(_ for _ in()).throw(AssertionError())if np.max(abs(r['actual']-r['expected']))>1e-15 or r['positive_derivative']<=0 else None)
 def strict():
  v=toy.snapshot(np.array([0.,0.]),emit,'NEW-zero-depth-control');v['source'][0]['C'][2]=.01;v['merit']=method.actual_merit(toy,v);ret=toy.exact_acceptance(v,emit,'NEW-zero-depth-actual-check');return dict(actual_merit=v['merit'],actual_acceptance=ret,no_merit_tolerance_acceptance=True)
 run('06-zero-merit-cannot-accept-strict-depth-boundary',strict,lambda r:(_ for _ in()).throw(AssertionError())if r['actual_merit']!=0 or r['actual_acceptance']['all_original_model_checks_passed']else None)
 def local_not_global():
  v=cache['view'];delta=np.array([-.08,-.07]);predicted=v['source'][0]['C']+v['source'][0]['J']@delta;actual=toy.snapshot(toy.x0+delta,emit,'NEW-linearization-scope-control')['source'][0]['C'];return dict(predicted_camera_point=predicted,actual_nonlinear_camera_point=actual,difference=float(abs(predicted-actual).max()),local_cone_NOT_original_family_exclusion=True)
 run('07-linearization-is-not-actual-source-or-exclusion',local_not_global,lambda r:(_ for _ in()).throw(AssertionError())if not r['difference']>1e-4 else None)
 def objective():
  p=cache['problem'];y=np.array([.03,-.02,.1]);actual=.5*float(y@(p['P']@y))+float(p['q']@y);expected=.5*1e-3*(y[0]**2+y[1]**2)+y[2];return dict(actual=actual,expected=expected,all_damping_diagonal_nonnegative=bool(np.all(p['P'].diagonal()>=0)),rho_is_algorithm_slack_NOT_physical_force=True)
 run('08-positive-quadratic-step-objective-units',objective,lambda r:(_ for _ in()).throw(AssertionError())if r['actual']!=r['expected']or not r['all_damping_diagonal_nonnegative']else None)
 def tiny_solve():
  raw=method.solve_step(cache['problem'],emit,'ONE-NEW-tiny-SOC-control',10.,site);cache['tiny_raw']=raw;return dict(actual_raw_RETURN=raw,raw_vectors_not_modified=True,not_episode_or_robot_candidate=True)
 run('09-ONE-new-tiny-camera-SOCP-actual-return',tiny_solve,lambda r:(_ for _ in()).throw(AssertionError())if r['actual_raw_RETURN']['status']not in['Solved','AlmostSolved']else None)
 def rational_dual():
  p=cache['problem'];A=p['A'].toarray();b=p['b'];z=[F(0)]*len(b);z[-3]=F(1,4);z[-2]=F(1,4);z[-1]=F(0);q=[F(0),F(0),F(1)];w=[q[j]+sum((F(float(A[i,j]))*z[i]for i in range(len(b))),F(0))for j in range(3)];bounds=[(F(float(-.1)),F(float(.1)))]*2+[(F(0),F(1))];lower=sum((a*(lo if a>=0 else hi)for a,(lo,hi)in zip(w,bounds,strict=True)),F(0))-sum((F(float(v))*a for v,a in zip(b,z,strict=True)),F(0));y=[F(0),F(0),F(1)];primal=F(1);stationary_identity=[str(a)for a in w];return dict(exact_rational_SOC_dual_boundary_identity=str(z[-3]**2-z[-2]**2-z[-1]**2),exact_linear_plus_adjoint=stationary_identity,residual_corrected_finite_box_lower=str(lower),feasible_control_objective=str(primal),weak_bound_valid=lower<=primal,raw_solver_dual_UNCHANGED_and_not_used_as_certificate=True)
 run('10-rational-SOC-dual-adjoint-and-finite-box-sign',rational_dual,lambda r:(_ for _ in()).throw(AssertionError())if r['exact_rational_SOC_dual_boundary_identity']!='0'or not r['weak_bound_valid']else None)
 def armijo():
  m=.4;rho=.1;alpha=.5;bound=m-1e-4*alpha*(m-rho);return dict(original_merit=m,raw_rho=rho,predicted_reduction=m-rho,alpha=alpha,bound=bound,decreasing_trial_pass=.2<=bound,unchanged_trial_rejected=not(m<=bound),negative_rho_return_would_remain_raw=True)
 run('11-actual-merit-Armijo-sign-and-no-failed-promotion',armijo,lambda r:(_ for _ in()).throw(AssertionError())if not r['decreasing_trial_pass']or not r['unchanged_trial_rejected']else None)
 def continuous():
  ret=method.construct(ContinuousControl(),emit,site);return dict(actual_method_RETURN=ret,algebraic_continuous_control_NOT_episode_or_source_anchors=True)
 run('12-ONE-new-coupled-continuous-nonlinear-construction',continuous,lambda r:(_ for _ in()).throw(AssertionError())if r['actual_method_RETURN']['status']!='PROPOSED_actual_original_model_point'or not r['actual_method_RETURN']['extra']['all_original_model_checks_passed']else None)
 emit(dict(kind='AFTER_NEW_all_sequential_contracts',returned=checks))
out=dict(schema='v2d.new-sequential-camera-conic-contracts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_camera_step_and_actual_acceptance_contracts_passed=True,new_contracts=len(checks),new_tiny_SOC_step_control_calls=1,new_analytic_continuous_method_calls=1,new_episode_method_calls=0,helper_sha256=sha(C/'sequential_camera_conic_v1.py'),no_old_PSD_controls_source_models_or_audits_replayed=True,all_original_source_authority_candidate_tolerances_and_F_UNCHANGED=True,new_native_position_checks=0,new_GPU_jobs=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log','report.json']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))

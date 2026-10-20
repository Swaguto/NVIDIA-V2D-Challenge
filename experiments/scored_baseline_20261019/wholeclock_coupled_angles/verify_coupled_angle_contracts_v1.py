"""NEW angle/URDF/source derivative and original temporal-row contracts only."""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,math,os,time
import numpy as np
from scipy.spatial.transform import Rotation
from coupled_angle_source_v1 import Chain,camera_source_factor,linear_angle_rows,unrat
from thumb_conic_geometry_v1 import plain,journal
S=Path('/Users/rohithiruku/v2d_scored_baseline_20261017');C=S/'compute';O=S/'e037-coupled-angle-contract-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
xml='''<robot name="proof-control"><link name="base"/><link name="a"/><link name="b"/><link name="tip"/><link name="other"/>
<joint name="qa" type="revolute"><parent link="base"/><child link="a"/><origin xyz=".1 0 0" rpy="0 0 .2"/><axis xyz="0 0 1"/></joint>
<joint name="qb" type="revolute"><parent link="a"/><child link="b"/><origin xyz=".4 0 0"/><axis xyz="0 0 1"/></joint>
<joint name="end" type="fixed"><parent link="b"/><child link="tip"/><origin xyz=".3 0 0"/></joint>
<joint name="qc" type="revolute"><parent link="base"/><child link="other"/><axis xyz="1 0 0"/></joint></robot>'''
chain=Chain(xml,['qb','qc','qa']);q=np.array([-.17,.23,.31]);rp=np.r_[[.11,.12,.7],Rotation.from_euler('x',.4).as_quat()];cache={};checks=[]
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,v)
 def run(name,build,verify):
  emit(dict(kind='BEFORE_NEW_analytic_contract',name=name));r=build();p=save(name+'.json',r);emit(dict(kind='AFTER_NEW_analytic_contract_RETURN',name=name,returned=r,return_sha256=sha(p)));verify(r);checks.append(dict(name=name,passed=True,return_sha256=sha(p)))
 def pose_control():
  actual=chain.point_jacobian(rp,q,'tip');a=.2+q[2];b=a+q[0];R=Rotation.from_quat(rp[3:]).as_matrix();local=np.array([.1+.4*math.cos(a)+.3*math.cos(b),.4*math.sin(a)+.3*math.sin(b),0.]);p=rp[:3]+R@local;J=np.zeros((3,3));J[:,0]=R@np.array([-.3*math.sin(b),.3*math.cos(b),0]);J[:,2]=R@np.array([-.4*math.sin(a)-.3*math.sin(b),.4*math.cos(a)+.3*math.cos(b),0]);cache['pose']=actual;return dict(actual=actual,expected_point=p,expected_J=J,point_error=float(abs(actual['point']-p).max()),jacobian_error=float(abs(actual['jacobian']-J).max()))
 run('01-noncommuting-root-origin-and-ordered-joint-derivative',pose_control,lambda r:(_ for _ in()).throw(AssertionError())if r['point_error']>1e-14 or r['jacobian_error']>1e-14 else None)
 def unrelated():
  r=cache['pose'];base=chain.point_jacobian(rp,q,'base');return dict(unrelated_column=r['jacobian'][:,1],base_J=base['jacobian'],ancestor_names=r['ancestor_joint_names'],explicit_joint_order=chain.names)
 run('02-branch-independence-and-explicit-axis-order',unrelated,lambda r:(_ for _ in()).throw(AssertionError())if np.any(r['unrelated_column'])or np.any(r['base_J'])or r['ancestor_names']!=['qa','qb','end']else None)
 cam=np.eye(4);cam[:3,:3]=Rotation.from_euler('zy',[-.11,.07]).as_matrix();cam[:3,3]=[.02,-.01,-.3];stage=Rotation.from_euler('yx',[.08,-.13]).as_matrix();K=np.array([[510.,0,320.],[0,495.,240.],[0,0,1.]]);pixel=np.array([410.,260.])
 def source_control():
  r=cache['pose'];s=camera_source_factor(r['point'],r['jacobian'],cam,stage,K,pixel,12.);a=.2+q[2];b=a+q[0];worldR=Rotation.from_quat(rp[3:]).as_matrix()
  def f(x):
   a=.2+x[2];b=a+x[0];p=rp[:3]+worldR@np.array([.1+.4*math.cos(a)+.3*math.cos(b),.4*math.sin(a)+.3*math.sin(b),0.]);c=(p@stage-cam[:3,3])@cam[:3,:3];u=K[0,0]*c[0]+(K[0,2]-pixel[0])*c[2];v=K[1,1]*c[1]+(K[1,2]-pixel[1])*c[2];return (12.+1e-6)**2*c[2]**2-u*u-v*v
  h=1e-4;numeric=[]
  for i in range(3):
   e=np.eye(3)[i]*h;numeric.append((f(q-2*e)-8*f(q-e)+8*f(q+e)-f(q+2*e))/(12*h))
  cache['source']=s;return dict(actual=s,independent_scalar=f(q),independent_four_point_derivative=numeric,scalar_error=abs(s['source_cone_value']-f(q)),jacobian_error=float(abs(s['source_cone_jacobian']-numeric).max()))
 run('03-squared-source-cone-shared-joint-camera-Jacobian',source_control,lambda r:(_ for _ in()).throw(AssertionError())if r['scalar_error']>1e-8 or r['jacobian_error']>2e-6 else None)
 def cone_signs():
  result=[];K0=np.eye(3)
  for z in[1.,0.,-1.]:
   s=camera_source_factor([0.,0.,z],np.eye(3),np.eye(4),np.eye(3),K0,[0,0],.2);result.append(s)
  return dict(cases=result,zero_depth_squared_cone_is_not_a_source_pass=True)
 run('04-positive-zero-negative-depth-is-separate',cone_signs,lambda r:(_ for _ in()).throw(AssertionError())if not(r['cases'][0]['strict_depth_value']>0 and r['cases'][1]['strict_depth_value']<0 and r['cases'][2]['strict_depth_value']<0 and r['cases'][2]['source_cone_value']>0 and r['cases'][2]['direct_image_error_px']is None)else None)
 def strict_boundary():
  a=camera_source_factor([0.,0.,.01],np.eye(3),np.eye(4),np.eye(3),np.eye(3),[0,0],0.);b=camera_source_factor([1e-6,0,1.],np.eye(3),np.eye(4),np.eye(3),np.eye(3),[0,0],0.);return dict(exact_depth_boundary=a,original_tolerance_boundary=b)
 run('05-original-image-tolerance-and-strict-depth-boundary',strict_boundary,lambda r:(_ for _ in()).throw(AssertionError())if r['exact_depth_boundary']['strict_depth_value']!=0 or r['original_tolerance_boundary']['constructed_radius_px']!=1e-6 or r['original_tolerance_boundary']['direct_image_error_px']!=1e-6 else None)
 def zero_residual():
  s=camera_source_factor([0,0,1],np.eye(3),np.eye(4),np.eye(3),np.eye(3),[0,0],0.);return dict(returned=s,finite_gradient=bool(np.isfinite(s['source_cone_jacobian']).all()),nonzero_tiny_z_derivative=float(s['source_cone_jacobian'][2]))
 run('06-zero-image-residual-retains-tiny-cone-terms',zero_residual,lambda r:(_ for _ in()).throw(AssertionError())if not r['finite_gradient']or r['nonzero_tiny_z_derivative']<=0 else None)
 original=np.array([[.1],[.12],[.11],[.14]]);fixed=np.array([[.1],[.121],[.109],[.14]]);ts=[0.,.05,.11,.17];limits={'joint':[-1.,1.,2.]};free={(1,'joint'),(2,'joint')}
 def temporal():
  rows=linear_angle_rows(original,fixed,ts,['joint'],limits,free,emit);cache['rows']=rows;errors=[]
  for r in rows:
   total=sum((unrat(t['coefficient'])*F(float(fixed[t['frame'],0]))for t in r['terms']),F(0));folded=sum((unrat(t['coefficient'])*F(float(fixed[t['frame'],0]))for t in r['active_terms']),F(0));errors.append(total-unrat(r['upper'])-(folded-unrat(r['folded_upper'])))
  return dict(rows=rows,exact_fold_residuals=[str(v)for v in errors],all_exact=all(v==0 for v in errors))
 run('07-whole-clock-angle-rows-exact-fixed-boundary-folding',temporal,lambda r:(_ for _ in()).throw(AssertionError())if len(r['rows'])!=32 or not r['all_exact']else None)
 def second():
  rows=[r for r in cache['rows']if r['kind']=='original_second_added'];passed=[]
  for r in rows:
   sign=r['origin']['sign'];t=r['origin']['clocks'][0];expr=sign*(F(float(fixed[t,0]))-2*F(float(fixed[t-1,0]))+F(float(fixed[t-2,0])));base=sign*(F(float(original[t,0]))-2*F(float(original[t-1,0]))+F(float(original[t-2,0])));passed.append(expr-unrat(r['upper'])==expr-base-F(float(.05))-F(float(1e-7)))
  return dict(rows=rows,minus_two_and_original_delta_identities=passed)
 run('08-second-added-difference-shared-middle-angle',second,lambda r:(_ for _ in()).throw(AssertionError())if not all(r['minus_two_and_original_delta_identities'])else None)
 def velocity():
  rows=[r for r in cache['rows']if r['kind']=='actual_URDF_velocity'];passed=[]
  for r in rows:
   t=r['origin']['clocks'][0];passed.append(unrat(r['upper'])==(F(2)+F(float(1e-7)))*(F(ts[t])-F(ts[t-1])))
  return dict(rows=rows,velocity_tolerance_is_rad_per_second_NOT_angle=passed)
 run('09-nonuniform-actual-timestamps-and-tolerance-units',velocity,lambda r:(_ for _ in()).throw(AssertionError())if not all(r['velocity_tolerance_is_rad_per_second_NOT_angle'])else None)
 def constant_failure():
  f=fixed.copy();f[3]=5.;rows=linear_angle_rows(original,f,ts,['joint'],limits,free,emit);bad=[r for r in rows if r['constant_only']and not r['constant_only_passed']];return dict(all_fixed_bad_rows_retained=bad)
 run('10-infeasible-immutable-boundary-is-retained',constant_failure,lambda r:(_ for _ in()).throw(AssertionError())if not r['all_fixed_bad_rows_retained']else None)
 def invalid():
  calls=[lambda:chain.point_jacobian(np.zeros(7),q,'tip'),lambda:chain.point_jacobian(rp,[0,1],'tip'),lambda:camera_source_factor([0,0,1],np.eye(3),np.eye(4),np.eye(3),K,[np.nan,0],1),lambda:linear_angle_rows(original,fixed,[0,.05,.04,.17],['joint'],limits,free,emit),lambda:linear_angle_rows(original,fixed,ts,['joint'],limits,{(4,'joint')},emit),lambda:Chain(xml.replace('xyz="0 0 1"','xyz="0 0 2"'),['qb','qc','qa'])];returned=[]
  for call in calls:
   try:call();returned.append('UNEXPECTED_acceptance')
   except Exception as ex:returned.append(type(ex).__name__)
  return dict(actual_returns=returned)
 run('11-missing-nonfinite-pole-clock-and-axis-failures-explicit',invalid,lambda r:(_ for _ in()).throw(AssertionError())if r['actual_returns']!=['ValueError']*6 else None)
 def signs():
  # A joint source-cone directional derivative: BOTH angles contribute to one point.
  s=cache['source'];direction=np.array([.2,0.,-.3]);J=s['camera_jacobian'];C0=s['camera_point'];B=np.array([[K[0,0],0.,K[0,2]-pixel[0]],[0.,K[1,1],K[1,2]-pixel[1]]]);dC=J@direction;d=2*(12.+1e-6)**2*C0[2]*dC[2]-2*float((B@C0)@(B@dC));actual=float(s['source_cone_jacobian']@direction);return dict(joint_direction=direction,independent_directional=d,actual_directional=actual,error=abs(d-actual),no_independent_copy_of_shared_joint=True)
 run('12-coupled-directional-source-cone-identity',signs,lambda r:(_ for _ in()).throw(AssertionError())if r['error']>1e-9 else None)
 emit(dict(kind='AFTER_NEW_all_contracts',returned=checks))
out=dict(schema='v2d.coupled-angle-new-analytic-contracts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_angle_source_temporal_contracts_passed=True,new_analytic_contracts=len(checks),helper_sha256=sha(C/'coupled_angle_source_v1.py'),new_episode_model_evaluations=0,new_solver_calls=0,no_old_source_polynomial_time_algebra_controls_or_audits_replayed=True,source_human_authority_and_original_candidate_tolerances_UNCHANGED=True,new_native_position_checks=0,new_GPU_jobs=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log','report.json']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))

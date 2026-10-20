"""ONE NEW SLSQP all-row method, frozen canonical, F start, no retry.

All original residual rows are evaluated at every unique optimizer input. Min
rows merely give a compact exact real constraint expression, never a cut bank.
Averaged tie slopes / zero-norm cone slopes serve SEARCH only. There is no
nonlinear stationarity or exact feasibility inference from SLSQP success.
"""
from structured_all_rows_v1 import *
from scipy.optimize import minimize
import time
O=S/'e037-structured-all-row-evaluation-v1'
start=time.monotonic();journal=O/'NEW-method-journal.jsonl'
def emit(x):
 with journal.open('a') as f:f.write(json.dumps(normjson(dict(utc=now(),**x)),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
class Limit(Exception):pass
for p in [S/'e037-structured-all-row-contracts-v1/report.json',S/'e037-structured-all-row-binding-v1/report.json']:
 assert p.exists()
A,b,q,P,groups,cones,meta=load_interface()
declaration=dict(schema='v2d.ONE-NEW-structured-all-row-method.v1',created_utc=now(),method='installed SciPy1.16.3 SLSQP',start='zero lifted coordinates, phase rho/r=1; F only; no source queries at start',maximum_iterations=20,ftol=1e-6,maximum_unique_queries=48,method_seconds=150,threads=1,outer_wall_seconds=240,outer_RSS_bytes=2147483648,all_rows_every_unique_query=4618162,affine_min_groups=len(groups),cones=len(cones),variables=1058,old_objective_and_conditioning_UNCHANGED=True,no_restart_warmstart_or_repair=True,terminal_selection='SOLE actual OptimizeResult.x only; no point on limit/exception',actual_source_gate_protocol='raw returned point sealed before actual gates; all affine/cone residuals must be nonnegative with NO tolerance repair, then nonlinear cells/strict depths, 6 original motion gates, actual all151 training; original fullpixel/heldout POST-only',input_layout_sha256=sha(PREFLIGHT/'NEW-structured-layout-RETURN.json'))
dump(O/'ONE-NEW-method-declaration-BEFORE.json',declaration)
cache_x=None;cache_return=None;count=0;callback_count=0

def ensure(x):
 global cache_x,cache_return,count
 x=np.asarray(x,float)
 if cache_x is not None and np.array_equal(x,cache_x):
  emit(dict(kind='actual_cache_reuse',query=count-1,input_sha256=hashlib.sha256(x.tobytes()).hexdigest()));return cache_return
 if count>=48 or time.monotonic()-start>150:raise Limit('declared query/time bound; no returned candidate')
 k=count;count+=1
 arrays(O/f'query-{k:03d}-NEW-input-BEFORE.npz',x=x)
 emit(dict(kind='BEFORE_ALL_rows_NEW_query',query=k,rows=4618162))
 ret=reduce_all(A,b,x,groups,cones)
 arrays(O/f'query-{k:03d}-NEW-ALL-row-RETURN.npz',**{j:v for j,v in ret.items() if j!='unknown'})
 dump(O/f'query-{k:03d}-NEW-unknown-RETURN.json',ret['unknown'])
 emit(dict(kind='AFTER_ALL_rows_NEW_RETURN',query=k,rows=len(ret['raw_rows']),groups=len(ret['values']),unknown_groups=len(ret['unknown'])))
 assert np.isfinite(x).all() and np.isfinite(ret['raw_rows']).all() and np.isfinite(ret['values']).all() and np.isfinite(ret['search_J']).all()
 cache_x=x.copy();cache_return=ret;return ret

def fun(x):
 # Query and seal all constraints even if the solver asks objective first.
 ensure(x);value=float(q@x+.5*x@(P@x));emit(dict(kind='actual_objective_RETURN',query=count-1,value=value));return value

def grad(x):
 ensure(x);value=q+P@x;arrays(O/f'query-{count-1:03d}-NEW-objective-J-RETURN.npz',J=value);return value

def cf(x):return ensure(x)['values']
def cj(x):return ensure(x)['search_J']
def callback(x):
 global callback_count
 arrays(O/f'callback-{callback_count:03d}-NEW-actual-RETURN.npz',x=x)
 emit(dict(kind='actual_SLSQP_callback_RETURN',callback=callback_count));callback_count+=1
 if time.monotonic()-start>150:raise Limit('declared 150second boundary; no returned candidate')

x0=np.zeros(1058);x0[-1]=1.;arrays(O/'ONE-F-start-BEFORE.npz',x=x0)
emit(dict(kind='BEFORE_ONE_NEW_SLSQP_call',declared_rows=4618162))
try:
 res=minimize(fun,x0,jac=grad,method='SLSQP',constraints=[dict(type='ineq',fun=cf,jac=cj)],callback=callback,options=dict(maxiter=20,ftol=1e-6,disp=False))
 raw={k:v for k,v in res.items() if isinstance(v,np.ndarray)}
 arrays(O/'ONE-actual-OptimizeResult-RETURN.npz',**raw)
 dump(O/'ONE-actual-OptimizeResult-metadata-RETURN.json',{k:v for k,v in res.items() if not isinstance(v,np.ndarray)})
 emit(dict(kind='AFTER_ONE_NEW_SLSQP_actual_RETURN',status=int(res.status),success=bool(res.success)))
 # No alternate or intermediate selection. This raw returned vector alone.
 x=np.asarray(res.x);terminal=ensure(x)
 final=dict(schema='v2d.NEW-structured-all-row-evaluation.v1',optimizer_status=int(res.status),optimizer_success=bool(res.success),optimizer_message=str(res.message),iterations=int(res.nit),unique_queries=count,callbacks=callback_count,wall_seconds=time.monotonic()-start,raw_original_min_affine=float(terminal['raw_rows'][:4613807].min()),raw_min_whole_cone=float(terminal['values'][len(groups):].min()),all_model_rows_numerically_nonnegative=bool(np.all(terminal['values']>=0)),actual_source_queries=0,no_qualification=True)
 # Numerical proposal screens do not inherit old certificates.
 if not np.all(terminal['values']>=0):
  final['scope_status']='REJECTED_raw_numeric_model_rows';dump(O/'report.json',final)
 else:
  # Decode new nonlinear pose and seal BEFORE original motion/cell/source checks.
  from full_pose_inputs_v1 import load
  from full_pose_construction_v1 import actual_motion
  from object_full_pose_maps_v1 import compose
  from full_pose_training_oracle_v1 import FullPoseTrainingOracle
  d=load();m=json.loads((MODEL/'NEW-canonical-metadata-RETURN.json').read_text());r=m['radius'];scale=m['scale']
  local=x[:1057].reshape(151,7);t=local[:,3:6]/scale;D=1+np.einsum('ij,ij->i',t,t)
  delta=r*local[:,:3]/D[:,None];length=np.linalg.norm(t,axis=1);theta=np.zeros_like(t)
  nz=length!=0;theta[nz]=t[nz]*(2*np.arctan(length[nz])/length[nz])[:,None]
  poses=np.array([compose(d['world'][f],delta[f],theta[f],np.eye(3)) for f in range(151)])
  arrays(O/'ONE-NEW-raw-sealed-proposal-BEFORE.npz',raw_x=x,delta=delta,theta=theta,world_from_object=poses)
  pid=sha(O/'ONE-NEW-raw-sealed-proposal-BEFORE.npz');emit(dict(kind='NEW_actual_raw_pose_sealed',proposal_id=pid))
  motion=actual_motion(poses,d['times'],d['motion_caps']);arrays(O/'ONE-NEW-actual-motion-RETURN.npz',**motion['arrays']);dump(O/'ONE-NEW-actual-motion-RETURN.json',{k:v for k,v in motion.items() if k!='arrays'})
  cellpass=True;depthpass=True;cells=[]
  for f in range(151):
   cam=compose(d['camera'][f],delta[f],theta[f],d['Rcw'][f]);vertices=d['V']@cam[:3,:3].T+cam[:3,3]
   finite=bool(np.isfinite(vertices).all());positive=bool(finite and np.all(vertices[:,2]>.01) and cam[2,3]>.05)
   with np.load(MODEL/f'frame-{f:03d}-NEW-hard-cell-coefficients-RETURN.npz') as z:target=z['target_integer_uv']
   if positive:
    uv=np.column_stack([d['K'][0,0]*vertices[:,0]/vertices[:,2]+d['K'][0,2],d['K'][1,1]*vertices[:,1]/vertices[:,2]+d['K'][1,2]])
    integers=np.clip(uv,-20000,20000).astype(np.int32);same=bool(np.array_equal(integers,target))
   else:uv=np.empty((0,2));integers=np.empty((0,2),np.int32);same=False
   arrays(O/f'f{f:03d}-NEW-actual-cell-depth-RETURN.npz',camera_from_object=cam,vertices=vertices,uv=uv,integers=integers)
   cells.append(dict(frame=f,finite=finite,strict_depth=positive,cells_identical=same));cellpass &= same;depthpass &= positive
  dump(O/'ONE-NEW-actual-cell-depth-decisions-RETURN.json',cells)
  final.update(actual_motion_passes=motion['passes'],actual_integer_cells=cellpass,actual_strict_depth=depthpass)
  if cellpass and depthpass and all(motion['passes']):
   def save(name,values):
    p=O/(name+'.npz');arrays(p,**values);return dict(path=str(p),sha256=sha(p))
   def training(f):
    base=d['stats'][f]
    with np.load(base['partition']['path']) as z:mask=z['training_mask'];domain=z['domain']
    with np.load(base['point_data']['path']) as z:points=z['points_C_m']
    return dict(training_mask=mask,domain=domain,points_C_m=points)
   oracle=FullPoseTrainingOracle(d,training,emit,save)
   source=[oracle.query(f,delta[f],theta[f],pid) for f in range(151)]
   dump(O/'ONE-NEW-actual151-training-RETURN.json',source);final['actual_source_queries']=151
   passed=acceptance_identity(pid,[z['proposal_id'] for z in source],motion['passes'],cellpass,depthpass,True) and all(z['passes_TRAINING_construction'] for z in source)
   final['training_pass']=passed;final['scope_status']='TRAINING_ONLY_pass_requires_independent_original_POST_gates' if passed else 'REJECTED_actual_training_source'
  else:final['scope_status']='REJECTED_actual_motion_cell_or_depth'
  dump(O/'report.json',final)
except Limit as e:
 emit(dict(kind='NEW_method_limit_RETURN',reason=str(e)))
 dump(O/'report.json',dict(schema='v2d.NEW-structured-all-row-evaluation.v1',scope_status='UNRESOLVED_declared_method_limit',reason=str(e),unique_queries=count,callbacks=callback_count,wall_seconds=time.monotonic()-start,actual_OptimizeResult_UNAVAILABLE=True,no_candidate_selected=True,actual_source_queries=0,no_qualification=True))

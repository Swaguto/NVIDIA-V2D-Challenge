"""ONE bounded hard-cell solver attempt and at most ONE raw nonlinear trial."""
from pathlib import Path
import datetime,json,hashlib,os,sys,time
import numpy as np
from scipy import sparse
S=Path(__file__).resolve().parents[1];C=S/'compute';I=S/'e037-hard-raster-model-v1';O=S/'e037-hard-raster-evaluation-v1'
sys.path.insert(0,str(S/'e037-direct-SDP-runtime-v2/site-packages'))
import clarabel
from hard_training_conic_returns_v1 import snapshot
from full_pose_inputs_v1 import load
from full_pose_training_oracle_v1 import FullPoseTrainingOracle
from full_pose_geometry_runtime_v1 import geometry_all
from full_pose_construction_v1 import actual_motion
from object_full_pose_maps_v1 import compose
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cv(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):cv(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [cv(v)for v in x]
 if isinstance(x,float)and not np.isfinite(x):return dict(nonfinite=repr(x))
 return x

def save(n,x):
 p=O/n
 with p.open('w')as f:json.dump(cv(x),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 p.chmod(0o600);return dict(path=str(p),sha256=sha(p))
def arrays(n,x):
 p=O/(n if n.endswith('.npz')else n+'.npz')
 with p.open('wb')as f:np.savez_compressed(f,**x);f.flush();os.fsync(f.fileno())
 p.chmod(0o600);return dict(path=str(p),sha256=sha(p))
def emit(x):
 with (O/'normalized-BEFORE-AFTER-journal.jsonl').open('a')as f:json.dump(cv(x),f,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
start=time.monotonic();meta=json.loads((I/'NEW-canonical-metadata-RETURN.json').read_text());binding=S/'e037-hard-raster-binding-v2/report.json';br=json.loads(binding.read_text());assert br['all151frame_copy_bindings_exact']and br['all_lift_cones_and_objective_bound']
save('ONE-hard-raster-attempt-declaration-BEFORE.json',dict(model_metadata_sha256=sha(I/'NEW-canonical-metadata-RETURN.json'),independent_binding=dict(path=str(binding),sha256=sha(binding)),solver_calls_maximum=1,raw_trial_maximum=1,QP_iterations=100,QP_seconds=60.,threads=1,default_tolerances=True,no_linefraction_retry_warmstart_continuation_or_tuning=True,all151_integer_cell_equalities_strict_depth_actual_training_NN_median_quality_and6motion_REQUIRED=True,original_fullpixel_heldout_POST_only=True,zeroF_cache_only=True,raw_variables_to_pose='eta=r*x[0:3], t=x[3:6]/s; delta=eta/(1+t.t), theta=2atan(||t||)*t/||t||; actual world/camera/native maps use Exp(theta) with numerical Cayley/Exp alias uncertainty retained',sigma_is_proposal_auxiliary_NOT_source_truth_or_command=True,source_or_original_model_feasibility_NOT_claimed_from_solver=True,resource_RSS_guard=2147483648,resource_wall_guard=240,no_dependency_or_VM_mutation=True,solver_peak_memory_UNVERIFIED_and_guard_stops_if_exceeded=True))
A=sparse.load_npz(I/'NEW-canonical-A-RETURN.npz');P=sparse.load_npz(I/'NEW-canonical-P-RETURN.npz')
with np.load(I/'NEW-canonical-vectors-RETURN.npz')as z:b=z['b'];q=z['q']
# Lossless zero-entry packing only. All rows/constraints and every nonzero value
# remain. Independent per-column digest binding precedes the constructor.
def hashes(matrix):
 out=[]
 for j in range(matrix.shape[1]):
  a0,a1=matrix.indptr[j:j+2];v=matrix.data[a0:a1];idx=matrix.indices[a0:a1];keep=v!=0;out.append(hashlib.sha256(idx[keep].astype(np.int64).tobytes()+v[keep].tobytes()).hexdigest())
 return out
before=hashes(A);oldnnz=A.nnz;A.eliminate_zeros();after=hashes(A);save('NEW-lossless-solver-packing-RETURN.json',dict(original_nnz=oldnnz,packed_nnz=A.nnz,all_rows_and_columns_preserved=A.shape,all_nonzero_column_hashes_before=before,all_nonzero_column_hashes_after=after,exactly_same_nonzero_matrix=before==after,zero_entries_only_removed=True));assert before==after
settings=clarabel.DefaultSettings();settings.max_iter=100;settings.time_limit=60.;settings.max_threads=1;settings.verbose=False
cones=[clarabel.NonnegativeConeT(meta['nonnegative'])]+[clarabel.SecondOrderConeT(4)for _ in range(900)]+[clarabel.SecondOrderConeT(5)for _ in range(151)]
save('NEW-actual-settings-cones-RETURN.json',dict(settings=snapshot(settings),cone_types=[type(v).__name__ for v in cones],rows=A.shape[0],columns=A.shape[1],packed_storage_bytes=A.data.nbytes+A.indices.nbytes+A.indptr.nbytes))
emit(dict(kind='BEFORE_ONE_NEW_hard_raster_solver_constructor',all_matrices_independently_bound=True,returned_constructor_UNKNOWN_until_AFTER=True));solver=clarabel.DefaultSolver(P,q,A,b,cones,settings);save('NEW-actual-solver-constructor-RETURN.json',dict(actual_solver=snapshot(solver),info=snapshot(solver.get_info())));del A,P;callbacks=[]
def callback(info):
 out=snapshot(info);callbacks.append(out);emit(dict(kind='AFTER_NEW_actual_info_callback_RETURN',callback=len(callbacks)-1,actual_RETURN=out));return False
solver.set_termination_callback(callback);emit(dict(kind='BEFORE_SOLE_NEW_hard_raster_solver_call'));res=solver.solve();raw=arrays('ONE-actual-raw-x-z-s-RETURN',dict(x=np.asarray(res.x),z=np.asarray(res.z),s=np.asarray(res.s)));save('ONE-actual-solver-RETURN.json',dict(solution=snapshot(res,exclude=('x','z','s')),info=snapshot(solver.get_info()),callbacks=callbacks,raw_RETURN=raw));emit(dict(kind='AFTER_SOLE_NEW_hard_raster_solver_RETURN',actual_raw_RETURN=raw,actual_status=str(res.status)));x=np.asarray(res.x,float)
report=dict(schema='v2d.NEW-hard-raster-evaluation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),solver_status=str(res.status),solver_calls=1,actual_callbacks=len(callbacks),raw_RETURN=raw,actual_nonzero_trials=0,accepted=False,no_original_fullpixel_heldout_rescue_or_fit=True,no_native_FK_position_physics_GPU_PPO=True,all_F_source_geometry_physics_original_gates_UNCHANGED=True,no_reference_task_policy_AUC_qualification=True)
if str(res.status)!='Solved'or x.shape!=(1058,)or not np.isfinite(x).all():
 report.update(status='UNRESOLVED_no_Solved_finite_raw_proposal',method_seconds=time.monotonic()-start);save('report.json',report);print(json.dumps(cv(report)));raise SystemExit(0)
y=x[:1057].reshape(151,7);eta=meta['radius']*y[:,:3];t=y[:,3:6]/meta['scale'];D=1+np.einsum('fi,fi->f',t,t,optimize=False);delta=eta/D[:,None];theta=np.empty_like(t)
for f in range(151):
 length=float(np.hypot(np.hypot(t[f,0],t[f,1]),t[f,2]));theta[f]=2*t[f]if length==0 else (2*np.arctan(length)/length)*t[f]
params=np.column_stack([delta,theta]);pid='ONE-hard-raster-'+hashlib.sha256(x.tobytes()).hexdigest()[:16];seal=arrays(pid+'-SEALED-BEFORE',dict(raw_x=x,eta=eta,cayley_t=t,D=D,parameters=params,sigma=y[:,6]));emit(dict(kind='AFTER_NEW_raw_proposal_SEALED_BEFORE_queries',proposal_id=pid,actual_RETURN=seal));del solver,res
# Fresh pure helpers receive only this NEW raw proposal. No old point replay.
d=load();world=np.array([compose(T,p[:3],p[3:])for T,p in zip(d['world'],params)]);pose=arrays('ONE-UNQUALIFIED-derived-world-poses-RETURN',dict(world_from_object=world,parameters=params,frame_indices=d['indices'],timestamps_s=d['times']));motion=actual_motion(world,d['times'],d['motion_caps']);ma=arrays('ONE-actual-motion-vectors-RETURN',motion['arrays']);save('ONE-six-actual-motion-decisions-RETURN.json',dict(values=motion['values'],caps=d['motion_caps'],passes=motion['passes'],comparison=1e-9,array_RETURN=ma))
def load_training(f):
 row=d['stats'][f];pp=Path(row['partition']['path']);pc=Path(row['point_data']['path']);assert sha(pp)==row['partition']['sha256']and sha(pc)==row['point_data']['sha256']
 with np.load(pp)as z:mask=z['training_mask'];domain=z['domain']
 with np.load(pc)as z:points=z['points_C_m']
 return dict(training_mask=mask,domain=domain,points_C_m=points)
oracle=FullPoseTrainingOracle(d,load_training,emit,arrays);source=[];cells=[]
for f in range(151):
 out=oracle.query(f,params[f,:3],params[f,3:],pid);source.append(out)
 with np.load(I/f'frame-{f:03d}-NEW-hard-cell-coefficients-RETURN.npz')as z:target=z['target_integer_uv']
 if out['status']=='CACHED_F_BY_HASH':mismatch=0;cellok=True;status='CACHED_F_integer_cells_BY_HASH'
 elif out['status']=='NEW_actual_full_pose_TRAINING_ONLY':
  with np.load(O/f'{pid}-f{f:03d}-image-RETURN.npz')as z:integer=z['uv_integer']
  mismatch=int(np.count_nonzero(integer!=target));cellok=mismatch==0;status='NEW_actual_integer_cells'
 else:mismatch=None;cellok=False;status='UNRESOLVED_actual_integer_cells'
 cell=dict(frame=f,proposal_id=pid,status=status,mismatched_coordinates=mismatch,passes_hard_cell_construction=cellok);cells.append(cell);save(f'frame-{f:03d}-NEW-actual-integer-cell-RETURN.json',cell)
sr=save('ONE-ALL151-actual-training-source-RETURNS.json',source);cr=save('ONE-ALL151-actual-integer-cell-RETURNS.json',cells);g=geometry_all(d,params);gr=arrays('ONE-NEW-current-fullpair-geometry-RETURN',{k:v for k,v in g.items()if isinstance(v,np.ndarray)});initial=json.loads((S/'e037-full-pose-preflight-v1/NEW-initial-proxy-RETURN.json').read_text())['proxy_phase_m'];bounds=bool(np.all(np.linalg.norm(eta,axis=1)<=meta['radius']/8)and np.all(np.linalg.norm(t,axis=1)<=np.pi/24));accepted=bool(all(v['passes_TRAINING_construction']for v in source)and all(v['passes_hard_cell_construction']for v in cells)and all(motion['passes'])and bounds and np.isfinite(g['gaps']).all()and g['proxy_phase_m']<initial)
report.update(status='UNQUALIFIED_hard_cell_training_motion_pass_REQUIRES_original_POST_gates'if accepted else'REJECTED_raw_hard_raster_proposal',proposal_id=pid,sealed_proposal=seal,pose_RETURN=pose,source_RETURN=sr,cell_RETURN=cr,geometry_RETURN=gr,actual_nonzero_trials=1,training_failed_frames=[v['frame']for v in source if not v['passes_TRAINING_construction']],hard_cell_failed_frames=[v['frame']for v in cells if not v['passes_hard_cell_construction']],all6motion_passes=motion['passes'],constructor_norms_pass=bounds,initial_proxy_phase_m=initial,raw_proxy_phase_m=g['proxy_phase_m'],raw_sigma_lift_violation_DIAGNOSTIC=float(np.maximum(0,np.einsum('fi,fi->f',y[:,3:6],y[:,3:6])-y[:,6]).max(initial=0)),accepted=accepted,method_seconds=time.monotonic()-start,actual_cell_gate_NOT_replaced_by_ideal_Cayley_or_closed_face_model=True,all_raw_returns_unaltered_and_unpromoted=True);save('report.json',report);print(json.dumps(cv(dict(status=report['status'],source_failed=len(report['training_failed_frames']),cells_failed=len(report['hard_cell_failed_frames']),motion=motion['passes'],accepted=accepted,seconds=report['method_seconds']))))

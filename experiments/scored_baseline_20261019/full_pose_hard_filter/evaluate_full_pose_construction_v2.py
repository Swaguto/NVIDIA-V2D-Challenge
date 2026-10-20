"""SOLE NEW one-QP/one-raw-trial full-object-pose construction."""
from pathlib import Path
import datetime,json,hashlib,os,sys,time
import numpy as np
from scipy import sparse
S=Path(__file__).resolve().parents[1];C=S/'compute';I=S/'e037-full-pose-preflight-v1';O=S/'e037-full-pose-evaluation-v2'
sys.path.insert(0,str(S/'e037-direct-SDP-runtime-v2/site-packages'))
import clarabel
from full_pose_inputs_v1 import load
from full_pose_geometry_runtime_v1 import geometry_all
from full_pose_training_oracle_v1 import FullPoseTrainingOracle
from full_pose_construction_v1 import actual_motion
from object_full_pose_maps_v1 import compose,full_pose_acceptance,source_dispatch
from hard_training_conic_returns_v1 import snapshot
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cv(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return{str(k):cv(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return[cv(v)for v in x]
 if isinstance(x,float)and not np.isfinite(x):return dict(nonfinite='nan'if np.isnan(x)else('inf'if x>0 else'-inf'))
 return x
def save(n,x):
 p=O/n
 with p.open('w')as f:json.dump(cv(x),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 p.chmod(0o600);return dict(path=str(p),sha256=sha(p))
def arrays(n,d):
 p=O/(n if n.endswith('.npz')else n+'.npz')
 with p.open('wb')as f:np.savez_compressed(f,**d);f.flush();os.fsync(f.fileno())
 p.chmod(0o600);return dict(path=str(p),sha256=sha(p))
def emit(x):
 with (O/'normalized-BEFORE-AFTER-journal.jsonl').open('a')as f:json.dump(cv(x),f,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
start=time.monotonic();d=load();meta=json.loads((I/'NEW-canonical-metadata-RETURN.json').read_text());binding=S/'e037-full-pose-preflight-binding-v2/report.json';br=json.loads(binding.read_text());assert br['all907_columns_and70495_rows_bound']
A=sparse.load_npz(I/'NEW-canonical-A-RETURN.npz');P=sparse.load_npz(I/'NEW-canonical-P-RETURN.npz')
with np.load(I/'NEW-canonical-vectors-RETURN.npz')as z:b=z['b'];q=z['q']
config=meta['config'];save('ONE-NEW-full-pose-method-declaration-BEFORE.json',dict(method='ONE fully coupled907column conic proposal, then ONE full raw nonzero hard-filter trial ONLY',constructor_provenance=d['provenance'],canonical_inputs={str(p):sha(p)for p in I.iterdir()if p.suffix in ['.npz','.json']},complete_independent_binding=dict(path=str(binding),sha256=sha(binding)),configuration=config,all151clocks_all6pose_coordinates=True,starts_ORIGINAL_F_only=True,no_failed_point_or_intermediate_warmstart=True,no_line_search_restart_or_parameter_tuning=True,actual_changed_rotation_training_k2_einsum_and_fulltriangle_counts_REQUIRED=True,all_six_original_object_motion_and_actual_angular_p95_REQUIRED=True,original_fullpixel_heldout_POST_only_NOT_input=True,all_actual_returns_sealed_BEFORE_queries=True,geometric_face_union_and_search_J_NOT_native_or_contact_force_grasp=True,no_GPU_or_PPO=True))
settings=clarabel.DefaultSettings();settings.max_iter=100;settings.time_limit=60.;settings.max_threads=1;settings.verbose=False
cones=[clarabel.NonnegativeConeT(meta['nonnegative'])]+[clarabel.SecondOrderConeT(c['size'])for c in meta['cones']]
save('NEW-actual-settings-and-cone-constructor-RETURN.json',dict(settings=snapshot(settings),cone_types=[type(x).__name__ for x in cones],dimensions=dict(rows=A.shape[0],columns=A.shape[1])))
emit(dict(kind='BEFORE_ONE_NEW_solver_constructor',complete_matrix_binding_sha256=sha(binding)));solver=clarabel.DefaultSolver(P,q,A,b,cones,settings);save('NEW-actual-solver-constructor-RETURN.json',dict(solver=snapshot(solver),initial_info=snapshot(solver.get_info())));calls=[]
def callback(info):
 out=snapshot(info);calls.append(out);emit(dict(kind='AFTER_NEW_actual_info_callback_RETURN',callback=len(calls)-1,actual_RETURN=out));return False
solver.set_termination_callback(callback);emit(dict(kind='BEFORE_SOLE_NEW_solver_call'));result=solver.solve();raw=arrays('ONE-actual-raw-x-z-s-RETURN',dict(x=np.asarray(result.x),z=np.asarray(result.z),s=np.asarray(result.s)));summary=save('ONE-actual-solver-RETURN.json',dict(solution=snapshot(result,exclude=('x','z','s')),info=snapshot(solver.get_info()),callbacks=calls,raw_RETURN=raw));emit(dict(kind='AFTER_SOLE_NEW_solver_RETURN',actual_RETURN=summary));x=np.asarray(result.x,float)
report=dict(schema='v2d.NEW-one-full-pose-evaluation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),solver_status=str(result.status),solver_calls=1,callbacks=len(calls),actual_raw_RETURN=raw,configuration=config,method_seconds=time.monotonic()-start,original_F_robot_commands_source_geometry_parameters_gates_UNCHANGED=True,no_original_fullpixel_heldout_query_or_rescue=True,no_native_position_physics_GPU_PPO=True,no_reference_task_policy_AUC_qualification=True)
if str(result.status)!='Solved'or x.shape!=(907,)or not np.isfinite(x).all():
 report.update(status='UNRESOLVED_no_Solved_finite_raw_proposal',actual_nonzero_trials=0,accepted=False);save('report.json',report);print(json.dumps(cv(report)));raise SystemExit(0)
y=x[:906].reshape(151,6);params=y.copy();params[:,:3]*=d['radius'];pid='ONE-raw-full-pose-'+hashlib.sha256(params.tobytes()).hexdigest()[:16];seal=arrays(pid+'-SEALED-BEFORE',dict(raw_x=x,parameters=params,frame_indices=d['indices'],timestamps_s=d['times']));emit(dict(kind='AFTER_NEW_proposal_SEALED_BEFORE_any_query',proposal_id=pid,sealed_RETURN=seal))
world=np.array([compose(T,p[:3],p[3:])for T,p in zip(d['world'],params)]);poses=arrays('ONE-UNQUALIFIED-derived-world-poses-RETURN',dict(world_from_object=world,parameters=params,frame_indices=d['indices'],timestamps_s=d['times']));emit(dict(kind='AFTER_NEW_actual_pose_constructor_RETURN',proposal_id=pid,actual_RETURN=poses))
motion=actual_motion(world,d['times'],d['motion_caps']);mr=arrays('ONE-actual-original-motion-arrays-RETURN',motion['arrays']);save('ONE-actual-original-six-motion-RETURN.json',dict(values=motion['values'],caps=d['motion_caps'],passes=motion['passes'],comparison=motion['comparison'],array_RETURN=mr));emit(dict(kind='AFTER_NEW_all6_actual_motion_RETURN',proposal_id=pid,values=motion['values'],passes=motion['passes']))
def training_load(f):
 st=d['stats'][f];part=Path(st['partition']['path']);point=Path(st['point_data']['path']);assert sha(part)==st['partition']['sha256']and sha(point)==st['point_data']['sha256']
 with np.load(part)as z:mask=z['training_mask'];domain=z['domain']
 with np.load(point)as z:points=z['points_C_m']
 return dict(training_mask=mask,domain=domain,points_C_m=points)
oracle=FullPoseTrainingOracle(d,training_load,emit,arrays);source=[oracle.query(f,params[f,:3],params[f,3:],pid)for f in range(151)];sr=save('ONE-ALL151-actual-training-source-RETURNS.json',source)
emit(dict(kind='BEFORE_NEW_raw_trial_allpair_rotating_support_query',proposal_id=pid,geometry_J_scope='current_RIGHT_tangent_only; no next QP or absolute-theta derivative claim'))
g=geometry_all(d,params);gr=arrays('ONE-NEW-allpair-current-geometry-RETURN',{k:v for k,v in g.items()if isinstance(v,np.ndarray)});emit(dict(kind='AFTER_NEW_actual_allpair_geometry_RETURN',proposal_id=pid,array_RETURN=gr,proxy_phase_m=g['proxy_phase_m'],all_axis_values=len(g['all_axis_gaps']),NOT_native_gap_or_contact=True));initial=json.loads((I/'NEW-initial-proxy-RETURN.json').read_text())['proxy_phase_m'];bounds=bool(np.all(np.linalg.norm(params[:,:3],axis=1)<=config['translation_norm_halfwidth_m'])and np.all(np.linalg.norm(params[:,3:],axis=1)<=config['rotation_norm_halfwidth_rad']));accepted=full_pose_acceptance(sealed_id=pid,return_id=pid,frames=[r['frame']for r in source],expected_frames=d['indices'],training_passes=[r['passes_TRAINING_construction']for r in source],motion_passes=motion['passes'],raw_finite=True,source_queries_correct=all(r['status']==('CACHED_F_BY_HASH'if source_dispatch(params[f,:3],params[f,3:])=='CACHED_F_BY_HASH'else'NEW_actual_full_pose_TRAINING_ONLY')for f,r in enumerate(source)));descent=bool(np.isfinite(g['gaps']).all()and g['proxy_phase_m']<initial);passed=bool(accepted['accepted_by_training_motion_construction']and bounds and descent)
report.update(status='UNQUALIFIED_training_motion_full_pose_component_REQUIRES_original_POST_gates'if passed else'REJECTED_raw_full_pose_training_motion_or_construction',proposal_id=pid,sealed_proposal=seal,pose_RETURN=poses,actual_nonzero_trials=1,training_source_RETURN=sr,geometry_RETURN=gr,training_failed_frames=[r['frame']for r in source if not r['passes_TRAINING_construction']],all6motion_passes=motion['passes'],initial_proxy_phase_m=initial,raw_trial_proxy_phase_m=g['proxy_phase_m'],actual_pair_instances=len(g['gaps']),actual_axis_values=len(g['all_axis_gaps']),construction_norm_bounds_pass=bounds,strict_proxy_descent=descent,actual_source_motion_acceptance=accepted,accepted=passed,method_seconds=time.monotonic()-start,raw_trial_NOT_promoted_to_reference=True,current_geometry_J_is_RIGHT_tangent_not_absolute_theta=True);save('report.json',report);print(json.dumps(cv(dict(status=report['status'],solver_status=report['solver_status'],callbacks=len(calls),training_failed_frames=len(report['training_failed_frames']),motion_passes=motion['passes'],accepted=passed,method_seconds=report['method_seconds']))))

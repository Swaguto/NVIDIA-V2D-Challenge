"""NEW complete-construction contracts, including two NEW analytical tiny QPs."""
from pathlib import Path
import datetime,hashlib,json,math,os,sys
import cv2,numpy as np
from scipy import sparse
S=Path(__file__).resolve().parents[1];C=S/'compute';O=S/'e037-hard-source-filter-contracts-v1'
sys.path.insert(0,str(S/'e037-direct-SDP-runtime-v2/site-packages'))
import clarabel
from hard_source_filter_direction_v1 import CONFIG,build_direction,actual_trial_decision
from hard_source_filter_binding_v1 import independently_bind
from hard_training_conic_returns_v1 import snapshot
from object_training_source_semantics_v1 import training_domain,raster_training_counts,vertex_median_semantics,exact_training_iou_nonregression,real_quality

def plain(v):
 if isinstance(v,np.ndarray):return plain(v.tolist())
 if isinstance(v,np.generic):return plain(v.item())
 if isinstance(v,dict):return {str(k):plain(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [plain(x)for x in v]
 if isinstance(v,float)and not math.isfinite(v):return dict(nonfinite_float=repr(v))
 return v
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def arrays(n,**values):
 p=O/n;np.savez_compressed(p,**values);p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):log.write(json.dumps(plain(v),allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
 controls=[]
 def control(n,before,fn,test):
  p=save(n+'-BEFORE.json',before);emit(dict(kind='BEFORE_NEW_contract',name=n,constructor=p))
  r=fn();p=save(n+'-RETURN.json',r);emit(dict(kind='AFTER_NEW_actual_RETURN',name=n,actual_RETURN=p));assert test(r),n;controls.append(n)
 declaration=dict(schema='v2d.NEW-hard-source-filtered-construction-declaration.v1',
  method='ONE sequential step-QP proposal and ACTUAL hard training-source/motion filter; no IoU derivative',
  config=CONFIG,initial_point='Fzero world translation ONLY; source/geometry baseline caches reused BY HASH',
  source_median_rows='chosen current gradient is a SEARCH predictor ONLY; actual nearest/ranks/ties and FULLtriangle IoU/quality are trial-filtered',
  source_authority='SAVED EVEN training only; ORIGINAL fullpixel/heldout gates POSTconstruction, never fitting inputs',
  geometry='active affine direction at current state of ALL retained compiled support directions, conservative chosen branch within a declared new method; not native contacts/forces',
  raw_return_policy='save constructor, raw QP x/z/s/info/callbacks, scaled line trials and actual source/motion/geometry RETURNS BEFORE assertions/furtherwork; never clip/repair raw step',
  matrices='independent complete binding before EACH solver constructor; initial matrix cached after preflight',
  acceptance='sealed trial identity + actual source on all151frames + six original motion gates + algorithmic box + strict clearance-proxy descent',
  algorithmic_box_step_slack_regularization_NOT_original_gates_or_source_weights=True,
  stops='8major/8trial/3fixed-backtrack/160sec caps; failed/no-descent return UNRESOLVED, no retry/warmstart/tuning/continuation',
  final='algorithm terminal accepted vector ONLY; no best/intermediate selection; if proxy/actual training/motion incomplete then REJECTED, no fullpixel rescue',
  original_source_native_self_swept_motor_grasp_NOT_qualified=True,
  no_episode_solver_or_nonzero_pose_in_this_contract_job=True)
 save('ONE-NEW-method-declaration-BEFORE.json',declaration);emit(dict(kind='BEFORE_NEW_contracts',declaration=declaration))
 # Source-metric functional fixture is newly synthetic, never an episode scene.
 K=np.array([[1024.,0,0],[0,1024.,0],[0,0,1.]])
 V=np.array([[5/1024,5/1024,1.],[50/1024,5/1024,1.],[5/1024,50/1024,1.]])
 faces=np.array([[0,1,2]]);mask=np.zeros((64,64),np.uint8);cv2.fillConvexPoly(mask,np.array([[5,5],[50,5],[5,50]],np.int32),255)
 D=training_domain(mask.shape);M=(mask>127)&D
 xx,yy=np.meshgrid(np.arange(6,16),np.arange(6,26));sel=(xx+yy)%2==0
 points=np.column_stack([xx[sel]/1024,yy[sel]/1024,np.ones(np.count_nonzero(sel))])
 T=np.eye(4);base_image=raster_training_counts(V,faces,K,M,D)
 save('NEW-synthetic-base-raster-RETURN.json',base_image)
 base_median=vertex_median_semantics(V,points,T,np.eye(3))
 save('NEW-synthetic-base-median-RETURN.json',base_median)
 data=dict(motion_M=sparse.csc_matrix((0,3)),motion_base=np.empty(0),motion_caps=np.empty(0),
  pair_frames=np.array([0]),baseline_median=np.array([base_median['median']]),Rcw=np.eye(3)[None],minimum_C0_z=np.array([1.]))
 state=dict(world_delta=np.zeros((1,3)),active_normals=np.array([[1.,0,0]]),gap_proxy=np.array([-.003]),
  median=np.array([base_median['median']]),median_gradient=base_median['world_translation_local_gradient'][None])
 tiny_returns=[]
 def tiny(name,local_state):
  problem=build_direction(local_state,data)
  arrays(name+'-matrix-RETURN.npz',A=problem['A'].toarray(),b=problem['b'],P=problem['P'].toarray(),q=problem['q'],scales=problem['scales'],raw_A=problem['raw_A'].toarray(),raw_b=problem['raw_b'])
  bind=independently_bind(problem,local_state,data,CONFIG)
  save(name+'-independent-binding-RETURN.json',bind);assert bind['passed']
  settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.max_iter=100;settings.time_limit=10.
  save(name+'-settings-RETURN.json',snapshot(settings))
  cones=[clarabel.NonnegativeConeT(problem['nonnegative_rows'])]
  save(name+'-cones-RETURN.json',[snapshot(x)for x in cones])
  solver=clarabel.DefaultSolver(problem['P'],problem['q'],problem['A'],problem['b'],cones,settings)
  save(name+'-constructor-RETURN.json',snapshot(solver));callbacks=[]
  def cb(info):
   value=snapshot(info);callbacks.append(value);save(name+f'-callback-{len(callbacks)-1:03d}-RETURN.json',value);return False
  solver.set_termination_callback(cb);sol=solver.solve()
  raw=arrays(name+'-raw-x-z-s-RETURN.npz',x=np.asarray(sol.x),z=np.asarray(sol.z),s=np.asarray(sol.s))
  result=dict(status=str(sol.status),x=np.asarray(sol.x),raw=raw,info=snapshot(solver.get_info()),callbacks=len(callbacks),independent_binding_pass=bind['passed'])
  save(name+'-actual-solver-RETURN.json',result);tiny_returns.append(result);return result
 control('NEW-step-QP-smooth-median-native-phase-analytic-optimum',dict(expected_step_x=.001,expected_rho=.001,source_gradient=state['median_gradient']),
  lambda:tiny('tiny-free-source',state),lambda r:r['status']=='Solved'and abs(r['x'][0]-.001)<1e-7 and abs(r['x'][3]-.001)<1e-7)
 blocked=dict(state,median_gradient=np.array([[1.,0,0]]))
 control('NEW-step-QP-source-plane-blocks-native-descent',dict(expected_step_x=0,expected_rho=.002,source_gradient=blocked['median_gradient']),
  lambda:tiny('tiny-source-block',blocked),lambda r:r['status']=='Solved'and abs(r['x'][0])<1e-6 and abs(r['x'][3]-.002)<1e-6)
 def actual_hard_filter():
  # This is the actual unaltered first tiny QP direction, not a clipped optimum.
  p=tiny_returns[0]['x'][:3].copy();TT=T.copy();TT[:3,3]+=p
  image=raster_training_counts(V+p,faces,K,M,D);save('NEW-tiny-raw-step-actual-raster-RETURN.json',image)
  median=vertex_median_semantics(V,points,TT,np.eye(3));save('NEW-tiny-raw-step-actual-median-RETURN.json',median)
  ratio=exact_training_iou_nonregression(image,base_image);save('NEW-tiny-raw-step-actual-IoU-RETURN.json',ratio)
  q0=real_quality(base_image['iou'],base_median['median']);q1=real_quality(image['iou'],median['median'])
  save('NEW-tiny-raw-step-actual-qualities-RETURN.json',dict(before=q0,after=q1))
  passed=bool(ratio['nonregressing']and median['median']<=base_median['median']and q1>=q0)
  source=[dict(frame=0,proposal_id='NEW-tiny-raw-step',status='NEW_training_source_predicate_ONLY',passes_TRAINING_construction=passed)]
  decision=actual_trial_decision(.002,.001,source,{str(i):True for i in range(6)},True,[0],'NEW-tiny-raw-step',{0})
  return dict(raw_QP_step=p,actual_IoU_before_after=[base_image['iou'],image['iou']],actual_median_before_after=[base_median['median'],median['median']],
   actual_source_pass=passed,decision=decision,no_IoU_gradient=True,actual_median_search_plane_not_source_certificate=True)
 control('NEW-solved-step-improves-median-but-actual-raster-filter-rejects',dict(actual_first_tiny_QP_raw_step=True,synthetic_original_functionals_only=True),actual_hard_filter,
  lambda r:r['actual_median_before_after'][1]<r['actual_median_before_after'][0]and r['actual_IoU_before_after'][1]<r['actual_IoU_before_after'][0]and not r['decision']['accepted'])
 def identity_binding():
  good=dict(frame=0,proposal_id='sealed-current',status='NEW_training_source_predicate_ONLY',passes_TRAINING_construction=True)
  wrong_id=dict(good,proposal_id='old-trial');wrong_cache=dict(good,status='CACHED_F_training_values_BY_HASH')
  out=[]
  for item in [good,wrong_id,wrong_cache]:out.append(actual_trial_decision(.002,.001,[item],{str(i):True for i in range(6)},True,[0],'sealed-current',{0}))
  return dict(decisions=out)
 control('NEW-sealed-proposal-and-nonzero-cache-authorization',dict(nonzero_frame=0,expected_proposal='sealed-current'),identity_binding,
  lambda r:[x['accepted']for x in r['decisions']]==[True,False,False])
 def scaling():
  nf=2;d=dict(data,motion_M=sparse.csc_matrix(np.array([[-20.,0,0,20,0,0],[0,-20,0,0,20,0],[0,0,-20,0,0,20]])),motion_base=np.array([.1,0,0]),motion_caps=np.array([.8]),pair_frames=np.array([0,1]),baseline_median=np.array([.1,.2]),Rcw=np.tile(np.eye(3),(2,1,1)),minimum_C0_z=np.array([1.,1.]))
  st=dict(world_delta=np.array([[.0001,0,0],[0,-.0002,0]]),active_normals=np.array([[1.,0,0],[0,1.,0]]),gap_proxy=np.array([-.002,-.0015]),median=np.array([.099,.199]),median_gradient=np.array([[-1.,0,0],[0,-1.,0]]))
  pr=build_direction(st,d);bound=independently_bind(pr,st,d,CONFIG);k=pr['nonnegative_rows'];scale=pr['scales'][k:k+4]
  step=np.r_[np.array([.0002,-.0001,0,-.0001,.0002,0]),.002]
  raw=pr['raw_b']-pr['raw_A']@step;scaled=pr['b']-pr['A']@step
  return dict(binding=bound,whole_SOC_scale=scale,raw=raw,scaled=scaled,closure=float(np.max(abs(scaled*pr['scales']-raw))),
   positive_scaling_NOT_gate_tolerance=True)
 control('NEW-coupled-step-motion-and-whole-SOC-conditioning',dict(two_clocks=True,nonzero_current_state=True),scaling,
  lambda r:r['binding']['passed']and len(set(r['whole_SOC_scale']))==1 and r['whole_SOC_scale'][0]>=20 and r['closure']<1e-12)
 def fraction_policy():
  raw=tiny_returns[0]['x'].copy();saved=raw.tobytes();cur=np.zeros(3)
  trials=[cur+a*raw[:3]for a in CONFIG['trial_fractions']]
  return dict(raw_direction_unchanged=raw.tobytes()==saved,line_fractions=CONFIG['trial_fractions'],separate_derived_trials=trials,
   raw_rho_unchanged=raw[3]==tiny_returns[0]['x'][3],not_candidate_repair_or_intermediate_selection=True)
 control('NEW-raw-step-versus-declared-line-trial-identities',dict(raw_first_tiny_RETURN_is_immutable=True),fraction_policy,
  lambda r:r['raw_direction_unchanged']and r['raw_rho_unchanged']and np.array_equal(r['separate_derived_trials'][1],r['separate_derived_trials'][0]*.5))
 report=dict(schema='v2d.NEW-hard-source-filter-contracts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  NEW_contracts=len(controls),contracts=controls,NEW_tiny_QP_solves=2,all_pass=True,
  old_controls_and_source_data_reused_BY_HASH_NOT_replayed=True,no_episode_pose_or_solver=True,
  actual_source_filter_rejects_solved_predictor_example=True,
  original_source_motion_native_self_swept_motor_grasp_gates_UNCHANGED=True)
 save('report.json',report);emit(dict(kind='AFTER_NEW_contracts',actual_RETURN=report));print(json.dumps(report))

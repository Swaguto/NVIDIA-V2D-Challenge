"""SOLE NEW bounded homogeneous whole-cone joint construction, terminal8 only."""
from joint_hsd_embedding_v1 import *
import traceback,shutil
D=S/'e037-joint-hsd-evaluation-v1';PRE=S/'e037-joint-hsd-preflight-v1';I=S/'e037-joint-executable-interface-v1';record=MethodRecorder(D)
start=time.monotonic();deadline=start+150;completed=0;point_sealed=False;full_query=False
paths=[PRE/(n+e) for n in ['NEW-embedded-Schur-normal-RETURN','NEW-embedding-constants-F-start-RETURN'] for e in ['.json','.npz']]+[PRE/'NEW-fixed-method-declaration-RETURN.json',S/'e037-joint-hsd-binding-v1/report.json',C/'joint_hsd_embedding_v1.py',C/'joint-mean-proof-checkpoint-v1.json',dynamic_file(),dynamic_file().with_suffix('.npz')]
save(D/'NEW-fixed-evaluation-declaration-BEFORE.json',dict(input_sha256={str(p):sha(p) for p in paths},method='NEW square-root-free objective homogeneous self-dual joint embedding',
 updates=8,terminal_update_ONLY=8,method_seconds=150,outer_seconds=240,RSS_bytes=2*1024**3,disk_reserve_bytes=int(1.5*1024**3),
 no_old_method_query_controls_solver_or_failed_state_replay=True,original_Pq_domain_rows_and_phase_zero_slice_unchanged=True,all_original_gates_still_required=True,
 no_retry_restart_tuning_rescale_clip_repair_warmstart_best_or_linefraction=True,every_NEW_return_typed_lossless_before_checks_further_work=True))
assert json.loads((S/'e037-joint-hsd-binding-v1/report.json').read_text())['passed']
try:
 maps=typed(paths[0]);con=typed(paths[2]);idx=con['idx'];p=con['p'];b=con['b'];c=con['c'];cones=[tuple(map(int,a)) for a in con['cones']];nx=len(c);nr=len(b)
 A=sparse.load_npz(I/'NEW-object-retained-A-RETURN.npz');R=sparse.load_npz(I/'NEW-robot-linear-A-RETURN.npz');raw=typed(dynamic_file());J=raw['J'];del raw
 normal=maps['normal'];B=maps['base_J'];Ds=maps['direction_J'];weight=maps['weight']
 ux=con['x0'].copy();uy=np.zeros(nr);ut=1.;vx=np.zeros(nx);vy=np.zeros(nr);vk=1.
 record('NEW-method-F-authority-input-BEFORE',dict(ux=ux,uy=uy,tau=ut,vx=vx,vy=vy,kappa=vk,normal=normal,original_objective_exact_epigraph=True,
  all_immutable_operator_inputs_and_saved_F_query_proof_BY_HASH=True,formal_phase_coordinate=PHASE,phase_zero_target_preserved=True))
 estimate=8*(11*nr*8+24*nx*8)+600*1024**2;free=shutil.disk_usage(S).free
 record('NEW-before-factor-resource-RETURN',dict(free_disk_bytes=free,declared_return_estimate=estimate,reserve_bytes=int(1.5*1024**3)))
 if free-estimate<int(1.5*1024**3):raise OSError('durable return disk reserve prerequisite')
 factor=linalg.cho_factor(normal,lower=True,check_finite=True);record('NEW-embedded-normal-factor-RETURN',dict(factor=factor[0],lower=factor[1]))
 def checked(rhs,tag):
  out=solve(B,Ds,weight,factor,rhs,record,tag)
  srhs=typed(next(D.glob(f'*-{tag}-Schur-RHS-RETURN.json')))['value'];guard=linear_return_guard(normal,out[:1522],srhs,record,tag)
  if not finite_or_unknown(out,guard['residual'],guard['conditional_RN_outer']) or not guard['all_within']:raise ValueError('UNRESOLVED_actual_NEW_HSD_linear_return_guard')
  return out
 g=adj(A,R,J,b,idx,p,record,'NEW-setup-b')
 wb=checked(g,'NEW-setup-Hinv-Atb');wc=checked(c,'NEW-setup-Hinv-c')
 awb=action(A,R,J,wb,idx,p,record,'NEW-setup-Awb');awc=action(A,R,J,wc,idx,p,record,'NEW-setup-Awc')
 rb=b-awb;record('NEW-setup-whole-row-ridge-residual-RETURN',dict(value=rb))
 squares=[square_dot(a,record,'NEW-setup-'+n) for a,n in [(wc,'wc'),(awc,'Awc'),(wb,'wb'),(rb,'b-Awb')]]
 denominator=1.+sum(squares);record('NEW-setup-positive-bordered-denominator-RETURN',dict(value=denominator,all_four_raw_square_terms=squares,exact_ideal_lower=1.,not_candidate_tolerance=True))
 k=c+g;direction=wc-wb;record('NEW-setup-bordered-fixed-vectors-RETURN',dict(k=k,direction=direction))
 if not finite_or_unknown(g,wb,wc,awb,awc,rb,k,direction,np.asarray(squares)) or not np.isfinite(denominator) or denominator<=0:raise ValueError('UNKNOWN_actual_embedding_denominator')
 del g,wb,wc,awb,awc,rb
 for step in range(1,9):
  if time.monotonic()>deadline:raise TimeoutError('declared method150sec cap')
  tag=f'NEW-HSD-update-{step:02d}';rx=ux+vx;ry=uy+vy;rt=ut+vk;record(tag+'-resolvent-input-BEFORE',dict(rx=rx,ry=ry,rt=rt))
  ay=adj(A,R,J,ry,idx,p,record,tag)
  rhs=rx-ay;record(tag+'-normal-RHS-RETURN',dict(value=rhs))
  v=checked(rhs,tag);dot_by=float(np.dot(b,ry));record(tag+'-b-dot-ry-RETURN',dict(value=dot_by));dot_kv=float(np.dot(k,v));record(tag+'-k-dot-v-RETURN',dict(value=dot_kv))
  numerator=rt+dot_by+dot_kv;record(tag+'-bordered-numerator-RETURN',dict(value=numerator))
  tt=numerator/denominator;record(tag+'-homogeneous-tau-resolvent-RETURN',dict(value=tt))
  tx=v-direction*tt;record(tag+'-primal-resolvent-RETURN',dict(value=tx))
  ax=action(A,R,J,tx,idx,p,record,tag)
  ty=ry+ax-b*tt;record(tag+'-dual-resolvent-RETURN',dict(value=ty))
  px=tx-vx;py=ty-vy;pt=tt-vk;record(tag+'-whole-C-projection-input-BEFORE',dict(x=px,y=py,tau=pt))
  nx,ny,nt,branches=split_projection(px,py,pt,cones);record(tag+'-whole-C-projection-RETURN',dict(x=nx,y=ny,tau=nt,branches=branches))
  nvx=vx-tx+nx;nvy=vy-ty+ny;nvk=vk-tt+nt;record(tag+'-dual-split-update-RETURN',dict(vx=nvx,vy=nvy,kappa=nvk,completed_update=step))
  if not finite_or_unknown(rx,ry,v,tx,ax,ty,nx,ny,nvx,nvy,np.asarray([rt,tt,nt,nvk])):raise ValueError('UNKNOWN_nonfinite_NEW_embedding_return')
  ux,uy,ut=nx,ny,nt;vx,vy,vk=nvx,nvy,nvk;completed=step
  if time.monotonic()>deadline:raise TimeoutError('declared method150sec cap')
 record('NEW-terminal8-homogeneous-state-RETURN',dict(x=ux,y=uy,tau=ut,vx=vx,vy=vy,kappa=vk,terminal_ONLY=True,UNQUALIFIED=True))
 if not np.isfinite(ut) or ut<=0:
  status='UNRESOLVED_fixed8_no_positive_homogeneous_recovery_NOT_infeasibility'
 else:
  recovered=ux/ut;record('NEW-terminal8-raw-embedding-primal-recovery-RETURN',dict(value=recovered,raw_tau=ut,embedding_recovery_NOT_old_return_repair=True,UNQUALIFIED=True))
  if not finite_or_unknown(recovered):raise ValueError('UNKNOWN_nonfinite_terminal_recovery')
  act=action(A,R,J,recovered,idx,p,record,'NEW-terminal8')
  F=act-b;full=lift(recovered[:NF]);record('NEW-terminal8-ALL-hard-affine-residual-RETURN',dict(F=F,full_shadow=full,epigraph_variables=recovered[NF:],phase=full[PHASE]))
  orthant=np.ones(nr,dtype=bool);margins=[]
  for st,wi in cones:
   orthant[st:st+wi]=False;z=F[st:st+wi];margins.append(float(-z[0]-np.linalg.norm(z[1:])))
  margins=np.asarray(margins);ma=float(F[orthant].max(initial=-np.inf));mc=float(margins.min());strict=bool(np.all(F[NOBJ+NROB+582:NOBJ+NROB+776]<0))
  decision=dict(all_finite=finite_or_unknown(F,full,margins),tau=ut,kappa=vk,maximum_affine_F=ma,minimum_negative_F_cone_margin=mc,all2151_cone_margins=margins,
   violated_affine_rows=int(np.count_nonzero(F[orthant]>0)),violated_cones=int(np.count_nonzero(margins<0)),ALL194_strict_depth=strict,exact_zero_phase=bool(full[PHASE]==0.),
   raw_epigraph_objective=float(np.dot(c,recovered)),original_quadratic_objective=float(.5*np.sum(con['p']*recovered[con['idx']]**2)),
   mixed_model_units_NOT_native_metres_or_physical_improvement=True,no_optimality_infeasibility_or_task_exclusion=True)
  record('NEW-terminal8-hard-affine-decision-RETURN',decision)
  passed=decision['all_finite'] and ma<=0 and mc>=0 and strict and decision['exact_zero_phase']
  if not passed:status='UNRESOLVED_fixed8_recovered_hard_primal_REJECTED_before_NEW_nonlinear_query'
  else:
   record('NEW-ONE-terminal8-raw-joint-point-BEFORE-query',dict(w=full,UNQUALIFIED=True));point_sealed=True
   api=JointExecutableInterface(I);record('NEW-runtime-constructor-RETURN',api.__dict__)
   Fnew,Jnew,Enew=method_oracle(api,full,record,'NEW-ONE-terminal8',deadline);full_query=True
   motion=api.actual_object_motion(full[:1523]);record('NEW-ONE-terminal8-ALL6-actual-motion-RETURN',motion)
   status='UNQUALIFIED_NEW_nonlinear_return_PENDING_all_independent_original_gates'
except BaseException as e:
 record('NEW-actual-method-boundary-RETURN',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),completed_updates=completed,NEW_point_sealed=point_sealed,complete_NEW_query=full_query,missing_returns_NOT_reconstructed=True))
 save(D/'report.json',dict(status='UNRESOLVED_declared_method_limit' if isinstance(e,TimeoutError) else 'UNRESOLVED_actual_method_exception',wall_seconds=time.monotonic()-start,completed_updates=completed,
  NEW_point_sealed=point_sealed,complete_NEW_query=full_query,no_retry_repair_or_qualification=True))
else:
 save(D/'report.json',dict(status=status,wall_seconds=time.monotonic()-start,completed_updates=completed,terminal_update_ONLY=8,NEW_point_sealed=point_sealed,complete_NEW_query=full_query,
  original_F_components_gates_parameters_8self5object_UNCHANGED=True,no_actual_training_NN_raster_integercells_fullpixel_heldout_native_self_swept_motor_grasp_physics_GPU_PPO=True,
  no_reference_component_trajectory_task_policy_AUC_qualification=True))

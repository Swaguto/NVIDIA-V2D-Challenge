"""NEW private zero-target checkpoint, integrity and provider bookkeeping."""
from pathlib import Path
import json,hashlib,datetime,subprocess,shutil
S=Path(__file__).resolve().parents[1];C=S/'compute';R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');now=datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
parent=C/'joint-proximal-checkpoint-v1.json';ppi=C/'private-integrity-joint-proximal-v1.json';prior=json.loads(ppi.read_text());files=prior['files'].copy()
A=R/'experiments/scored_baseline_20261019/joint_zero_target_conic';manifest=json.loads((A/'exact-code-plan-manifest.json').read_text())
outputs=['e037-joint-zero-contracts-v1','e037-joint-zero-preflight-v1','e037-joint-zero-binding-v1','e037-joint-zero-return-contract-v1','e037-joint-zero-evaluation-v1','e037-joint-zero-return-audit-v1']
paths=[parent,ppi,C/'joint-zero-conic-startup-integrity-v1.json',C/'provider-status-after-joint-zero-conic-v1.json',Path(__file__),*[C/n for n in manifest['files']]]
for out in outputs:paths.extend(p for p in (S/out).rglob('*') if p.is_file())
added=0
for p in paths:
 k=str(p.relative_to(S));v=dict(sha256=sha(p),bytes=p.stat().st_size)
 if k in files:assert files[k]==v
 else:added+=1
 files[k]=v
np=C/'private-integrity-joint-zero-conic-v1.json';cp_path=C/'joint-zero-conic-checkpoint-v1.json'
save(np,dict(schema='v2d.NEW-joint-zero-conic-private-integrity.v1',created_utc=now,parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),parent_integrity_receipt=str(ppi),parent_integrity_receipt_sha256=sha(ppi),
 parent13541files_preserved_BY_HASH=True,current_relevant1343files_reverified_receipt=str(C/'joint-zero-conic-startup-integrity-v1.json'),new_files_freshly_hashed=added,verified_files=len(files),files=files))
guards={o:json.loads((S/o/'guard.json').read_text()) for o in outputs};method=json.loads((S/outputs[4]/'report.json').read_text());audit=json.loads((S/outputs[5]/'report.json').read_text());assert audit['passed']
pr=C/'provider-status-after-joint-zero-conic-v1.json';provider=json.loads(pr.read_text());assert provider['current_provider_status_verified'] and provider['provider_status']=='STOPPED'
NEXT='''NEXT: Develop ONE genuinely NEW homogeneous self-dual whole-cone hard joint primal construction with the ORIGINAL zero-phase target explicit. This is a new embedding/primal-recovery construction, NOT continuation/retry/tuning of finished16-update KKT splitting, one-step GN, explicit two-update method, old object ADMM/SLSQP or unavailable materialized constructor. Derive only concrete embedding/sign/primal-recovery/whole-cone/ALL-correlated-tie/continuous-direction/original-Pq/conditioning/resource/typed-return contracts and independent complete NEW maps before ONE separately declared bounded evaluation. No new solver, optimizer, point, episode/native/physical/GPU candidate is currently selected/queued. Do not spend another phase rebuilding interfaces, unrelated arithmetic inventories or varying iteration limits/steps/weights/boxes/tolerances/profiles.
Reuse complete immutable original joint matrices/interfaces/source/clock maps and exact SAVED F query/numerical certificate BY HASH only. F remains sole verified start, SAME saved wrist/middle hypotheses remain UNQUALIFIED; Fleft/early-latethumb/ALLmissingboth-handcommands remain exact. Preserve ALL1523 joint/object coordinates, continuous geometric directions, original4810140rows/1245cones/249603pairscopes, ALL8114robotrows/194source/38domains/fullregion2/three unknown bands, ALL compiled vertices/correlated ties, original RHS/Pq/conditioning/strict flags/native-stage-Cayley-order/ORIGINALsupportqpos0. Exact phase==0 is the already-required qualified-target slice, not a repaired phase, new authority/box or family/task exclusion. No selected cut/pair/row/axis/face/vertex/rank/pose bank or broadphase drop; native MJB and original URDF remain separate operators unchanged.
The finished hard affine KKT construction has16updates/terminal16ONLY, zero literal phase,3316affine/168cone failures and no new nonlinear point/query. Its shadow/primal/split/projected/dual/intermediate states stay UNQUALIFIED and unchanged, never warmstarts, commands or selected candidates. Six guards,10NEWcontrols, complete bindings, sole factor/16linear solves/16whole-cone returns and independent saved-return audit are COMPLETE; never rerun/extend/tune/repair/normalize/clip/restart/continue them.1745exact-dyadic projected SOC defects and six finite-return matmul warnings remain raw; zero exact branch-condition failures does not make projections certificates. Cause/future reliability remains unproven. The affine model failure is not stationarity/infeasibility/global source-family/task exclusion. Current normal/resolvent/bounds are SEARCH/numerical-construction operators, not hard physical feasibility certificates or candidate tolerances; saved RN proof only covers its actual saved inputs, never future ties/global binary/source/native identity.
Every genuinely NEW input/constructor/proposal/operator/solver/intermediate RETURN must be durable typed-lossless BEFORE assertions/further queries. No old missing constructor/internal state/native returns reconstructed. Cover each new actual SEARCH/tie return before an update. Keep all original whole151source/actualURDF/object-root-finger component/adjacent/second/actualvelocity/root-motion/worldboxes/modelorientation/angularp95 and object1e-9motion, source-fullpixel-heldoutPOST1e-10, actual integercells/strictdepth/changedNN-rank/roundedmedian/quality/fulltriangletraining and FULL1mmnativehand-object-supportANDself/fullswept/actualmotor-grasp gates beforeGPU/PPO. Fullpixel/heldout strictlyPOST, never fitting/rescue/selection feedback. Object-only changes cannot clear8self. Owner actual trajectory qualification remains ACTIVE/UNACHIEVED.'''
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
cp=dict(schema='v2d.NEW-joint-zero-conic-checkpoint.v1',created_utc=now,parent_checkpoint=str(parent),private_integrity_checkpoint=str(np),private_files_verified=len(files),CPU_jobs=len(guards),
 process_successes=sum(g['worker_exit_code']==0 for g in guards.values()),process_failures=sum(g['worker_exit_code']!=0 for g in guards.values()),completed_CPU_guards=guards,NEW_passed_contracts=10,
 separate_preparation_patch_header_rejection_BEFORE_file_module_worker_guard=True,
 archive=dict(path=str(A),exact_code_plan_metadata_files=len(manifest['files']),manifest_sha256=sha(A/'exact-code-plan-manifest.json')),production_runtime_tests_UNCHANGED=True,
 complete_preflight=json.loads((S/outputs[1]/'report.json').read_text()),complete_independent_binding=json.loads((S/outputs[2]/'report.json').read_text()),method=method,independent_saved_return_audit=audit,
 hard_zero_phase_affine_slice_preserves_original_phase_coordinate_domain_Pq_all_rows=True,all_returned_terminal_and_intermediate_states_UNQUALIFIED=True,
 no_NEW_nonlinear_source_geometry_episode_native_physics_motion_query=True,no_actual_training_NN_raster_integercells_fullpixel_heldout_native_self_swept_motor_grasp_GPU_PPO=True,
 all_source_physics_parameters_gates_F_components_1072anchors_missing136_heldout31of32_8self5object_UNCHANGED=True,no_reference_trajectory_task_policy_AUC_qualification=True,
 provider_receipt=str(pr),provider_status='STOPPED',provider_observed_utc=provider['checked_utc'],provider_attempt_started_utc=provider['attempt_started_utc'],provider_attempt_finished_utc=provider['attempt_finished_utc'],
 no_VM_mutation_or_paid_queue=True,read_success_NOT_future_lifecycle_reliability=True,repo_head=head,pushed=True,owner_goal_achieved=False,owner_continuation_intent='active/unachieved',
 historical_goal_tool_status='blocked',goal_tool_observation_utc='2026-10-20T15:44:42.193354+00:00',research_checkpoint_NOT_goal_completion=True,next_work=NEXT)
save(cp_path,cp)
rows=subprocess.check_output(['ps','-axo','pid=,comm=,args='],text=True).splitlines();active=[];workers=[str(C/n) for n in manifest['files'] if n.endswith('.py') and n.startswith(('verify_','build_','bind_','evaluate_','audit_'))]
for row in rows:
 parts=row.strip().split(None,2)
 if len(parts)==3 and any(w in parts[2] for w in workers):active.append(dict(pid=parts[0],basename=Path(parts[1]).name))
assert not active
idle=C/'idle-after-joint-zero-conic-v1.json';save(idle,dict(checked_utc=now,filtered_related_workers=active,all_related_workers_idle=True,free_disk_bytes=shutil.disk_usage(S).free,no_VM_mutation_or_paid_queue=True,provider_receipt=str(pr),provider_observed_utc=provider['checked_utc']))
for p in [S/'phase_state_20261018.json',C/'controller-lifecycle-runtime-ledger-v1.json']:
 d=json.loads(p.read_text());assert not d.get('active_cpu_only_job') and not d.get('active_cpu_work')
 d.update(latest_quality_checkpoint=str(cp_path),latest_private_files_verified=len(files),latest_code_only_repo_head=head,latest_provider_receipt=str(pr),latest_provider_attempt=str(pr),current_provider_status_verified=True,
  provider_status='STOPPED',provider_observed_utc=provider['checked_utc'],provider_attempt_started_utc=provider['attempt_started_utc'],provider_attempt_finished_utc=provider['attempt_finished_utc'],
  owner_continuation_intent='active/unachieved',owner_trajectory_qualification_achieved=False,trajectory_goal_achieved=False,training_approved=False,research_checkpoint_NOT_owner_goal_completion=True,
  updated_utc=now,latest_idle_reconciliation=str(idle),pending_quality_preparation=NEXT)
 b=d.get('brev_current',{});b.update(provider_status='STOPPED',checked_utc=provider['checked_utc'],current_provider_status_verified=True,latest_provider_receipt=str(pr),last_confirmed_provider_receipt=str(pr),latest_provider_attempt=str(pr),
  last_confirmed_provider_status='STOPPED',last_confirmed_observed_utc=provider['checked_utc'],status_attempt_started_utc=provider['attempt_started_utc'],status_attempt_finished_utc=provider['attempt_finished_utc'],
  paid_work_queued=False,latest_provider_error=None,successful_current_read_required_before_GPU=True,no_auth_cause_inferred=True);d['brev_current']=b
 if 'queued_cpu' in d:d['queued_cpu']=[]
 save(p,d)
print(json.dumps(dict(checkpoint=str(cp_path),private_files=len(files),new_files=added,CPU_jobs=len(guards),passed_contracts=10,head=head,provider_observed_utc=provider['checked_utc'])))

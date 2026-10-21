"""NEW HSD resource-boundary checkpoint, integrity and lifecycle bookkeeping."""
from pathlib import Path
import json,hashlib,datetime,subprocess,shutil
S=Path(__file__).resolve().parents[1];C=S/'compute';R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');now=datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
parent=C/'joint-zero-conic-checkpoint-v1.json';ppi=C/'private-integrity-joint-zero-conic-v1.json';files=json.loads(ppi.read_text())['files'].copy()
A=R/'experiments/scored_baseline_20261019/joint_homogeneous_embedding';manifest=json.loads((A/'exact-code-plan-manifest.json').read_text())
outputs=['e037-joint-hsd-contracts-v1','e037-joint-hsd-preflight-v1','e037-joint-hsd-binding-v1','e037-joint-hsd-evaluation-v1','e037-joint-hsd-boundary-audit-v1']
paths=[parent,ppi,C/'joint-hsd-startup-integrity-v1.json',C/'joint-hsd-storage-request-v1.json',C/'provider-status-after-joint-hsd-v1.json',*[C/n for n in manifest['files']]]
for out in outputs:paths.extend(p for p in (S/out).rglob('*') if p.is_file())
added=0
for p in paths:
 k=str(p.relative_to(S));v=dict(sha256=sha(p),bytes=p.stat().st_size)
 if k in files:assert files[k]==v
 else:added+=1
 files[k]=v
pi_path=C/'private-integrity-joint-hsd-v1.json';cp_path=C/'joint-hsd-checkpoint-v1.json'
save(pi_path,dict(schema='v2d.NEW-joint-hsd-private-integrity.v1',created_utc=now,parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),parent_integrity_receipt=str(ppi),parent_integrity_receipt_sha256=sha(ppi),
 parent14583files_preserved_BY_HASH=True,current_relevant1084files_reverified_receipt=str(C/'joint-hsd-startup-integrity-v1.json'),new_files_freshly_hashed=added,verified_files=len(files),files=files))
guards={o:json.loads((S/o/'guard.json').read_text()) for o in outputs};method=json.loads((S/outputs[3]/'report.json').read_text());audit=json.loads((S/outputs[4]/'report.json').read_text());assert audit['passed']
pr=C/'provider-status-after-joint-hsd-v1.json';provider=json.loads(pr.read_text());assert provider['current_provider_status_verified'] and provider['provider_status']=='STOPPED'
NEXT='''NEXT: Storage capacity is the concrete pending prerequisite. A user-input request is already pending: free about2GB locally or provide a private writable volume with at least7GB free. Check any human response and current capacity read-only. Never delete/move private evidence, alter unrelated projects/preview/cache, change reserves, retryWatcloud/NAS, start GPU or silently choose a new output volume. Stay quiet while the same missing-input condition persists; do not repeatedly ask or notify. Owner intent remains active/unachieved, no fabricated goal-tool resume/completion/pause. Historicalblocked tool observation stays separate.
The HSD embedding/maps/10controls/independentbindings and SOLEdeclared8-update method attempt are FINISHED. That attempt stops BEFOREfactorization/solve/denominator/projection/update/recovery:0updates/.948039333sec, free5440933888bytes versus declaredreturns4787907456+requiredreserve1610612736=6398520192bytes, shortfall957586304. There is NO numeric factor/solver/iterate/candidate return because these stages were never called, not lost/reconstructed state. Do not rerun/restart/continue this completed attempt or change its8updates/150sec/resource checks. No policy/source-family/task infeasibility follows from disk shortage. Completed contracts/maps are reusable BY HASH for genuinely NEW declared inputs/operators only; no method retry disguised by a new path/name/storage policy, old failed vector/dual/Gram warmstart, tuning/sweep or numerical-return reconstruction.
After sufficient storage and applicable owner direction, develop ONE genuinely NEW resource-feasible whole151 joint primal candidate construction, deriving only concrete new method-specific operators/contracts and complete independent new bindings before ONE separatelydeclaredbounded evaluation. Do not rebuild old interfaces/maps/controls or spend a phase on generic arithmetic inventories. No new solver/optimizer/candidate/physical/native/GPU evaluation is selected or queued. Preserve originalPq/phase coordinate/domain/requiredzero-targetslice, all1523joint-object and continuousdirection coordinates/all4810140originalrows/1245originalcones/249603pairs/ALLcompiledvertices/correlatedties/8114robot/194source/38domains/fullregion2/threeunknownbands/source-clock-RHS-conditioning-strictflags/nativeMJB-versusURDF/stage-Cayley-order/originalsupportqpos0. New epigraph/HSD auxiliaries are only optimization variables, not observations/source authority/commands/physical phase or gates. Saved arithmetic proof is conditional on saved inputs only, never future/global binary/source/native identity or candidate tolerance.
F remains sole verified start, SAMEsaved wrist/middle UNQUALIFIED, Fleft/early-latethumb/ALLmissingcommands exact. No old cohort/model/source/geometry/NN/raster/native/method/solver/audit replay, selected pair/cut/axis/face/vertex/rank/posebank, broadphase drop, morphology/physics/effort/environment/gate change. All genuinelyNEWinput/constructor/proposal/operator/solver/intermediateRETURNs durable typed-lossless BEFORE checks/further queries; cover actual new SEARCH/tie returns before updates. Whole151source/actualURDF/object-root-finger component/adjacent/second/actualvelocity/rootmotion/worldboxes/modelorientation/angularp95/object1e-9motion/source-fullpixel-heldoutPOST1e-10 and actualintegercells/strictdepth/changedNN-ranks/roundedmedian/quality/fulltriangletraining/FULL1mmnativehand-object-supportANDself/fullswept/motor-grasp remain beforeGPU/PPO. Fullpixel/heldout strictlyPOST, neverfit/rescue/select. Object-only changes cannot clear8self. Actualtrajectorygoal remainsACTIVE/UNACHIEVED.'''
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
cp=dict(schema='v2d.NEW-joint-hsd-checkpoint.v1',created_utc=now,parent_checkpoint=str(parent),private_integrity_checkpoint=str(pi_path),private_files_verified=len(files),CPU_jobs=len(guards),
 process_successes=sum(g['worker_exit_code']==0 for g in guards.values()),process_failures=sum(g['worker_exit_code']!=0 for g in guards.values()),completed_CPU_guards=guards,NEW_passed_contracts=10,
 archive=dict(path=str(A),exact_code_plan_files=len(manifest['files']),manifest_sha256=sha(A/'exact-code-plan-manifest.json')),production_runtime_tests_UNCHANGED=True,
 complete_preflight=json.loads((S/outputs[1]/'report.json').read_text()),complete_independent_binding=json.loads((S/outputs[2]/'report.json').read_text()),method=method,independent_boundary_audit=audit,
 method_stops_BEFORE_any_factorization_solve_operator_projection_update_recovery=True,required_storage_user_input_pending=str(C/'joint-hsd-storage-request-v1.json'),
 original_Pq_exact_ideal_square_root_free_epigraph=True,no_NEW_candidate_source_geometry_native_motion_physics_training_GPU_query=True,
 F_components_parameters_gates_1072anchors_missing136_heldout31of32_8self5object_UNCHANGED=True,no_reference_trajectory_task_policy_AUC_qualification=True,
 provider_receipt=str(pr),provider_status='STOPPED',provider_observed_utc=provider['checked_utc'],repo_head=head,pushed=True,no_VM_mutation_or_paid_queue=True,
 owner_goal_achieved=False,owner_continuation_intent='active/unachieved',historical_goal_tool_status='blocked',goal_tool_observation_utc='2026-10-20T15:44:42.193354+00:00',research_checkpoint_NOT_goal_completion=True,next_work=NEXT)
save(cp_path,cp)
rows=subprocess.check_output(['ps','-axo','pid=,comm=,args='],text=True).splitlines();workers=[str(C/n) for n in manifest['files'] if n.endswith('.py') and n.startswith(('verify_','build_','bind_','evaluate_','audit_'))];active=[]
for row in rows:
 a=row.strip().split(None,2)
 if len(a)==3 and any(w in a[2] for w in workers):active.append(dict(pid=a[0],basename=Path(a[1]).name))
assert not active
idle=C/'idle-after-joint-hsd-v1.json';save(idle,dict(checked_utc=now,filtered_related_workers=active,all_related_workers_idle=True,free_disk_bytes=shutil.disk_usage(S).free,no_VM_mutation_or_paid_queue=True,provider_receipt=str(pr),provider_observed_utc=provider['checked_utc']))
for p in [S/'phase_state_20261018.json',C/'controller-lifecycle-runtime-ledger-v1.json']:
 d=json.loads(p.read_text());assert not d.get('active_cpu_only_job') and not d.get('active_cpu_work')
 d.update(latest_quality_checkpoint=str(cp_path),latest_private_files_verified=len(files),latest_code_only_repo_head=head,latest_provider_receipt=str(pr),latest_provider_attempt=str(pr),current_provider_status_verified=True,
 provider_status='STOPPED',provider_observed_utc=provider['checked_utc'],provider_attempt_started_utc=provider['attempt_started_utc'],provider_attempt_finished_utc=provider['attempt_finished_utc'],owner_continuation_intent='active/unachieved',
 owner_trajectory_qualification_achieved=False,trajectory_goal_achieved=False,training_approved=False,research_checkpoint_NOT_owner_goal_completion=True,updated_utc=now,latest_idle_reconciliation=str(idle),pending_quality_preparation=NEXT,
 required_storage_input_pending=str(C/'joint-hsd-storage-request-v1.json'))
 b=d.get('brev_current',{});b.update(provider_status='STOPPED',checked_utc=provider['checked_utc'],current_provider_status_verified=True,latest_provider_receipt=str(pr),last_confirmed_provider_receipt=str(pr),latest_provider_attempt=str(pr),
 last_confirmed_provider_status='STOPPED',last_confirmed_observed_utc=provider['checked_utc'],status_attempt_started_utc=provider['attempt_started_utc'],status_attempt_finished_utc=provider['attempt_finished_utc'],paid_work_queued=False,
 latest_provider_error=None,successful_current_read_required_before_GPU=True,no_auth_cause_inferred=True);d['brev_current']=b
 if 'queued_cpu' in d:d['queued_cpu']=[]
 save(p,d)
print(json.dumps(dict(checkpoint=str(cp_path),private_files=len(files),new_files=added,CPU_jobs=len(guards),passed_contracts=10,actual_updates=0,provider_observed_utc=provider['checked_utc'],head=head)))

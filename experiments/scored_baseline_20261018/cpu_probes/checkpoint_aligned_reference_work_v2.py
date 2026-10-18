"""Seal all closed CPU repairs and reconcile phase/ledger; no GPU command."""
import datetime,hashlib,json,subprocess
from pathlib import Path
S=Path(__file__).resolve().parents[1];C=S/'compute';R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
now=datetime.datetime.now(datetime.timezone.utc).isoformat();read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
validation=read(S/'plane-aligned-articulated-native-validation-v2/report.json');sweep=read(S/'plane-aligned-reference-sweep-v2/report.json');pack=read(C/'e037-aligned-reference-candidate-pack-v1.json')
assert validation['complete_static_source_contact_motion_stage_passed'] and sweep['discrete_interpolated_contact_gate_passed']
provider=read(C/'provider-status-after-login-refresh-v3.json');assert provider['provider_status']=='STOPPED'
# Retain prior verified history and add every file in the new closed diagnostics.
old=read(C/'cpu-fast-stereo-quality-checkpoint-v2.json');paths={S/n for n in old['files']}
patterns=['fast-stereo-full-object-*','source-pose-pool-repair-v*','source-pose-temporal-path-v*','source-support-failure-window-v*','plane-aligned-*','full-static-support-observations-v*','smooth-plane-camera-alignment-v*']
for pattern in patterns:
 for root in S.glob(pattern):
  if root.is_dir():paths.update(p for p in root.rglob('*') if p.is_file() and not p.is_symlink())
for p in C.glob('*.py'):
 if any(x in p.name for x in ['plane_aligned','temporal_pose','temporal_pool','source_pose_pool','source_support_failure','observe_full_static','smooth_plane','aligned_reference','swept_hand']):paths.add(p)
for name in ['e037-aligned-reference-candidate-v1.private.tar.gz','e037-aligned-reference-candidate-pack-v1.json','e037-aligned-reference-gpu-plan-v1.json','provider-status-after-login-refresh-v3.json','brev-login-refresh-and-stop-preflight-v3.json']:paths.add(C/name)
for n,h in old['files'].items():assert sha(S/n)==h['sha256'],'Prior verified file changed: '+n
files={str(p.relative_to(S)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(paths)}
for n,h in files.items():assert sha(S/n)==h['sha256']
checkpoint={'schema':'v2d.cpu-source-contact-quality-checkpoint.v3','created_utc':now,'files':files,'verified_files':len(files),'independent_off_machine_backup':False,'new_paid_sessions':0,'new_training_steps':0,'complete_static_reference_stage_passed':True,'independent_discrete_sweep_passed':True,'physical_task_accepted':False}
(C/'cpu-source-contact-quality-checkpoint-v3.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
phase=read(S/'phase_state_20261018.json');phase.update(updated_utc=now,active_cpu=None,active_gpu=None,active_gpu_session=None,queued_gpu=[])
phase['latest_results']['e037_aligned_independent_static_reference_v2']={k:validation[k] for k in ['candidate_manifest_sha256','candidate_sha256','native_contact_gate','independent_source_projection','inferred_motion_bounds','motion_limits_stage_passed','complete_static_source_contact_motion_stage_passed','physical_task_accepted']}
phase['latest_results']['e037_final_discrete_sweep_v2']={k:sweep[k] for k in ['candidate_manifest_sha256','samples','interior_samples','source_knots','failed_samples','maximum_sample_spacing_s','discrete_interpolated_contact_gate_passed','continuous_collision_certified','actual_policy_motion_assessed','contact_forces_assessed','physical_task_accepted']}
for version in [3,4,5]:
 p=S/f'plane-aligned-articulated-trajectory-v{version}';g=read(p/'guard.json');assert 'completed_utc' in g
 r=read(p/'report.json');phase['latest_results'][f'e037_swept_hand_v{version}']={'output':p.name,'guard':g,'hand_source_contact_motion_stage_passed':r['hand_source_contact_motion_stage_passed'],'sweep_final':r['sweep_final'],'last_solver_status':r['iterations'][-1]['status'],'physical_task_accepted':False}
phase['outstanding']=[
 'Brev normal login/listing restored; last provider read20:39:46UTC confirms STOPPED. STOP remains HTTP400 invalid-security-token; service-disruption banner observed. No START/paid work while shutdown control broken. Do not retry blindly, create another VM or send external support messages.',
 'e037 new inferred reference now qualifies all151sourceobject fits, unchanged motion maxima/p95, fresh static1mm native FK/contact/source/actualURDF gates, and independent3805sample discrete interpolation. Latest candidate privatepack and GPUplan are sealed. No grasp force, controlled dynamics, new training, leaderboard score or continuouscollision certificate.',
 'Preserve static V1 sweep failure11samples/3.961mm between67/68 and rejectedV3/V4SQP proposals. V5 local proximal refinement repaired tested interpolation under same source/contact/morph/motion acceptance gates. No source or confidence upgrades;14planes remain inferred.',
 'All20selectedbaselinepolicies remain MP-SR0. Packedbaseline/replaypacks unchanged. Do not rerun completedcohorts/replays/finalizers/renderers or retune reserved14/32/36. No artifact promotion or submission.',
 'GPU newcandidate full unassisted zeroresidual controller/motor/contact test comes before furtherPPO. Use sealed prepared plan only after stopcontrol restored. Then short25iteration chunk if physics review passes and independent repeat before performanceclaim. New inferred reference changes localAUC itself; no acrossreference scorepromise.',
 'Replay startup variability remains unproven; passive tracing CPU tested, GPU integration pending. Use next justified newcandidate run; do not rerun completedreplay.',
 'Independent teammate/freshhost reproduction still requires human participation. No external messages, collaborator grants, merges, leaderboardupload, privatepack or weight publication, or subagents. Heartbeat remainsACTIVE.'
]
phase['queued_cpu']=['No live CPU work. First reconcile provider/read/stop control and phase receipts; all currentCPUprobes are CLOSED. Do not rerun finished jobs. GPU smoke plan is prepared, NOT QUEUED.','If cloud control remains unavailable, useful next CPU investigation should target source-supported hand/object contact geometry or other nonreserved first failures, preserving measured-vs-inferred flags. Do not assume contact-free geometry establishes a force-bearing grasp.']
phase['compute_current'].update(provider_status='STOPPED',last_independent_status_check_utc=provider['checked_utc'],latest_provider_receipt='compute/provider-status-after-login-refresh-v3.json',latest_cpu_checkpoint='compute/cpu-source-contact-quality-checkpoint-v3.json',paid_work_queued=False)
phase['mac_awake_requirement'].update(required=False,reason='No current CPU or GPU jobs; next heartbeat supervision requires Mac awake, but no paid lease is active.')
phase['new_private_candidate']={'pack':str((C/pack['pack']).relative_to(S)),'sha256':pack['pack_sha256'],'bytes':pack['pack_bytes'],'uploaded':False,'gpu_plan':'compute/e037-aligned-reference-gpu-plan-v1.json','gpu_plan_queued':False}
phase['verification']['latest_e037_reference']='Freshcompiled151knots:0object/0handpenetration;904sourcepointsnonregressing;all14URDFlimits;independent3805interpolationstatesclear. Static/discreteonly;no physics/training.'
(S/'phase_state_20261018.json').write_text(json.dumps(phase,indent=2)+'\n')
ledger=read(C/'controller-lifecycle-runtime-ledger-v1.json');ledger.update(latest_update_utc=now,active_cpu_only_job=None,provider_status='STOPPED',latest_provider_receipt='provider-status-after-login-refresh-v3.json')
ledger['cpu_only_quality_checkpoints'].append({'name':'e037-full-static-and-discrete-source-contact-pass-v2','updated_utc':now,'new_paid_sessions':0,'new_training_steps':0,'source_knots':151,'source_points':904,'independent_static_gate_passed':True,'independent_discrete_samples':3805,'independent_discrete_failed_samples':0,'local_hash_verified_files':len(files),'independent_off_machine_backup':False,'local_receipt':'cpu-source-contact-quality-checkpoint-v3.json','physical_task_accepted':False,'provider_status':'STOPPED'})
(C/'controller-lifecycle-runtime-ledger-v1.json').write_text(json.dumps(ledger,indent=2)+'\n')
print(json.dumps({'closed_cpu_hash_verified_files':len(files),'active_jobs':None,'provider_status':'STOPPED','private_pack_bytes':pack['pack_bytes'],'gpu_work_queued':False}))

"""Reconcile closed CPU diagnostics and real provider receipt; no cloud work."""
import datetime,hashlib,json,subprocess
from pathlib import Path
S=Path(__file__).resolve().parents[1];C=S/'compute';R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
fit=read(S/'plane-aligned-pose-temporal-path-v4/report.json');validation=read(S/'plane-aligned-articulated-native-validation-v1/report.json');sweep=read(S/'plane-aligned-reference-sweep-v1/report.json')
provider=read(C/'provider-status-after-login-refresh-v3.json');auth=read(C/'brev-login-refresh-and-stop-preflight-v3.json')
phase=read(S/'phase_state_20261018.json')
phase.update(updated_utc=now,active_cpu=None,queued_cpu=[],active_gpu=None,active_gpu_session=None,queued_gpu=[])
phase['latest_results']['e037_aligned_object_source_motion']={'output':'plane-aligned-pose-temporal-path-v4','all_frames_source_qualified':fit['all_source_fits_qualified'],'motion_quality_certified':fit['motion_quality_certified'],'motion':fit['motion'],'source_confidence_upgraded':False,'physical_task_accepted':False}
phase['latest_results']['e037_aligned_independent_static_reference']={'output':'plane-aligned-articulated-native-validation-v1','complete_static_source_contact_motion_stage_passed':validation['complete_static_source_contact_motion_stage_passed'],'native_contact_gate':validation['native_contact_gate'],'independent_source_projection':validation['independent_source_projection'],'motion_limits_stage_passed':validation['motion_limits_stage_passed'],'source_plane_inferred_only_frames':validation['source_plane_inferred_only_frames'],'physics_stepped':False,'new_training_steps':0,'physical_task_accepted':False}
phase['latest_results']['e037_between_knot_sweep_v1']={k:v for k,v in sweep.items() if k not in ['failures','near_spoon_bodies']}
active=[]
for version in [3,4]:
 p=S/f'plane-aligned-articulated-trajectory-v{version}'
 if not (p/'guard.json').exists():continue
 g=read(p/'guard.json')
 if 'completed_utc' not in g:active.append({'job':p.name,'guard':str((p/'guard.json').relative_to(S)),'worker_pid':g['worker_pid'],'wall_cap_seconds':g['wall_limit_seconds'],'paid_gpu':False})
 else:
  r=read(p/'report.json');phase['latest_results'][f'e037_swept_hand_v{version}']={'output':p.name,'guard':g,'hand_source_contact_motion_stage_passed':r.get('hand_source_contact_motion_stage_passed',False),'sweep_final':r.get('sweep_final'),'last_solver_status':r.get('iterations',[{}])[-1].get('status'),'physical_task_accepted':False}
assert len(active)<=1,'Unexpected overlapping CPU jobs'
phase['active_cpu']=active[0] if active else None
cc=phase['compute_current'];cc.update(provider_status=provider.get('provider_status','UNAVAILABLE'),last_independent_status_check_utc=provider['checked_utc'],latest_provider_receipt=str((C/'provider-status-after-login-refresh-v3.json').relative_to(S)),normal_nvidia_login_refresh_succeeded=True,latest_error='Login and listing restored. STOP preflight still HTTP400 invalid provider security token. Brev console reports cross-provider service disruption. No START attempted.',stop_preflight_after_both_auth_refreshes='Still UnrecognizedClientException after latest normal login',paid_work_queued=False)
phase['outstanding']=[
 'Brev normal login/listing restored using owner email; latest provider receipt confirms STOPPED. Stop control still HTTP400 invalid-security-token, console reports service disruption. No START or paid work while shutdown control broken. No repeated blind retries or external message.',
 'e037 all151 inferred object source fits and original motion caps PASS. Fresh aligned articulated151-knot static native/source/actualURDFmotion gate PASS (zero hand/object penetration). Original source/confidence and14inferred planes preserved. This is not grasp or physics success.',
 'Additional discrete<=2ms interpolation check FAILS11samples between67/68, max3.961mm right middle finger/spoon. Correct before policy trial; retain static success and swept failure separately. New bounded solver recorded in active_cpu/results.',
 'All20 selected baseline policies still MP-SR0; no artifact promotion, leaderboard upload or reserved-task retuning.',
 'Replay startup variability remains unproven; passive tracing CPU tested, not yet GPU integrated. Do not rerun completed replay/cohorts/renderers/finalizers.',
 'Independent teammate/freshhost reproduction needs later human participation; private packs/assets/weights not published. No external messages, grants, subagents, merge or submission.',
 'Heartbeat remains ACTIVE. Follow live phase/lifecycle/processes and unchanged1mm native gate. Keep paid compute stopped when idle.'
]
phase['queued_cpu']=['Reconcile active bounded swept-hand solver first. Preserve failures and do not relax source/contact/morph/motion bounds. A successful solver needs NEW independent freeze/compiler/native/source/URDF validation plus NEW dense discrete sweep before physics.', 'Once geometry/pose/robot source and contact checks qualify, prepare one bounded GPU fresh native/reference smoke and short policy test only after actual Brev STOP control works. Backup private result hashes and confirm STOPPED. Do not change official evaluator.']
phase['mac_awake_requirement']['reason']='Active local bounded CPU solver under caffeinate' if active else 'No live CPU/GPU job; next work reads phase and receipts first.'
(S/'phase_state_20261018.json').write_text(json.dumps(phase,indent=2)+'\n')
ledger=read(C/'controller-lifecycle-runtime-ledger-v1.json');ledger.update(latest_update_utc=now,provider_status=provider.get('provider_status','UNAVAILABLE'),latest_provider_receipt='provider-status-after-login-refresh-v3.json',latest_auth_refresh_note=cc['latest_error'],active_cpu_only_job=active[0] if active else None)
ledger.setdefault('cpu_only_quality_checkpoints',[])
name='e037-source-motion-static-qualified-between-knot-rejected-v1'
if not any(x['name']==name for x in ledger['cpu_only_quality_checkpoints']):ledger['cpu_only_quality_checkpoints'].append({'name':name,'updated_utc':now,'new_paid_sessions':0,'new_training_steps':0,'source_knots':151,'independent_static_gate_passed':True,'static_native_object_failures':0,'static_native_hand_failures':0,'observed_source_points':904,'discrete_sweep_samples':sweep['samples'],'discrete_sweep_failed_samples':sweep['failed_samples'],'maximum_swept_penetration_m':sweep['maximum_penetration_by_pair_m'].get('right/spoon'),'physical_task_accepted':False,'provider_status':provider.get('provider_status','UNAVAILABLE')})
(C/'controller-lifecycle-runtime-ledger-v1.json').write_text(json.dumps(ledger,indent=2)+'\n')
print(json.dumps({'updated_utc':now,'active_cpu':phase['active_cpu'],'provider':cc['provider_status'],'static_stage_pass':True,'dense_sweep_pass':False}))

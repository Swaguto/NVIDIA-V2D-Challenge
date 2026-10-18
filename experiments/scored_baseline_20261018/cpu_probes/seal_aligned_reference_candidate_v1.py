"""Seal the private candidate and a bounded, unqueued GPU test specification."""
import datetime,hashlib,json,tarfile
from pathlib import Path
S=Path(__file__).resolve().parents[1];C=S/'compute';P=S/'plane-aligned-articulated-native-validation-v2/frozen'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
validation=read(P.parent/'report.json');sweep=read(S/'plane-aligned-reference-sweep-v2/report.json')
assert validation['complete_static_source_contact_motion_stage_passed'] and sweep['discrete_interpolated_contact_gate_passed']
assert validation['candidate_manifest_sha256']==sha(P/'manifest.json')==sweep['candidate_manifest_sha256']
m=read(P/'manifest.json');files={}
for name,h in m['files'].items():
 path=P/name;assert path.resolve().is_relative_to(P.resolve()) and sha(path)==h
 files[name]={'sha256':h,'bytes':path.stat().st_size}
files['manifest.json']={'sha256':sha(P/'manifest.json'),'bytes':(P/'manifest.json').stat().st_size}
pack=C/'e037-aligned-reference-candidate-v1.private.tar.gz'
if pack.exists():raise FileExistsError(pack)
with tarfile.open(pack,'w:gz',dereference=True) as a:
 for name in sorted(files):a.add(P/name,arcname='frozen/'+name,recursive=False)
with tarfile.open(pack) as a:
 seen=set()
 for member in a:
  name=member.name.removeprefix('frozen/');assert name in files and member.isfile() and name not in seen
  data=a.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==files[name]['sha256'] and len(data)==files[name]['bytes'];seen.add(name)
 assert seen==set(files)
receipt={'schema':'v2d.private-candidate-pack.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pack':pack.name,'pack_sha256':sha(pack),'pack_bytes':pack.stat().st_size,'files':files,'archive_reopened_and_every_member_hash_verified':True,'uploaded':False,'published':False,'independent_off_machine_backup':False,'physical_task_accepted':False}
(C/'e037-aligned-reference-candidate-pack-v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
plan={'schema':'v2d.bounded-quality-gpu-plan.v1','session':'e037-aligned-reference-smoke-v1','status':'PREPARED_NOT_QUEUED','episode':37,'reserved_episode':False,'instance_id':'dngrtbldz','maximum_simultaneous_instances':1,'input_pack':pack.name,'input_pack_sha256':receipt['pack_sha256'],'frozen_manifest_sha256':sha(P/'manifest.json'),'require_reference_contact_feasibility':True,'reference_contact_tolerance_m':.001,'source_motion_speed':1.0,'worker_limit_seconds':1200,'independent_provider_stop_guard_seconds':1800,'stop_guard_must_be_armed_before_start':True,'require_actual_stop_control_success_before_start':True,'backup_before_stop':True,'hash_verify_every_returned_private_file':True,'confirm_provider_stopped_after_backup':True,'paid_work_queued':False,'official_evaluator_source_changes':False,'completed_jobs_to_rerun':[],'training_before_preflight_or_controller_trial_pass':False,'training_approved_by_cpu_reference_gate_only':False,'training_chunk_if_physics_passes':{'maximum_iterations':25,'world_count':64,'rollout_steps':24,'seed':42,'maximum_environment_steps':38400,'retain_iteration_zero_and_final':True,'stop_on_nonfinite_or_regression':True},'stages':[{'name':'fresh native closure/FK/contact preflight','acceptance':'Recompile exact frozen pack on Linux. Both untouched full auditors and committed1mm gate must pass all151knots. Check geometry buffers/source targets and actual URDF limits; no Mac MJB reuse.'},{'name':'full unassisted zero-residual controller readback','entrypoint':'/opt/dex3-extension/diagnose_task.py','experiment':'dex3_scored_baseline','world_count':8,'source_frames':151,'acceptance':'No external object assist, scripted welding, resets afterstart, slowdown, nonfinite state or catastrophic loss. Record full native motor/contact/body-state readbacks; assess tracking, gravity/support and hand-object contact. Stable grasp and taskcompletion need measured evidence, not merely process completion.'},{'name':'bounded learning, only after physics review passes','entrypoint':'/opt/dex3-extension/train.py','experiment':'dex3_scored_baseline','acceptance':'One25iteration chunk and matched frozen-input iteration0/final evaluations; retain failures. Independent repeat required before promotion. No change to mass/friction/rewards/reset/gates within comparison.'}],'metric_comparison_caveat':'New geometry/reference differs from old baseline, so local AUC differences across them cannot establish leaderboard improvement. No hidden score available.','startup_variability':'Passive tracing remains separate. Do not rerun completed3case replay. Record missing startup buffers on next justified new-policy run if supported.'}
(C/'e037-aligned-reference-gpu-plan-v1.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}));print('Prepared unqueued bounded GPU specification; no cloud command issued.')

"""Record completed source work, private integrity and actual Slurm idle state once."""
from pathlib import Path
import datetime,hashlib,json,subprocess
S=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent;R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
out=W/'private-integrity-checkpoint-v2.json';assert not out.exists(),'Preserve completed checkpoint'
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
q=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','watcloud','/opt/slurm/bin/squeue -u rohith --json'],text=True,timeout=30));assert not q['jobs']
save(W/'queue-idle-after-native-RGB-v1.json',dict(observed_utc=stamp,jobs=q['jobs'],errors=q['errors'],cluster=q['meta']['slurm']['cluster'],Brev_commands_executed=False))
scheduler=json.loads((W/'native-RGB-scheduler-finish-v1.json').read_text());assert scheduler['controller_returncode']==0 and '711102|COMPLETED|0:0|' in scheduler['accounting']
backup=json.loads((W/'native-RGB-private-backup-verified-v1.json').read_text());assert backup['worker_complete'] and backup['files_verified']==6
worker=S/'e037-watcloud-native-RGB-v1/outputs';receipt=json.loads((worker/'session-receipt.json').read_text())
for name,row in receipt['files'].items():assert sha(worker/name)==row['sha256'] and (worker/name).stat().st_size==row['bytes']
rgbfit=json.loads((S/'e037-native-RGB-multiview-source-fit-v1/report.json').read_text());coverage=json.loads((S/'e037-native-RGB-skin-coverage-fit-v1/report.json').read_text());assert not rgbfit['training_approved'] and not coverage['training_approved']
ci=json.loads((W/'source-camera-geometry-ci-v1.json').read_text());assert ci['conclusion']=='success'
parent=W/'private-integrity-checkpoint-v1.json';pd=json.loads(parent.read_text());dirs=['e037-RGB-stereo-registration-v1','e037-RGB-stereo-registration-v2','e037-RGB-stereo-registration-v3','e037-RGB-stereo-registration-v4','e037-RGB-hand-correspondence-v1','e037-RGB-multiview-source-fit-v1','e037-native-RGB-hand-input-v1','e037-watcloud-native-RGB-v1','e037-native-RGB-hand-source-audit-v1','e037-native-RGB-multiview-source-fit-v1','e037-native-RGB-surface-coverage-v1','e037-native-RGB-skin-coverage-fit-v1']
files={}
for root in [S/x for x in dirs]+[W]:
 for f in sorted(root.rglob('*')):
  if not f.is_file() or f==parent or f==out:continue
  name=str(f.relative_to(S))
  if name in pd['files']:continue
  files[name]=dict(sha256=sha(f),bytes=f.stat().st_size)
save(out,dict(schema='v2d.watcloud-private-integrity-checkpoint.v2',created_utc=stamp,repo_head=head,parent_checkpoint_sha256=sha(parent),parent_checkpoint_verified_files=pd['verified_new_files'],parent_files_not_unnecessarily_rehashed=True,verified_new_files=len(files),files=files,independent_teammate_backup=False,source_flags_modified=False))
summary=dict(schema='v2d.watcloud-native-RGB-quality-checkpoint.v1',created_utc=stamp,source_commit=head,job_id='711102',Slurm_state='COMPLETED',exit_code='0:0',allocation_elapsed_s=261,GPU_shard_MiB=16384,host_RAM_GiB=16,tmpdisk_GiB=60,transfer_bytes=2701768929,private_GPU_output_files_verified=6,private_new_checkpoint_files=len(files),GPU_model_inference_s=13.93367762491107,SAM2_loaded_or_transferred=False,new_PPO_steps=0,all_prior_failed_controls_preserved=True,native_contact_gate_m=.001,RGB_only_image_screen_passed=True,complete_source_screen_passed=False,full_carry_reference_qualified=False,physical_task_accepted=False,hidden_score_claimed=False,queue_empty=True,source_fit_report=str(S/'e037-native-RGB-multiview-source-fit-v1/report.json'),coverage_failure_report=str(S/'e037-native-RGB-skin-coverage-fit-v1/report.json'),integrity_checkpoint=str(out),private_NAS_path_question_still_pending=True)
save(W/'native-RGB-quality-checkpoint-v1.json',summary)
current=dict(stage='idle_after_native_RGB_source_checks',checked_utc=stamp,queue_empty=True,completed_jobs=['711097','711098','711100','711102'],job_receipts=[str(W/x) for x in ['preflight-result-v1.json','scheduler-finish-v1.json','mask-refinement-scheduler-finish-v1.json','native-RGB-scheduler-finish-v1.json']],latest_queue_receipt=str(W/'queue-idle-after-native-RGB-v1.json'),latest_private_integrity_checkpoint=str(out),new_private_files_verified=len(files),private_NAS_path_question_pending=True,home_quota_GiB=20,home_usage_GiB_at_preflight=20.36,GPU_work_queued=False,Brev_left_stopped=True)
phase_path=S/'phase_state_20261018.json';phase=json.loads(phase_path.read_text());ledger_path=S/'compute/controller-lifecycle-runtime-ledger-v1.json';ledger=json.loads(ledger_path.read_text())
for d in [phase,ledger]:
 d.update(preferred_compute='watcloud',watcloud_current=current,active_gpu=None,active_gpu_session=None,active_cpu=None,active_cpu_only_job=None,queued_gpu=[],latest_update_utc=stamp)
phase['updated_utc']=stamp;phase['repo_head']=head
phase['latest_results']['native_RGB_source_probe_v1']=summary
phase['latest_results']['native_RGB_source_fit_v1']=dict(report=summary['source_fit_report'],RGB_heldout_median_px=7.967116594358984,RGB_heldout_p95_px=14.671711751481135,original_IR_heldout_median_px=11.9738191764744,legacy_estimated_joint_depth_error_m=.05081981827658025,heldout_skin_hits=31,heldout_skin_rays=32,heldout_skin_median_error_m=.0021953127260701533,source_qualified=False)
phase['latest_results']['native_RGB_skin_coverage_control_v1']=dict(report=summary['coverage_failure_report'],RGB_heldout_median_px=7.821627201336309,RGB_heldout_p95_px=15.083725399176354,heldout_skin_hits=31,heldout_skin_rays=32,heldout_skin_median_error_m=.0018973579624511072,source_qualified=False)
phase['verification']['latest_source_camera_geometry']=dict(focused_CPU_tests_passed=8,private_finite_camera_controls=20,maximum_private_control_error_m=2.9730017891971272e-9,actual_GPU_prediction_to_canonical_vertex_joint_closure_m=2.660558690026349e-9,CI=ci,all_source_flags_retained=True)
phase['queued_cpu']=[
 'Watcloud jobs711097/711098/711100/711102 are COMPLETED and privately SHA-backed-up; queue empty. Do not repeat eight-frame probes, SAM2 mask refinement, one-frame native RGB inference, completed source/coverage fits, old baseline cohorts, finalizers, renderers or replay jobs.',
 'Next investigate source thumb-index web silhouette and visibility/authority of the inferred landmark estimates. Diagnostic e037-native-RGB-surface-coverage-v1 locates the frozen missing heldout ray at C pixel759,234 with estimated depth.356629m, outside mesh silhouette and5.77px from nearest projected vertex. Source images visually inspected; do not relabel/delete this heldout failure or adjust thresholds.',
 'Native RGB bounded fit passes RGB-only 7.97/14.67px median/p95 but fails complete source screen: original IR median11.97px, historical estimated joint-depth discrepancy50.82mm,31/32skin rays. New training-only skin projection coverage still31/32; retain both failed controls. Occluded fingertips are not visible skin centers; constant5mm skin-to-joint depth is unproven. Investigate source-bound visibility/observation model before another candidate, not another weight sweep.',
 'Only after source qualification, retarget under original morphology/URDF/motion bounds and require committed1mm full native reference-contact plus swept/grasp checks before PPO. No physical task or policy improvement is qualified. Passive startup traces only on a justified new physical candidate, never completed replay jobs.',
 'Use Watcloud Slurm scratch or a verified private NAS path; existing NAS question pending, do not ask again. Preserve unrelated home files/quota. One bounded GPU job at a time with VRAM/tmpdisk/hard scheduler limit, private SHA-verified backup before cleanup and confirmed scheduler idle. Keep Brev stopped; do not retry Brev.'
]
phase['outstanding'][0]='Continue source-bound thumb-web/visibility investigation in queued_cpu. RGB calibration/helper and one-frame native pose probe are now completed; do not rerun them. No complete source reference is qualified.'
phase['outstanding'][1]='Watcloud operational and idle. Four jobs completed with verified backups. Generic source camera conversion has8focusedCPUtests, CI,20privateMANOcontrols and actualGPU-output closure. Mask/hand metrics remain estimates; no physical grip qualified.'
ledger['latest_quality_code_commit']=head;ledger['latest_quality_code_pushed']=True;ledger['latest_quality_ci']=ci;ledger['latest_private_integrity_checkpoint']=str(out)
ledger['latest_idle_reconciliation']=dict(checked_utc=stamp,provider='watcloud',queue_empty=True,active_GPU_job=None,active_CPU_job=None,Brev_commands_executed=False,source_images_and_weights_publicly_uploaded=False,new_PPO_steps=0,qualified_physical_tasks=0)
session=dict(job_id='711102',kind='one-frame calibrated native RGB HaMeR source probe',state='COMPLETED',exit_code='0:0',elapsed_s=261,GPU_shard_MiB=16384,private_transfer_bytes=2701768929,private_backup_verified_files=6,training_steps=0)
assert not any(x['job_id']=='711102' for x in ledger['watcloud_sessions']);ledger['watcloud_sessions'].append(session)
save(phase_path,phase);save(ledger_path,ledger);print(json.dumps(summary,indent=2))

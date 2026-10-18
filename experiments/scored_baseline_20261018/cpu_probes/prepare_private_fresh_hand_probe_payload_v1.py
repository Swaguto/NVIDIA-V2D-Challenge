"""Prepare one small private source-prior inference payload; no provider calls.

Reuse existing source crops, hand records and committed code. No video decoding,
model inference, replay export, policy finalizer or renderer is run.
"""
from pathlib import Path
import ast,hashlib,io,json,re,shutil,sys,tarfile,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/track3'))
from recover_source_hand import validate_request,canonical_sources,digest
O=S/'e037-fresh-hand-probe-payload-v1';O.mkdir();P=O/'inputs';P.mkdir();request=S/'e037-fresh-source-hand-request-v2/request.json';data=validate_request(request);sources=canonical_sources(data);B=Path(data['canonical_bundle']);context=S/'perception-overlay-fresh-hand-v1';cm=json.load(open(context/'context-manifest.json'));assert not cm['private_footage_included'] and not cm['model_weights_included'] and not cm['credentials_included'];started=time.monotonic()
files=[request,Path(data['evidence_receipt']),Path(data['evidence_arrays']),B,B.parent/'manifest.json',*[Path(row['image']) for row in data['frames']],*[p for p,r in sources.values()]];files=list(dict.fromkeys(files));total=sum(p.stat().st_size for p in files);assert total<128*1024**2 and shutil.disk_usage(S).free-total>int(1.5*1024**3)
manifest={}
for source in files:
 relative=source.relative_to(S);target=P/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);assert digest(target)==digest(source);manifest[str(relative)]=dict(bytes=target.stat().st_size,sha256=digest(target))
for rel,h in cm['files'].items():assert digest(context/rel)==h
# Relocate only file paths in a separate CPU rehearsal request. Contents,
# source clock, subject shape, source flags and all pixel coordinates stay fixed.
relocated=json.loads(request.read_text())
for field in ['evidence_receipt','evidence_arrays','canonical_bundle']:relocated[field]=str(P/Path(data[field]).relative_to(S))
for row in relocated['frames']:row['image']=str(P/Path(row['image']).relative_to(S))
rp=O/'relocated-cpu-validation-request.json';rp.write_text(json.dumps(relocated,indent=2)+'\n');relocated_data=validate_request(rp);relocated_sources=canonical_sources(relocated_data);assert len(relocated_sources)==len(sources)
# All helper-read data paths resolve into the payload, not the original source.
assert all(Path(row['image']).is_relative_to(P) for row in relocated['frames'])
assert all(Path(relocated[field]).is_relative_to(P) for field in ['evidence_receipt','evidence_arrays','canonical_bundle'])
assert all(p.is_relative_to(P) for p,r in relocated_sources.values())
payload=dict(schema='v2d.private-source-hand-probe-payload.v1',files=manifest,original_read_only_container_input_namespace=str(S),request_relative_path=str(request.relative_to(S)),request_sha256=digest(request),source_images=8,canonical_hand_records=len(sources),full_source_videos_included=False,model_weights_included=False,credentials_included=False,complete_episode_replay_pack=False,footage_is_private=True,reference_accepted=False,training_approved=False,source_commit=cm['source_commit'])
(O/'payload-manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
archive=O/'source-inputs.private.tar.gz'
with archive.open('xb') as stream:
 with tarfile.open(fileobj=stream,mode='w:gz') as tar:
  for name in sorted(manifest):tar.add(P/name,arcname='inputs/'+name,recursive=False)
  tar.add(O/'payload-manifest.json',arcname='payload-manifest.json',recursive=False)
with tarfile.open(archive) as tar:
 expected={'inputs/'+name:entry['sha256'] for name,entry in manifest.items()};expected['payload-manifest.json']=digest(O/'payload-manifest.json');assert {m.name for m in tar}==set(expected)
 for member in tar:
  assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts;assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==expected[member.name]
code_archive=O/'committed-perception-code.tar.gz'
with code_archive.open('xb') as stream:
 with tarfile.open(fileobj=stream,mode='w:gz') as tar:
  for name in sorted(cm['files']):tar.add(context/name,arcname=name,recursive=False)
  tar.add(context/'context-manifest.json',arcname='context-manifest.json',recursive=False)
image='ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:6e2039d0a54e14a188a1798efe57c3465d002f063c782c827b791e4418d2dce6'
args=['docker','run','--rm','--pull=never','--network','none','--gpus','all','--shm-size','8g','--name','v2d-e037-fresh-source-hand','-v','{REMOTE_INPUTS}:'+str(S)+':ro','-v','{REMOTE_CODE_CONTEXT}/code:/workspace/pr32:ro','-v','{PRIVATE_HAMER_WEIGHTS}:/workspace/hamer-weights:ro','-v','{PRIVATE_SAM2_WEIGHT_FILE}:/workspace/pr32-models/sam2.1_hiera_large.pt:ro','-v','{REMOTE_OUTPUTS}:/outputs','-e','OMP_NUM_THREADS=4','-e','OPENBLAS_NUM_THREADS=4','--entrypoint','/workspace/hand-runtime/bin/python',image,'/workspace/pr32/scripts/track3/recover_source_hand.py','--request',str(request),'--out','/outputs/recovery','--sam-weights','/workspace/pr32-models/sam2.1_hiera_large.pt','--hamer-weights','/workspace/hamer-weights']
gate=R/'scripts/track3/reference_contact_gate.py'
plan=dict(schema='v2d.prepared-source-hand-probe-plan.v1',stage='fresh_source_prior_inference_only',status='BLOCKED_OWNER_CONFIRMED_PROVIDER_OUTAGE',queued=False,do_not_launch=True,existing_instance_only='dngrtbldz',maximum_gpu_vms=1,worker_wall_limit_s=900,independent_stop_guard_s=1200,requires_verified_working_provider_stop_before_start=True,requires_independent_stop_guard_before_start=True,requires_hash_verified_private_backup_before_stop=True,requires_confirmed_provider_STOPPED_after_run=True,no_policy_training_or_physics_in_this_probe=True,source_commit=cm['source_commit'],image=image,code_context_manifest_sha256=digest(context/'context-manifest.json'),private_input_archive_sha256=digest(archive),private_code_archive_sha256=digest(code_archive),weights_are_existing_private_read_only_mounts=True,placeholder_container_argv=args,source_pose_results_accepted=False,reference_gate_before_any_PPO=dict(code=str(gate.relative_to(R)),sha256=digest(gate),tolerance_m=.001,passed=False,other_gates=['source_fit_nonregression','original_missing_observation_authority','original_robot_morphology_bounds','actual_URDF_limits','motion_limits','swept_native_contacts','grasp_proximity']),provider_commands_executed=False)
(O/'blocked-gpu-probe-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
for source in files:assert digest(source)==manifest[str(source.relative_to(S))]['sha256']
receipt=dict(schema='v2d.private-source-hand-probe-preparation.v1',producer_sha256=digest(Path(__file__)),private_inputs=len(manifest),input_bytes=total,input_archive_bytes=archive.stat().st_size,input_archive_sha256=digest(archive),committed_code_files=len(cm['files']),committed_code_source=cm['source_commit'],code_archive_bytes=code_archive.stat().st_size,code_archive_sha256=digest(code_archive),independent_relocated_CPU_source_validation=True,relocated_canonical_records_verified=len(relocated_sources),relocated_request_sha256=digest(rp),actual_GPU_execution=False,actual_container_execution=False,all_source_files_unchanged=True,all_input_tar_members_hash_verified=True,footage_private=True,new_paid_sessions=0,new_training_steps=0,queued=False,wall_seconds=time.monotonic()-started)
(O/'preparation-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

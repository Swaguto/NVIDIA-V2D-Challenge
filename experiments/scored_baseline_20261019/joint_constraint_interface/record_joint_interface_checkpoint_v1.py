"""Bookkeeping of NEW interface, retained failures, exact receipt timestamps."""
from pathlib import Path
import json,hashlib,datetime,shutil
S=Path('/Users/rohithiruku/v2d_scored_baseline_20261017');C=S/'compute';R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
provider_path=C/'provider-status-after-joint-interface-v1.json';provider=json.loads(provider_path.read_text())
assert provider['current_provider_status_verified'] and provider['CLI_exit_code']==0
before={}
for p in [S/'phase_state_20261018.json',C/'controller-lifecycle-runtime-ledger-v1.json']:
 j=json.loads(p.read_text());before[str(p)]=dict(top=j.get('latest_provider_receipt'),nested=j.get('brev_current'))
 assert not j.get('active_cpu_only_job') and not j.get('active_cpu_work')
 b=j['brev_current'];b.update(provider_status=provider['provider_status'],checked_utc=provider['checked_utc'],
 latest_provider_receipt=str(provider_path),last_confirmed_provider_receipt=str(provider_path),latest_provider_attempt=str(provider_path),
 current_provider_status_verified=True,last_confirmed_provider_status=provider['provider_status'],last_confirmed_observed_utc=provider['checked_utc'],
 status_attempt_started_utc=provider['attempt_started_utc'],status_attempt_finished_utc=provider['attempt_finished_utc'],latest_provider_error=None,
 future_start_stop_reliability_NOT_established=True,health_and_SSH_observations_NOT_refreshed_this_phase=True)
 j.update(updated_utc=now,latest_provider_receipt=str(provider_path),latest_provider_attempt=str(provider_path),
 current_provider_status_verified=True,provider_status=provider['provider_status'],provider_observed_utc=provider['checked_utc'])
 write(p,j)
write(C/'joint-interface-provider-nested-reconciliation-v1.json',dict(updated_utc=now,before=before,
 actual_latest_receipt=str(provider_path),actual_observed_utc=provider['checked_utc'],
 latest_successful_receipt_kept_its_older_fallback_metadata_UNCHANGED=True,
 explanation='Top-level prior receipt advanced while nested brev_current fallback remained older. Both authoritative state locations now bind the actual latest successful receipt. Raw historical receipts are unchanged.',
 no_new_provider_attempt_or_VM_mutation=True,no_auth_or_future_lifecycle_cause_inferred=True))
archive=R/'experiments/scored_baseline_20261019/joint_constraint_interface';archive.mkdir()
code=['joint_constraint_interface_v1.py','verify_joint_interface_contracts_v1.py','build_joint_executable_interface_v1.py',
 'joint_constraint_derivatives_v1.py','verify_joint_directional_contracts_v1.py','verify_joint_executable_interface_v1.py',
 'load_joint_executable_runtime_v1.py','joint_lossless_returns_v1.py','verify_joint_lossless_returns_v1.py',
 'record_joint_interface_checkpoint_v1.py','joint-interface-native-kinematics-public-reference-v1.json',
 'joint-interface-provider-nested-reconciliation-v1.json']
plans=['joint-interface-contracts','joint-executable-interface','joint-directional-contracts','joint-executable-binding','joint-runtime-loader','joint-lossless-return-contracts']
code += [n+'-CPU-plan-v1.json' for n in plans]
for n in code:shutil.copy2(C/n,archive/n)
write(archive/'code-manifest.json',dict(schema='v2d.NEW-joint-executable-interface-code-only.v1',
 files={n:sha(archive/n) for n in code},production_runtime_tests_UNCHANGED=True))
readme='''# Executable joint constraint interface — 2026-10-20

Code, plans and provenance metadata only. Geometry, arrays, matrices, source, numerical returns and all episode inputs remain private. Production runtime and tests are unchanged.

This implements the saved joint construction without selecting command or direction values. The namespace remains 1,523 coordinates; all 151 clocks, 8,114 robot linear rows, 194 original URDF source factors, 38 half-angle domains/full region2/unknown bands, 4,551,571 unchanged object rows, 900 SOC4 and 151 SOC5 cones remain. A new executable sparse copy preserves all 31,839,917 retained A coefficients, RHS/P/q and conditioning; all 32,306,054 original coefficient positions receive retained/invalid routing. All 8,798 active robot terms preserve original exact rational provenance. Numeric sparse actions are construction operators, not original-gate certificates.

Articulated native body/support/normal functions consume the saved compiled geometry and native tree, with the original object stage/Cayley order and free support qpos0. URDF source geometry remains distinct from native metadata. Each pair has a free body-A-local unit direction, transformed with the current body. Every compiled vertex and support tie is retained. Joint AND direction slopes remain correlated by tied vertex; the full directional derivative takes the maximum of their combined slopes. The mean tie selection is SEARCH only. Zero cone-tail directional derivatives retain the whole tail norm. No selected face, axis, pose, rank or failing-pair bank is introduced.

Whole-object actual motion uses the unchanged six-cap function on new world poses, actual timestamps, relative principal-Log angular velocity, midpoint acceleration and actual p95. Original source disks and separate strict depth stay original; numerical half-angle screens retain unknowns. Fixed palm/immutable thumb/middle/changed-root late repair and whole-source gates still apply after construction. The interface does not replace actual integer cells, training NN/ranks/rounded median/quality/full-triangle raster, fullpixel/heldout POST or native+self/swept/motor/grasp gates.

Six serial CPU guards: five successes, one retained failure. Nineteen new controls pass: 12 primitive executable controls, 3 full directional/zero-cone/sign controls, 1 full-namespace actual action/adjoint API control, and 3 typed return controls. Independent bindings cover every new sparse coefficient, RHS/objective/cone/conditioning entry and all 194 original source factor routes. Preflight sparse storage is 418,702,899 bytes. Peak observed preflight RSS 1,109,835,776 bytes stays below 2GiB; no resource limit was increased.

The loader constructor reaches its metadata writer, which fails on a tuple mapping key before a constructor receipt is saved. Its raw in-memory object state and unsaved receipt remain UNAVAILABLE; the loader/writer is not rerun and its state is not reconstructed. A NEW typed serializer is checked only on NEW synthetic mappings, IEEE bit patterns (including signed zero/nonfinite payloads), and exact sparse pointer/index storage with explicit zeros. Future genuinely NEW method constructors must seal full raw arrays and typed keys before assertions/further queries. No old missing return is filled in.

Three actual full-namespace API matmul RuntimeWarnings (divide/overflow/invalid) are retained. Its actual saved scalar returns are finite and the local adjoint check passes, but the cause and future numerical reliability remain unproven. No environment/cache/dependency change or warning suppression repairs this.

The native analytical order follows the [pinned MuJoCo 3.8.1 kinematics source](https://raw.githubusercontent.com/google-deepmind/mujoco/3.8.1/src/engine/engine_core_smooth.c), SHA-256 3206f73347fc60f15c94f923d2f81288e25831c3439bef752ef313fa26b748eb. Source/license are retained privately. This is not installed-binary FK equivalence or a replay of episode/native FK.

No episode input, pose/source/NN/raster/FK/native position/distance, optimizer/physics/GPU/PPO evaluation occurs. All command coordinates and direction witnesses remain unassigned. F and separate unqualified components, source uncertainty and all original gates remain unchanged. No trajectory/reference/task/policy/AUC qualifies. The owner qualification objective remains unachieved.
'''
(archive/'README.md').write_text(readme)
append='''\n\n## 2026-10-20 — executable joint constraint/support derivative interface\n\nCode-only archive: `experiments/scored_baseline_20261019/joint_constraint_interface`; private checkpoint/integrity: `compute/joint-interface-checkpoint-v1.json` / `compute/private-integrity-joint-interface-v1.json`. Production runtime/tests unchanged.\n\nThe new executable interface keeps all 1523 unassigned coordinates, 151 clocks, 8114 robot rows/194 original URDF source factors/38 domains/fullregion2/unknown bands, 4551571 retained object rows and all cones. Independent bindings verify all 31839917 retained A coefficients and every original coefficient routing, RHS/P/q/conditioning/source route. All compiled support vertices, moving body/direction normals and correlated ties remain; mean SEARCH slopes are not universal directional derivatives. Actual six-motion/angular-p95 and all original training/fullpixel-heldout POST/native+self/swept/motor/grasp gates remain.\n\nSix serial guards complete5success/1failure;19 new controls pass. A tuple-key metadata writer fails after loader invocation before a saved constructor receipt; raw object state/receipt remain UNAVAILABLE, never rerun/reconstructed. New typed return controls cover only new synthetic mappings/IEEE bits/sparse pointers and explicit zeros. Three API matmul warnings are retained with finite saved returns; cause/future reliability unproven. Nested Brev fallback bookkeeping is reconciled to the latest real receipt without changing raw receipts or observation timestamps.\n\nNo optimizer, episode pose/source/NN/raster/FK/native/physics/GPU/PPO evaluation occurs. Coordinates/directions remain unassigned; no physical/reference/trajectory/task/policy/AUC qualification. F, separate components and all original failures/gates remain unchanged. Owner intent remains active/unachieved; research completion is not goal completion.\n'''
for p in [R/'docs/team/dex3-isaac-review/WRIST_RETARGET_CONTACT_20261019.md',R/'docs/team/dex3-isaac-review/SOURCE_OBJECT_POSE_DEPTH_20261019.md',R/'scripts/dex3_training/SELF_CONTACTS.md',R/'scripts/dex3_training/SOURCE_RAY_TEMPORAL.md']:
 with p.open('a') as f:f.write(append)
parent=C/'joint-collision-checkpoint-v1.json';parentpi=C/'private-integrity-joint-collision-v1.json'
old=json.loads(parentpi.read_text());files=dict(old['files'])
for rel,rec in files.items():assert sha(S/rel)==rec['sha256'],rel
jobs=['e037-joint-interface-contracts-v1','e037-joint-executable-interface-v1','e037-joint-directional-contracts-v1','e037-joint-executable-binding-v1','e037-joint-runtime-loader-v1','e037-joint-lossless-return-contracts-v1']
newpaths=[C/n for n in code]+[provider_path,C/'joint-interface-mujoco-3.8.1-engine_core_smooth.c.txt']
for name in jobs:newpaths += [p for p in (S/name).rglob('*') if p.is_file()]
for p in newpaths:files[str(p.relative_to(S))]=dict(sha256=sha(p),bytes=p.stat().st_size)
pi=dict(schema='v2d.NEW-joint-interface-private-integrity.v1',created_utc=now,
 parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),parent_integrity_receipt=str(parentpi),parent_integrity_receipt_sha256=sha(parentpi),
 parent10996files_verified=True,verified_files=len(files),files=files,
 code_archive_manifest_sha256=sha(archive/'code-manifest.json'),
 private_source_geometry_matrices_arrays_results_uploaded=False,old_completed_producers_controls_solvers_source_native_audits_NOT_replayed=True,new_GPU_jobs=0,new_PPO_steps=0)
write(C/'private-integrity-joint-interface-v1.json',pi)
guards={n:json.loads((S/n/'guard.json').read_text()) for n in jobs}
cp=dict(schema='v2d.NEW-joint-interface-checkpoint.v1',created_utc=now,parent_checkpoint=str(parent),private_integrity_checkpoint=str(C/'private-integrity-joint-interface-v1.json'),
 private_files_verified=len(files),CPU_jobs=len(jobs),process_successes=sum(g['worker_exit_code']==0 for g in guards.values()),process_failures=sum(g['worker_exit_code']!=0 for g in guards.values()),completed_CPU_guards=guards,
 archive=dict(path=str(archive),exact_code_plan_provenance_files=len(code),manifest_sha256=sha(archive/'code-manifest.json')),
 NEW_contracts=19,production_runtime_tests_UNCHANGED=True,
 executable_preflight=json.loads((S/'e037-joint-executable-interface-v1/report.json').read_text()),
 complete_independent_binding=json.loads((S/'e037-joint-executable-binding-v1/report.json').read_text()),
 retained_loader_scope=json.loads((S/'e037-joint-lossless-return-contracts-v1/NEW-failed-loader-boundary-scope-RETURN.json').read_text()),
 actual_API_warnings=3,warning_cause_and_future_reliability_UNPROVEN=True,
 provider_receipt=str(provider_path),provider_status=provider['provider_status'],provider_observed_utc=provider['checked_utc'],
 all_episode_inputs_commands_directions_UNASSIGNED=True,no_episode_pose_source_NN_raster_FK_native_solver_physics_GPU_PPO_evaluation=True,
 F_and_separate_components_UNCHANGED_and_UNQUALIFIED=True,all_original_gates_and_failures_preserved=True,
 owner_goal_achieved=False,owner_continuation_intent='active/unachieved',historical_goal_tool_status='blocked',goal_tool_observation_utc='2026-10-20T15:44:42.193354+00:00',research_checkpoint_NOT_goal_completion=True)
write(C/'joint-interface-checkpoint-v1.json',cp)
print(json.dumps(dict(code_files=len(code),private_files=len(files),CPU_jobs=len(jobs),successes=cp['process_successes'],failures=cp['process_failures'])))

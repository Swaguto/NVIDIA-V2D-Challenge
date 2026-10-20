"""Bookkeeping only: private receipts, code-only archive, preserved owner goal."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
S=Path('/Users/rohithiruku/v2d_scored_baseline_20261017');C=S/'compute';R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,j):Path(p).write_text(json.dumps(j,indent=2)+'\n')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
archive=R/'experiments/scored_baseline_20261019/joint_robot_object_collision'
archive.mkdir()
code_names=['joint_collision_maps_v1.py','verify_joint_collision_contracts_v1.py','build_joint_collision_maps_v1.py','verify_joint_collision_maps_v1.py','verify_joint_collision_unreached_bindings_v1.py','joint_collision_predicates_v1.py','build_joint_collision_predicates_v1.py','build_joint_native_object_gauge_v1.py','verify_joint_complete_predicates_v1.py','verify_joint_one_load_bindings_v1.py','record_joint_collision_checkpoint_v1.py','build_joint_collision_maps_v1.preparation-unmatched-parenthesis.raw.py.txt','joint-complete-binding-IO-interruption-v1.json','joint-collision-native-filter-public-reference-v1.json']
plans=['joint-collision-contracts','joint-collision-maps','joint-collision-binding','joint-collision-unreached-binding','joint-collision-predicates','joint-native-object-gauge','joint-complete-binding','joint-one-load-binding']
code_names += [p+'-CPU-plan-v1.json' for p in plans]
for n in code_names:shutil.copy2(C/n,archive/n)
manifest={n:sha(archive/n) for n in code_names}
write(archive/'code-manifest.json',dict(schema='v2d.NEW-joint-collision-code-only.v1',files=manifest,production_runtime_tests_UNCHANGED=True))
text='''# Joint robot/object collision construction — 2026-10-20

Code and plans only. Private geometry, arrays, matrices, poses, source data and worker returns remain outside Git. Production runtime and tests are unchanged.

The new symbolic model combines 465 saved continuous right index/thumb coordinates with 1,058 saved object Cayley/lift/phase coordinates. All 1,523 coordinates and continuous support directions remain unassigned. The same saved wrist and middle components remain unqualified hypotheses; F remains the sole verified starting reference. Original repairs and missing commands are immutable indirections.

All 31 native bodies, 24 joints and 58 compiled geometries are represented. The maps retain all 1,653 unordered geometry pairs at each of 151 clocks (249,603 instances), including every right-self pair. Conditional built-in static filters leave typed tautology records; no geometric broadphase or selected failing-pair bank removes coverage. ALL compiled body vertices define a continuous unit-sphere Minkowski-support predicate. This is an ideal convex-geometry construction, not numerical native EPA/contact equivalence or a physical qualification.

Moving-body and moving-normal chain rules retain all ancestors and support ties. Compiled BODY vertices already include geom placement and are transformed once. The native object rotation composes on the right of the saved native object rotation; translations use the saved world-to-native stage map. Sigma remains a construction lift, and the original free support qpos0 is preserved. Cached F hand world transforms are not used for moving digits/composed roots.

The 8,114 robot linear rows, 194 source factors and 38 domains/full region2/unknown bands remain original by hash. All unchanged object source/clock/motion rows and 900 SOC4/151 SOC5 cones remain. The 66,591 old fixed-hand geometric rows are invalidated and replaced by the new complete articulated predicate family. All 32,306,054 old coefficient positions receive explicit retained/invalid routing; the original P/q is embedded without a new robot cost or scaling. Actual source, integer-cell, strict-depth, changing NN/rank, rounded median, quality, six motion, fullpixel/heldout POST and native+self/swept/motor/grasp gates remain binding.

Eight serial CPU guards: six successes and two failures/interruption. Eighteen new concrete controls pass (12 chain/support/return +4 filter/predicate/embedding +1 native object frame +1 one-load resource control). No optimizer, episode pose, source/NN/raster/native position/distance/physics/GPU/PPO evaluation occurred.

The first independent metadata check fails an attempted native/URDF numeric identity. Saved native origins differ by up to 3.4689903311013737e-9 m and ranges by 1.3116372499233364e-7 rad; cause unproven. Both original maps remain unchanged and distinct, with both required for actual acceptance. No tolerance increase or metadata repair was applied. Earlier successful tree bindings are reused by hash, then only unreached route/support bindings execute.

A preparation SyntaxError is retained before worker/module/guard execution. A later binding implementation was interrupted after 45.9404 s when repeated compressed NPZ field loading was detected; peak observed RSS 927,481,856 bytes. Its unfinished coefficient return remains unavailable. A new one-load binding interface has its own control and evaluates only previously unfinished bindings. It independently binds every new routing coefficient/pointer, 8,798 active robot terms, all pair predicates and 2,425 object-gauge components. It does not reconstruct the interrupted internal state.

The conditional static filter follows the [pinned MuJoCo 3.8.1 collision driver](https://raw.githubusercontent.com/google-deepmind/mujoco/3.8.1/src/engine/engine_collision_driver.c), SHA-256 41cce70938fa3bd789419308f64febdd0f1d4639e08984d7984ebffcbc48f11c. Public source/license are retained privately. No callback, sleep, dynamic broadphase or installed binary equivalence is inferred from the tag. Actual native filters and all-pair/self gates remain mandatory.

No component/reference/trajectory/task/policy/AUC qualifies. F retains its eight right-self and five hand/object failures. All historical failures/unknowns remain; no completed solver, model, source or native cohort was replayed. The owner qualification goal is unachieved; research completion is not goal completion.
'''
(archive/'README.md').write_text(text)
append='''\n\n## 2026-10-20 — joint robot/object symbolic collision construction\n\nCode-only archive: `experiments/scored_baseline_20261019/joint_robot_object_collision`. Private checkpoint: `compute/joint-collision-checkpoint-v1.json`; integrity chain: `compute/private-integrity-joint-collision-v1.json`. Production runtime/tests unchanged.\n\nThe 465 saved robot-angle variables and 1058 saved object coordinates now share an unassigned 1523-coordinate namespace. All 31 native bodies/24 joints/58 compiled geometries and all 249603 pair-clock instances are represented, including all right-self pairs. New articulated body/support/normal ancestry and native-object gauge maps replace invalid fixed-hand geometry; all original robot/source/object clock-motion-cone restrictions remain by hash. Continuous all-vertex unit-sphere support predicates are construction geometry, not numerical native contact, force or task certificates.\n\nEight serial guards finish with six successes and two retained failures/interruption. Eighteen new controls pass. Native/URDF metadata identity fails (origins up to 3.469e-9m; ranges up to 1.312e-7rad); both maps remain separate and unchanged, cause unproven, no tolerance repair. A redundant compressed-I/O verifier is stopped before its fixed resource limit; unfinished return stays unavailable. New one-load bindings verify every new coefficient routing and support/native-frame incidence without replaying source, FK/native, model or solver cohorts. A preparation SyntaxError is retained separately.\n\nNo optimizer, episode pose/source/NN/raster/native position/distance/physics/GPU/PPO evaluation occurred. Wrist/middle components remain unqualified hypotheses; all F repairs/missing commands remain fixed. Original 8 right-self/5 hand-object failures persist. Whole source/URDF/motion, actual training/fullpixel-heldout POST, native+self/swept/motor/grasp gates still precede training. Owner intent remains active and unachieved; historical blocked goal-tool observation stays separate.\n'''
for p in [R/'docs/team/dex3-isaac-review/WRIST_RETARGET_CONTACT_20261019.md',R/'docs/team/dex3-isaac-review/SOURCE_OBJECT_POSE_DEPTH_20261019.md',R/'scripts/dex3_training/SOURCE_RAY_TEMPORAL.md',R/'scripts/dex3_training/SELF_CONTACTS.md']:
 with p.open('a') as f:f.write(append)
# Retain and independently re-verify the full parent private chain.
parent=C/'exact-terminal-dual-checkpoint-v1.json';parentpi=C/'private-integrity-exact-terminal-dual-v1.json'
pi0=json.loads(parentpi.read_text());files=dict(pi0['files'])
for rel,rec in files.items():assert sha(S/rel)==rec['sha256'],rel
job_names=['e037-joint-collision-contracts-v1','e037-joint-collision-maps-v1','e037-joint-collision-binding-v1','e037-joint-collision-binding-continuation-v1','e037-joint-collision-predicates-v1','e037-joint-native-object-gauge-v1','e037-joint-complete-binding-v1','e037-joint-one-load-binding-v1']
newpaths=[C/n for n in code_names]+[C/'joint-collision-mujoco-3.8.1-engine_collision_driver.c.txt',C/'provider-status-after-joint-collision-v1.json']
for n in job_names:newpaths += [p for p in (S/n).rglob('*') if p.is_file()]
for p in newpaths:files[str(p.relative_to(S))]=dict(sha256=sha(p),bytes=p.stat().st_size)
pi=dict(schema='v2d.NEW-joint-collision-private-integrity.v1',created_utc=now,
 parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),
 parent_integrity_receipt=str(parentpi),parent_integrity_receipt_sha256=sha(parentpi),
 parent10892files_verified=True,verified_files=len(files),files=files,
 code_archive_manifest_sha256=sha(archive/'code-manifest.json'),
 private_source_geometry_arrays_matrices_results_uploaded=False,
 old_completed_producers_controls_solvers_source_native_audits_NOT_replayed=True,
 new_GPU_jobs=0,new_PPO_steps=0)
write(C/'private-integrity-joint-collision-v1.json',pi)
guards={n:json.loads((S/n/'guard.json').read_text()) for n in job_names}
provider=json.loads((C/'provider-status-after-joint-collision-v1.json').read_text())
cp=dict(schema='v2d.NEW-joint-collision-checkpoint.v1',created_utc=now,parent_checkpoint=str(parent),
 private_integrity_checkpoint=str(C/'private-integrity-joint-collision-v1.json'),private_files_verified=len(files),
 CPU_jobs=len(guards),process_successes=sum(g['worker_exit_code']==0 for g in guards.values()),
 process_failures_or_interruptions=sum(g['worker_exit_code']!=0 for g in guards.values()),completed_CPU_guards=guards,
 archive=dict(path=str(archive),exact_code_plan_preparation_metadata_files=len(code_names),manifest_sha256=sha(archive/'code-manifest.json')),
 NEW_controls=18,production_runtime_tests_UNCHANGED=True,
 symbolic_maps=json.loads((O if False else S/'e037-joint-collision-maps-v1').joinpath('report.json').read_text()),
 complete_predicate_maps=json.loads((S/'e037-joint-collision-predicates-v1/report.json').read_text()),
 independent_complete_bindings=json.loads((S/'e037-joint-one-load-binding-v1/report.json').read_text()),
 distinct_native_URDF_scope=json.loads((S/'e037-joint-collision-binding-continuation-v1/NEW-native-URDF-distinct-map-scope-RETURN.json').read_text()),
 interrupted_binding_RETURN='UNAVAILABLE, no internal state reconstructed; new one-load interface binds unfinished obligations',
 provider_receipt=str(C/'provider-status-after-joint-collision-v1.json'),provider_status=provider['provider_status'],provider_observed_utc=provider['checked_utc'],
 no_episode_pose_source_NN_raster_FK_native_solver_physics_GPU_PPO_evaluation=True,
 all_commands_and_directions_UNASSIGNED=True,
 F_unchanged=True,separate_components_UNQUALIFIED=True,
 all_original_gates_and_failures_preserved=True,
 owner_goal_achieved=False,owner_continuation_intent='active/unachieved',
 historical_goal_tool_status='blocked',goal_tool_observation_utc='2026-10-20T15:44:42.193354+00:00',
 research_checkpoint_NOT_goal_completion=True)
write(C/'joint-collision-checkpoint-v1.json',cp)
print(json.dumps(dict(archive=str(archive),code_files=len(code_names),private_files=len(files),guards=len(guards),successes=cp['process_successes'],failures=cp['process_failures_or_interruptions'])))

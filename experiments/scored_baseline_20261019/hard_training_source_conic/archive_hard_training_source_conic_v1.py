"""Code-only archive of completed bounded work; no scientific reexecution."""
from pathlib import Path
import datetime,hashlib,json,shutil
S=Path(__file__).resolve().parents[1];C=S/'compute';R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');A=R/'experiments/scored_baseline_20261019/hard_training_source_conic'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
jobs=[('e037-training-proxy-roundoff-v1','training-proxy-roundoff-CPU-plan-v1.json'),('e037-hard-training-source-model-v1','hard-training-source-model-CPU-plan-v1.json'),('e037-hard-training-source-binding-v1','hard-training-source-binding-CPU-plan-v1.json'),('e037-hard-training-callback-contract-v1','hard-training-callback-CPU-plan-v1.json'),('e037-hard-training-source-evaluation-v1','hard-training-source-evaluation-CPU-plan-v1.json'),('e037-hard-training-source-evaluation-v2','hard-training-source-evaluation-CPU-plan-v2.json'),('e037-hard-training-source-return-audit-v1','hard-training-source-return-audit-CPU-plan-v1.json')]
guards={};last=None
for folder,plan in jobs:
 g=read(S/folder/'guard.json');p=read(C/plan);assert g['worker_sha256']==sha(p['worker'])==p['worker_sha256'];assert g['plan_sha256']==sha(C/plan)
 for f,h in p['dependencies_sha256'].items():assert sha(f)==h,f
 assert g['wall_seconds']<240 and g['maximum_observed_resident_bytes']<2147483648
 if last:assert last<g['started_utc']
 last=g['completed_utc'];guards[folder]=g
assert sum(g['worker_exit_code']==0 for g in guards.values())==6
assert guards['e037-hard-training-source-evaluation-v1']['worker_exit_code']==1
names=['training_proxy_roundoff_v1.py','derive_training_proxy_roundoff_v1.py','training-proxy-roundoff-CPU-plan-v1.json','build_hard_training_source_conic_v1.py','hard-training-source-model-CPU-plan-v1.json','verify_hard_training_source_model_v1.py','hard-training-source-binding-CPU-plan-v1.json','hard_training_conic_returns_v1.py','verify_hard_training_callback_v1.py','hard-training-callback-CPU-plan-v1.json','evaluate_hard_training_source_conic_v1.py','hard-training-source-evaluation-CPU-plan-v1.json','evaluate_hard_training_source_conic_v2.py','hard-training-source-evaluation-CPU-plan-v2.json','audit_hard_training_source_return_v1.py','hard-training-source-return-audit-CPU-plan-v1.json','thumb_conic_geometry_v1.py','guard_right_self_contact_job_v1.py','hard-training-source-syntax-only-repair-v1.json','hard-training-source-callback-api-inventory-v1.json','read_provider_status_safe_v3.py','archive_hard_training_source_conic_v1.py']
assert not A.exists();A.mkdir(parents=True);files=[]
for n in names:
 dest=n+'.txt'if n=='evaluate_hard_training_source_conic_v1.py'else n;shutil.copyfile(C/n,A/dest);assert sha(C/n)==sha(A/dest);files.append(dict(file=dest,restore_relative_path='compute/'+n,executed_or_declared_source_sha256=sha(C/n),expected_syntax_failure=n=='evaluate_hard_training_source_conic_v1.py'))
(A/'source-manifest.json').write_text(json.dumps(dict(schema='v2d.exact-code-only-research-archive.v1',files=files,private_inputs_geometry_arrays_results_NOT_included=True),indent=2)+'\n')
note='''## Hard training-source conic construction, 2026-10-20

Seven NEW serial guarded CPU processes complete: six succeed and one fails at
Python parsing BEFORE any module execution/constructor/solver. Twenty-two exact
source/plan/helper/metadata files are code-only under `hard_training_source_conic`.
The failed source is retained verbatim as.py.txt, with its private failed log,
guard and plan. V2 inserts one missing whitespace and changes only its output
directory; the syntax-only diff is retained. There is ONE completed episode
solver call, with no restart/warmstart/tolerance/weight/profile/budget sweep.
Production runtime/tests and all original source/physical authority stay unchanged.

Five NEW binary64 contracts derive uniform source-translation/point-local and
outward-proposal bounds. Independent exact operation-by-operation recurrences
cover all six three-term dot orders on all151frames. Maximum distance allowance
is9.984e-15m, covering the inherited124nominal F proxy defects without modifying
old radii/arrays or candidate tolerances. These OUTWARD proposal guards do NOT
certify KD-tree/raster source operators, source accuracy or actual acceptance.
All124defects and original1e-10source/1e-9object-motion comparisons remain.

The NEW canonical model has454columns/1,166,056rows/286,907SOC4cones and18,428
nonnegative rows. It combines ALL286,608saved sufficient median proxies, ALL299
original max-speed/acceleration maps and17,520selected geometry directions from
the saved INITIAL F union return. All453world translations are continuous in a
declared20mm algorithmic box; one bounded clearance slack is optimized. Frobot
commands/object rotations remain fixed. No old rejected point is a warmstart,
and no fullpixel/heldout/old60failure value enters fitting or framewise tuning.
The phase objective is rho+.5*1e-3*sum(delta^2), an algorithmic choice, not physics.
Original actual p95 and all source/native/self/swept/motor/grasp gates still rule.

Independent complete binding verifies all151new source blocks/3,439,296copied
coefficient cells,17,520geometry rows,299motion cones,908box rows and objective.
One NEW tiny analytic callback control passes, saving all six actual info returns;
actual cone dimensions and callback API are checked without dependency installs.
All old source/NN/control/packing/model audits are reused BY HASH, not replayed.

ONE100iteration/60sec/one-thread/default-tolerance solve returns Solved in30
iterations/17.088963sec, with31durable info callbacks. Raw x/z/s and constructor
returns are saved before further work, never clipped/repaired/normalized. The
complete guarded evaluation takes26.120sec/peakRSS853,426,176bytes. Solver success
does NOT qualify a point. The ONEraw proposal fails training silhouette IoU at
19clocks and training quality at18 (overlap), while training median has0failures.
All151actual hard-training decisions are retained. ALLsix ORIGINAL full-clock
object motion checks pass, at the original1e-9 comparison. The point is REJECTED.
No fullpixel/heldout source rescue, native-position/physics/GPU/PPO or new source-
motion component/reference follows. These19training failures are NOT comparable
to the old60fullpixel failures and do not establish a physical improvement.

Four NEW analytic dual-defect/dot-bound contracts precede the independent saved-
return audit. It binds2,652,701canonical coefficients,31callbacks,151training
decisions, all six saved motion decisions and raw transform/unchanged robot and
rotation bytes, without source/NN/FK/native/solver replay. All286,907new rational
root certificates pass; raw dual nonnegative/SOC failures are0/0. A finite-box,
residual/dual-defect-corrected certificate gives rho>=.005355725241031122m for
THIS frozen numerical median-proxy/selected-direction/20mm-box construction.
Thus zero clearance slack is excluded only for that restricted numeric model.
This is NOT native penetration, a source/time-family/task exclusion, physical
force/importance or a required wrist change. Raw rho=.005355750741394552m and
all rejected returns/statuses remain unchanged. Numerical primal residuals are
retained separately; they are not original candidate acceptance tolerances.

NEXT derive a genuinely NEW training-source feasibility representation that can
handle changing median-rank/nearest-vertex subsets AND discrete silhouette
coverage, using original source definitions and saved training authority. Do not
swap a prefix/axis or reweight/restart this completed conic model to repair its
return. First derive meaningful new pointwise/cardinality/raster/unknown/error
contracts and independent maps; no new solver, pose or physical job is queued.
No old data/controls/budgets/audits or fullpixel/heldout failures may be replayed
or used for fitting/selection. A sufficient construction exclusion does not
exclude original source-valid paths. Object changes alone cannot fix8right-self
failures. F/separate components/missing136/heldout31of32/parameters/boxes/centers/
caps/tolerances/gates stay unchanged; actual whole trajectory remains unqualified.
'''
(A/'README.md').write_text('# Hard training-source conic construction\n\n'+note.rstrip()+'\n')
for n in ['docs/team/dex3-isaac-review/SOURCE_OBJECT_POSE_DEPTH_20261019.md','docs/team/dex3-isaac-review/WRIST_RETARGET_CONTACT_20261019.md','scripts/dex3_training/SOURCE_RAY_TEMPORAL.md']:
 p=R/n;p.write_text(p.read_text().rstrip()+'\n\n'+note.rstrip()+'\n')
p=C/'hard-training-source-archive-preparation-v1.json';assert not p.exists();p.write_text(json.dumps(dict(schema='v2d.hard-training-source-archive-preparation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=str(A),exact_sources_plans_helpers=len(names),manifest_sha256=sha(A/'source-manifest.json'),guards=guards,all7jobs_serial=True,process_successes=6,process_failures=1,worker_plan_dependencies_sha_verified=True,NEWcontracts=10,scientific_outcome='REJECTED_training_source; restricted numeric zero-clearance-slack excluded ONLY'),indent=2)+'\n');p.chmod(0o600)
print(json.dumps(dict(archive=str(A),exact_files=len(names),jobs=7,successes=6,failures=1,NEWcontracts=10,trajectory_qualified=False)))

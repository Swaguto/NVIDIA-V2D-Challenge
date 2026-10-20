"""Retain completed order-statistic map work, without rerunning it."""
from pathlib import Path
import hashlib,json,shutil,datetime
S=Path(__file__).resolve().parents[1];C=S/'compute'
R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
A=R/'experiments/scored_baseline_20261019/training_orderstat_support'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
names=['training_orderstat_support_v1.py','derive_training_orderstat_support_v1.py',
 'training-orderstat-support-CPU-plan-v1.json','guard_right_self_contact_job_v1.py',
 'thumb_conic_geometry_v1.py','archive_training_orderstat_support_v1.py']
guard=json.loads((S/'e037-training-orderstat-support-v1/guard.json').read_text())
plan=json.loads((C/names[2]).read_text());assert guard['worker_exit_code']==0
assert guard['worker_sha256']==sha(C/names[1])==plan['worker_sha256']
assert guard['plan_sha256']==sha(C/names[2])
for f,h in plan['dependencies_sha256'].items():assert sha(f)==h,f
assert not A.exists();A.mkdir(parents=True)
items=[]
for name in names:
 shutil.copyfile(C/name,A/name);assert sha(C/name)==sha(A/name)
 items.append(dict(file=name,restore_relative_path='compute/'+name,executed_or_declared_source_sha256=sha(C/name)))
(A/'source-manifest.json').write_text(json.dumps(dict(schema='v2d.exact-code-only-research-archive.v1',files=items,private_inputs_arrays_geometry_results_NOT_included=True),indent=2)+'\n')
note='''## Sufficient training-median order-statistic support, 2026-10-20

ONE further serial guarded CPU worker succeeds with seven NEW analytic contracts
and ONE new saved-data map inventory. Six exact source/plan/helper files are
code-only under `training_orderstat_support`; all numerical/source/geometry
records stay private. No old training/source/nearest-neighbor/control audit is
rerun. No solver or actual nonzero episode pose is selected/evaluated. Production
runtime/tests/F/source/parameters/original gates remain unchanged.

For n training distances, nested sets of floor((n-1)/2)+1 and floor(n/2)+1
training points suffice to bound the two median ranks. Distance to ANY fixed
source visual vertex upper-bounds actual nearest-vertex distance. Thus fixed
proxy-vertex norm bounds on those sets can ensure a training median upper bound
even when actual nearest vertices/ranks change. Proxy choices are conservative
construction witnesses, not observed correspondences, mandatory source poses or
necessary source-family restrictions. Original fullpixel/heldout gates and the
actual hard training raster/median/quality oracle remain separate and mandatory.

Seven NEW contracts cover even/odd nested-rank cardinalities, nearest-vertex
switches under a valid proxy bound, SOC4 world-map/adjoint signs, exact zero-radius
semantics, all-frame hard-filter completeness and a continuous interpolation
counterexample for p95 feasibility. Original actual p95 must be checked on each
trial; interpolation of feasible endpoints does not ensure p95 feasibility.
Raster IoU remains discrete, with no invented classical gradient.

ONE fixed151clock inventory proposes286,608SOC4maps/1,146,432canonical rows on
453UNASSIGNED world-translation columns. It reuses the saved training points,
nearest-vertex returns and original transform domains BY HASH. New independent
cardinality/row-sign/exact affine-map-error bindings are durable. The original
source statistic is not recomputed and no candidate value is assigned.

Exact squared-norm comparison retains124nominal F proxy-radius defects, maximum
10524878282246375/1329227995784915872903807060280344576m^2
(about7.92e-21m^2). These result from distinct saved numerical distance/point
representations; they are not clipped, silently repaired or used to change any
original candidate tolerance. Initial F feasibility for this exact proxy model
is NOT established. No arithmetic/packing allowance or radius repair is selected.
Any future allowance needs a justified numerical-operator binding and explicit
scope before a solve, while actual source acceptance remains unchanged.

NEXT design one bounded hard-source-filtered geometric/motion construction using
these saved proxy maps only if its numerical semantics are justified. Preserve
all124defects and retain the distinction between this sufficient restriction,
the nominal training oracle and original fullpixel source gates. First establish
new packing/arithmetic/actual-acceptance contracts and independent complete-model
bindings; no solver/pose evaluation is yet selected or queued. Never replay these
seven controls/maps or the completed19training controls/data/BVH/oracle cache.
No old rejected point/60postchecks/heldout values may enter fitting, frame weights
or parameter selection. No old budget continuation/restart/tolerance repair.
Current Brev remains UNVERIFIED after the saved read-only timeout, last actual
STOPPED03:42:30.871525UTC; no VM mutation or paid work. All source/missing136/
heldout31of32/F/separate components/gates remain unchanged. Object changes alone
cannot clear the8right-self failures. No source/native/self/swept/motor/grasp/
trajectory/policy/AUC qualification follows; the owner objective is unachieved.
'''
(A/'README.md').write_text('# Training-median order-statistic support\n\n'+note.rstrip()+'\n')
for name in ['docs/team/dex3-isaac-review/SOURCE_OBJECT_POSE_DEPTH_20261019.md',
 'docs/team/dex3-isaac-review/WRIST_RETARGET_CONTACT_20261019.md','scripts/dex3_training/SOURCE_RAY_TEMPORAL.md']:
 p=R/name;p.write_text(p.read_text().rstrip()+'\n\n'+note.rstrip()+'\n')
p=C/'training-orderstat-archive-preparation-v1.json';assert not p.exists()
p.write_text(json.dumps(dict(schema='v2d.training-orderstat-archive-preparation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=str(A),exact_sources_plans_helpers=len(names),archive_manifest_sha256=sha(A/'source-manifest.json'),guard=guard,worker_plan_dependency_hashes_verified=True,no_old_producer_controls_audit_or_solver_replayed=True),indent=2)+'\n');p.chmod(0o600)
print(json.dumps(dict(archive=str(A),exact_files=len(names),guarded_workers=1,contracts=7,no_solver_or_candidate=True)))

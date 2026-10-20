"""Code-only retention of the completed bounded source-filter method."""
from pathlib import Path
import datetime,hashlib,json,shutil
S=Path(__file__).resolve().parents[1];C=S/'compute';R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');A=R/'experiments/scored_baseline_20261019/hard_source_filtered_motion'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
jobs=[('e037-hard-source-filter-contracts-v1','hard-source-filter-contracts-CPU-plan-v1.json'),('e037-hard-source-filter-preflight-v1','hard-source-filter-preflight-CPU-plan-v1.json'),('e037-hard-source-filter-preflight-binding-v1','hard-source-filter-binding-CPU-plan-v1.json'),('e037-hard-source-filter-evaluation-v1','hard-source-filter-evaluation-CPU-plan-v1.json'),('e037-hard-source-filter-return-audit-v1','hard-source-filter-return-audit-CPU-plan-v1.json')]
guards={};last=None
for folder,pname in jobs:
 g=read(S/folder/'guard.json');p=read(C/pname)
 assert g['worker_exit_code']==0 and g['worker_sha256']==p['worker_sha256']==sha(p['worker'])and g['plan_sha256']==sha(C/pname)
 for name,digest in p['dependencies_sha256'].items():assert sha(name)==digest
 assert g['wall_seconds']<240 and g['maximum_observed_resident_bytes']<2147483648
 if last:assert last<g['started_utc']
 last=g['completed_utc'];guards[folder]=g
names=['hard_source_filter_direction_v1.py','hard_source_filter_binding_v1.py','derive_hard_source_filter_contracts_v1.py','hard-source-filter-contracts-CPU-plan-v1.json','hard_source_filter_inputs_v1.py','build_hard_source_filter_preflight_v1.py','hard-source-filter-preflight-CPU-plan-v1.json','verify_hard_source_filter_preflight_v1.py','hard-source-filter-binding-CPU-plan-v1.json','evaluate_hard_source_filter_v1.py','hard-source-filter-evaluation-CPU-plan-v1.json','audit_hard_source_filter_return_v1.py','hard-source-filter-return-audit-CPU-plan-v1.json','archive_hard_source_filter_v1.py']
assert not A.exists();A.mkdir(parents=True);files=[]
for name in names:
 shutil.copyfile(C/name,A/name);assert sha(C/name)==sha(A/name)
 files.append(dict(file=name,restore_relative_path='compute/'+name,executed_or_declared_source_sha256=sha(C/name)))
(A/'source-manifest.json').write_text(json.dumps(dict(schema='v2d.exact-code-only-research-archive.v1',files=files,private_inputs_geometry_arrays_results_NOT_included=True),indent=2)+'\n')
note='''## Bounded actual-source-filtered object construction, 2026-10-20

Five NEW serial guarded CPU workers succeed; fourteen exact code/plan/helper
files are code-only under `hard_source_filtered_motion`. Six NEW complete-
construction contracts pass, including two NEW analytical tiny QPs. They bind
the coupled step/clearance phase, source search plane, whole-SOC conditioning,
raw-step versus line-trial identities and sealed proposal/nonzero-cache scopes.
One actual synthetic raster/median example improves the median while worsening
IoU; the actual hard filter rejects the solved QP direction. It does not invent
a classical IoU derivative. Old controls/data are reused BY HASH, not rerun.
Production runtime/tests, original source/physics authority and gates stay fixed.

The ONE declared method starts only from F zero object translations, with F
robot commands/object rotations fixed. Median chosen-gradient rows are SEARCH
predictors, including branches at ties; they are not source certificates or
measured correspondences. Actual training NN/ranks/full-triangle silhouette/
quality are trial-filtered. All six ORIGINAL object-motion gates/actual p95 and
timestamps are mandatory. The model has454columns/19,926rows:18,730nonnegative
rows and299SOC4cones, including17,520initial geometry search branches,151median
predictor and151strict-depth predictor rows. Positive row/whole-cone scaling is
conditioning only, not source confidence, a score weight or a candidate tolerance.
Independent complete preflight binding passes all raw/conditioned rows/objective,
151source-stat/gradient/depth copies,17,520cached initial branches and299motion
cones. Later models would require independent binding before EACH constructor.
The first model and zero-F source/geometry are reused BY HASH, never re-queried.

The declared limits are8major steps/8trials/3fixed internal line fractions
1,.5,.25/160seconds, one thread; subproblems use100iterations/10seconds/default
tolerances. A20mm world-translation box,1mm step box, bounded phase slack and
rho+.5*1e-3*||step||^2 are algorithmic choices, not original object/robot boxes,
source caps, physical parameters or gate changes. Raw QP x/z/s are never
clipped/normalized/repaired. A line fraction creates a separately sealed NEW
trial. Source returns must match that proposal identity, every clock and the
correct zero-cache/nonzero actual-query scope. Actual source and motion pass,
algorithmic box and strict union-clearance-proxy descent are all required to
accept a method step. No best/intermediate or failed-point selection is allowed.
No fullpixel/heldout/old60failure value enters fitting, weights or selection.

The SOLE bounded episode returns
UNRESOLVED_no_hard_source_motion_feasible_descent after ONE Solved QP,
20actual info callbacks and3NEW trial vectors;0steps are accepted. Method wall
is21.280263seconds, complete guard23.119333seconds/peakRSS603,422,720bytes.
All raw constructors/matrices/QP x/z/s/callbacks/proposals/poses/union-axis values/
source and motion returns are durable before further work. Every one of the
three trials passes ALL6ORIGINAL151clock object-motion gates at original1e-9,
and reduces the numerical clearance proxy, but fails the actual training source
filter at78/60/37frames. Those are NEW TRAINING failures, not the old60FULLPIXEL
failures, and they are not a physical improvement or an accepted reference.
The method stops as declared, without restart/tuning/extra line fractions.

Terminal world translations and state are EXACTLY initial F. Every nonzero
trial remains REJECTED; no source-motion component/reference, original fullpixel/
heldout rescue, native-position/physics/GPU/PPO/task/policy/AUC result follows.
The numerical proxy phases (.005360372 initial; .004423318/.004891845/.005126108
for the rejected trials) are model values, not native penetration or forces.
No unaccepted point replaces F, and no source confidence/missing label/geometry/
stiction/effort/physical parameter/root box/motion tolerance/gate is modified.

Independent SAVED-return verification binds the raw QP and20callbacks, all3
sealed trial/raw-step/pose/motion/acceptance identities,14,496,552NEW returned
axis values and453actual returned-mask/distance/source decisions. Scalar axis
closures are<=5.552e-17m within explicit arithmetic guards, not native gates.
Raw returned masks provide exact I/U counts, and raw returned distance arrays
provide the actual rank/median/quality decision maps; no raster/KD/source/FK/
native/method/model/solver is replayed. All6motion decisions per trial and the
zero-accepted-step terminal identity independently verify. All failures/raw
returns stay unchanged. This rejects only this finite method's trials; it does
not exclude the translation family/task, establish stationarity or prove that
rotation/a new wrist profile is necessary. Existing8right-self/5object failures
and all F/separate component/source/heldout/missing/authority limitations remain.

NEXT investigate a genuinely NEW full object-pose construction with continuous
object rotation, not a smaller step/reweight/restart of this completed method.
FIRST reuse original object authority/6caps/clock/source receipts BY HASH and
derive only NEW changed-object-rotation source/motion/compiled-geometry maps and
meaningful contracts. Consider all3object rotation components at every151clock
about ORIGINAL F object poses; no selected axis/amplitude/time-profile/pose bank
or borrowed robot8mm/.08object bound. Object rotations would be derived PRIVATE
reference variables, never new measured observations or altered human/source
arrays. Robot wrist mode/centers/commands/repairs and all missing commands remain
fixed as authorized. Original angular speed/acceleration/p95 must be evaluated
for changed rotations; old fixed-rotation invariance is NOT inherited.

Rotating object geometry invalidates the old translation-only affine planes,
fixed axes and49071uniform-box omission certificates. Bind proper compiled
MuJoCo3.8.1object/body/geom/support-qpos0 transforms, full eligible pair coverage
and any NEW conservative broadphase before a future complete model or solve.
Do not regenerate old FK/native/source/ray/cohort audits or reconstruct unknown
returns; reuse cached input evidence BY HASH within its scope. Preserve source
uncertainty, ORIGINAL fullpixel/heldout POST-only verification, all151clocks,
all6object caps and original object1e-9/source1e-10 comparisons. No new solve/
pose/physics/GPU is queued. Source/native/motion/conditioning/actual-acceptance
contracts and independent NEW maps must precede ONE declared bounded future
evaluation. No axis/anchor/partition/cut/pathgraph/optimizer/degree/budget/profile/
tolerance sweep, old failed-point warmstart or physical-score source-truth choice.
This possible extension is not proved necessary or feasible by the rejection.
Any actual new reference still needs whole151source/actualURDF/all object/root/
finger motion, FULL1mm native hand/object/support AND self, full swept, actual
motor/grasp and whole-source qualification before GPU/PPO. The owner objective
remains unachieved; a completed research/method checkpoint never completes it.
'''
(A/'README.md').write_text('# Bounded actual-source-filtered motion\n\n'+note)
for name in ['docs/team/dex3-isaac-review/SOURCE_OBJECT_POSE_DEPTH_20261019.md','docs/team/dex3-isaac-review/WRIST_RETARGET_CONTACT_20261019.md','scripts/dex3_training/SOURCE_RAY_TEMPORAL.md']:
 p=R/name;p.write_text(p.read_text().rstrip()+'\n\n'+note)
p=C/'hard-source-filter-archive-preparation-v1.json';assert not p.exists();p.write_text(json.dumps(dict(schema='v2d.hard-source-filter-archive-preparation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=str(A),exact_sources_plans_helpers=len(files),manifest_sha256=sha(A/'source-manifest.json'),guards=guards,all5jobs_serial=True,process_successes=5,process_failures=0,NEWcontracts=6,scientific_status='UNRESOLVED; all3trials REJECTED actual training source; terminal F unchanged'),indent=2)+'\n');p.chmod(0o600)
print(json.dumps(dict(archive=str(A),exact_files=len(files),CPUjobs=5,NEWcontracts=6,accepted_steps=0,trajectory_qualified=False)))

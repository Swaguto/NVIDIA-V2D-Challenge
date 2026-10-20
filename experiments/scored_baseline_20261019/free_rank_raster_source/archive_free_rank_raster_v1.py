"""Code-only retention of completed new source logic; no scientific reexecution."""
from pathlib import Path
import datetime,hashlib,json,shutil
S=Path(__file__).resolve().parents[1];C=S/'compute';R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
A=R/'experiments/scored_baseline_20261019/free_rank_raster_source'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
jobs=[('e037-exact-training-source-contracts-v1','exact-training-source-contracts-CPU-plan-v1.json'),
 ('e037-free-rank-raster-representation-v1','free-rank-raster-representation-CPU-plan-v1.json'),
 ('e037-free-rank-raster-independent-v1','free-rank-raster-independent-CPU-plan-v1.json')]
guards={};last=None
for folder,pname in jobs:
 g=read(S/folder/'guard.json');p=read(C/pname)
 assert g['worker_exit_code']==0 and g['worker_sha256']==p['worker_sha256']==sha(p['worker'])
 assert g['plan_sha256']==sha(C/pname)
 for name,digest in p['dependencies_sha256'].items():assert sha(name)==digest
 assert g['wall_seconds']<240 and g['maximum_observed_resident_bytes']<2147483648
 if last:assert last<g['started_utc']
 last=g['completed_utc'];guards[folder]=g
assert not A.exists();A.mkdir(parents=True)
names=['exact_training_source_logic_v1.py','derive_exact_training_source_contracts_v1.py',
 'exact-training-source-contracts-CPU-plan-v1.json','build_free_rank_raster_representation_v1.py',
 'free-rank-raster-representation-CPU-plan-v1.json','verify_free_rank_raster_representation_v1.py',
 'free-rank-raster-independent-CPU-plan-v1.json','archive_free_rank_raster_v1.py']
files=[]
for n in names:
 shutil.copyfile(C/n,A/n);assert sha(C/n)==sha(A/n)
 files.append(dict(file=n,restore_relative_path='compute/'+n,executed_or_declared_source_sha256=sha(C/n)))
ref=S/'e037-exact-training-source-reference-v1/reference-receipt.json'
shutil.copyfile(ref,A/'public-reference-receipt.json')
files.append(dict(file='public-reference-receipt.json',public_source_provenance_ONLY=True,executed_or_declared_source_sha256=sha(ref)))
(A/'source-manifest.json').write_text(json.dumps(dict(schema='v2d.exact-code-only-research-archive.v1',files=files,
 private_source_geometry_pixels_arrays_and_results_NOT_included=True),indent=2)+'\n')
note='''## Free-rank, nearest-union and discrete training raster, 2026-10-20

Three NEW serial guarded CPU workers succeed. Nine exact sources/plans/public
reference metadata files are code-only under `free_rank_raster_source`. Private
source arrays/results remain private. All18NEW analytic contracts pass their
stated scopes; no completed old control/producer/audit/solver is replayed.
Production runtime/tests and all original gates/parameters remain unchanged.

The NEW median representation leaves thresholds, rank subsets and nearest
vertices FREE. For n=2k ideal distances, count(d<=a)>=k and count(d<=b)>=k+1,
0<=a<=b and a+b<=2*m0 are equivalent to median<=m0. Odd n uses one threshold.
General forward/reverse proofs, tie/zero/missing behavior and a changing-nearest
vertex union are retained. These are algebraic proof controls, not measured
source poses, fixed human anchors or robot commands. All original visual6022
vertices are alternatives; no fixed nearest/prefix map is substituted.

Numerical semantics remain explicit. The original binary64 two-rank average
can differ from the rational average at round-to-even and subnormal boundaries;
NEW controls retain both examples. The numerical branch uses the ORIGINAL
rounded average on numerical distances. Exact Euclidean ball unions bind the
ideal saved-binary-input model, not an unproved KD-tree/norm implementation or
upstream camera/source truth. Missing/uncertain events retain separate certain
and possible cardinalities; they are never dropped or set false.

The NEW raster branch program follows original clip-then-truncate, LINE_8,
shift0, full visual-triangle OR and EVEN-domain I/U. It includes integer line
ties, double-intersection clipping, degenerate triangles and inclusive scan
spans. Fifteen initial contracts include fixed NEW synthetic triangles/lines;
no episode baseline/rejected scene is rendered. Public OpenCV4.11.0 source
hashes and the installed4.11.0binary hash are retained. Matching a version tag
and synthetic cases is NOT a proof of full installed binary equivalence.
Source-informed code retains the original Intel license and attribution.
See official source: https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/drawing.cpp

All integer projected-vertex cells remain FREE in[-20000,20000], with correct
positive/negative/zero/saturation preimages under strict positive depth. No
classical IoU derivative, fixed face/visibility branch or front-surface/depth
buffer surrogate is invented. Crossed quantizer boundaries remain UNRESOLVED
unless original numerical execution or a justified outward bound resolves them.
Certain/possible full-triangle coverage yields exact outward integer-count
determinants; empty unions and missing/parallel/strict-depth cases stay explicit.
Fullpixel/heldout source gates remain POSTconstruction only.

ONE factored151clock formulation uses453continuous UNASSIGNED world translations
and260FREE median thresholds. It represents572,956saved training points and
3,450,341,032point/vertex distance predicates (6,015,628,724threshold instances),
1,818,644FREE integer projection coordinates,1,812,000triangle programs and
77,312,000EVEN pixel expressions without expanding billions of clauses or
assigning any value. Original fixed F rotations are a declared scope, not new
source truth. No new20mm box/radius/source cap/tolerance or solver is selected.
All six original object caps/actual timestamps and tracker/full-source gates
remain mandatory; p95 is not replaced by a convex maximum-speed envelope.

Three further NEW inverse-cell/shared-variable-polynomial/unknown-cardinality
contracts precede independent binding of4,983NEW exact factor coefficients,
all151rank/domain/incidence records,302input point/partition hashes and151
ordered durable BEFORE/AFTER frame pairs. The compiler and its independent
matrix embedding agree exactly. Old training statistics,6022vertex geometry,
domains, motion maps and old19+7+10contracts are reused BY HASH, not rerun.
The new representation is FORMAL IDEAL equivalence with explicit numerical
operator boundary obligations. No numeric episode pose, source/NN/raster
evaluation, solver, native position, physics, GPU/PPO or trajectory qualifies.

NEXT derive only NEW numerical-operator/error and actual-acceptance contracts
needed to evaluate this free-rank/full-raster construction, or retain unknowns
where no justified bound exists. Keep the actual original numerical operator
separate from rational Euclidean geometry; a proposal allowance is not a source
gate. Before any ONE bounded new evaluation, declare the complete construction,
independently bind all NEW maps/returns and make every numerical/unknown boundary
decision durable. No solver/pose method is currently selected or queued. Do not
swap/tune/restart the completed fixed-prefix conic model, repair its rejected
point or use any old fullpixel/heldout failure as fitting feedback. No clause,
vertex/rank/axis bank, finite pose search, partition/cut loop or budget sweep is
authorized by this symbolic representation.

F/separate unqualified components/136missing robot anchors and commands/31of32
heldout failures/model/source authority/parameters/original caps and tolerances
remain unchanged. Object changes alone cannot clear8right-self failures. Any
actual new reference requires fresh WHOLE151 source/URDF/all object/root/finger
motion, FULL1mm native hand/object/support AND self, full swept, actual motor/
grasp and whole-source qualification before GPU/PPO. The owner trajectory goal
remains unachieved; this research checkpoint does not complete it.
'''
(A/'README.md').write_text('# Free-rank and raster training-source representation\n\n'+note)
for n in ['docs/team/dex3-isaac-review/SOURCE_OBJECT_POSE_DEPTH_20261019.md',
 'docs/team/dex3-isaac-review/WRIST_RETARGET_CONTACT_20261019.md','scripts/dex3_training/SOURCE_RAY_TEMPORAL.md']:
 p=R/n;p.write_text(p.read_text().rstrip()+'\n\n'+note)
p=C/'free-rank-raster-archive-preparation-v1.json';assert not p.exists()
p.write_text(json.dumps(dict(schema='v2d.free-rank-raster-archive-preparation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 archive=str(A),exact_sources_plans_metadata=len(files),manifest_sha256=sha(A/'source-manifest.json'),guards=guards,
 all3jobs_serial=True,process_successes=3,process_failures=0,NEWcontracts=18,
 scientific_status='formal source-event representation; numerical boundaries unresolved; no pose or trajectory'),indent=2)+'\n');p.chmod(0o600)
print(json.dumps(dict(archive=str(A),exact_files=len(files),jobs=3,NEWcontracts=18,trajectory_qualified=False)))

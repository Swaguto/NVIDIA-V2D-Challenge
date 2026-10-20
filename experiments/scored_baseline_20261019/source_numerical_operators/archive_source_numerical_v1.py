"""Code-only retention of NEW operator arithmetic; no scientific reexecution."""
from pathlib import Path
import datetime,hashlib,json,shutil
S=Path(__file__).resolve().parents[1];C=S/'compute';R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
A=R/'experiments/scored_baseline_20261019/source_numerical_operators'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
jobs=[('e037-source-numerical-contracts-v1','source-numerical-contracts-CPU-plan-v1.json'),('e037-source-numerical-bindings-v1','source-numerical-bindings-CPU-plan-v1.json'),('e037-source-numerical-independent-v1','source-numerical-independent-CPU-plan-v1.json')]
guards={};last=None
for folder,pname in jobs:
 g=read(S/folder/'guard.json');p=read(C/pname)
 assert g['worker_exit_code']==0 and g['worker_sha256']==p['worker_sha256']==sha(p['worker'])and g['plan_sha256']==sha(C/pname)
 for name,digest in p['dependencies_sha256'].items():assert sha(name)==digest
 if last:assert last<g['started_utc']
 last=g['completed_utc'];guards[folder]=g
names=['source_numerical_enclosures_v1.py','derive_source_numerical_contracts_v1.py','source-numerical-contracts-CPU-plan-v1.json','build_source_numerical_bindings_v1.py','source-numerical-bindings-CPU-plan-v1.json','verify_source_numerical_bindings_v1.py','source-numerical-independent-CPU-plan-v1.json','archive_source_numerical_v1.py']
assert not A.exists();A.mkdir(parents=True);files=[]
for name in names:
 shutil.copyfile(C/name,A/name);assert sha(C/name)==sha(A/name)
 files.append(dict(file=name,restore_relative_path='compute/'+name,executed_or_declared_source_sha256=sha(C/name)))
p=S/'e037-source-numerical-reference-v1/reference-receipt.json';shutil.copyfile(p,A/'public-reference-receipt.json');files.append(dict(file='public-reference-receipt.json',public_source_provenance_ONLY=True,executed_or_declared_source_sha256=sha(p)))
(A/'source-manifest.json').write_text(json.dumps(dict(schema='v2d.exact-code-only-research-archive.v1',files=files,private_inputs_geometry_arrays_results_NOT_included=True),indent=2)+'\n')
note='''## Conditional numerical source operators, 2026-10-20

Three NEW serial guarded CPU workers succeed; nine exact source/plan/public
reference metadata files are code-only under `source_numerical_operators`.
Ten NEW analytic contracts pass. Production runtime/tests and all original
source/physical authority remain unchanged. No episode pose, source/NN/raster,
solver, native position, physics, GPU/PPO or trajectory qualification occurs.

NEW exact rational interval recurrences cover the original multiply/divide/add
projection graph under stated finite round-to-nearest binary64 assumptions.
Strict near-plane/singular/overflow/missing cases stay UNRESOLVED; a quantizer
cell crossed by an outward interval is not assigned a convenient integer. NEW
selected-vertex leaf-norm bounds retain zero and underflow behavior, signed
coordinates and a fixed128bit outward root certificate grid. The grid is not
a candidate tolerance. No returned norm/radius/weight is clipped or repaired.
A separate control shows a selected-vertex norm check can pass even when that
vertex is not globally nearest. Leaf arithmetic does NOT prove KD search.

Public SciPy1.16.3 source and the installed binary are SHA-pinned privately,
without installation/build/global environment changes. Matching a source tag
or synthetic returns is not a compiler/libm/KD-search binary equivalence proof.
The source norm accumulates squared differences before a sqrt; tree pruning
has its own floating operations and remains outside the leaf enclosure claim.
Official source: https://github.com/scipy/scipy/blob/v1.16.3/scipy/spatial/ckdtree/src/query.cxx

NEW operator-scope bindings record a distinction already present in the saved
code: original fullpixel observed_fit calls query(k=1 default) and NumPy matmul;
the declared EVEN-training statistic calls k=2 for first/second-neighbor and
tie diagnostics and uses einsum. Both use the conceptual nearest-vertex median,
but exact numeric k1/k2/search/matmul/einsum identity is NOT asserted. All actual
old sources/returns remain unchanged. The prior free-rank rational identities
and rounded two-rank controls retain their scopes; they do not replace the
original fullpixel scorer or its original1e-10 comparison.

The NEW cache alias is C0-(V*R^T+t) in exact saved-binary-input arithmetic.
All2,727,966signed source-visual vertex components are retained losslessly as
integer numerators/common dyadic denominators. All are exactly nonzero; maximum
absolute difference is1324166867436149043/
20769187434139310514121985316880384m, approximately6.375632e-17m. This is a
representation difference, not camera/source accuracy, physical improvement,
data repair or a changed tolerance. Cached C0 and the ideal affine formula
remain distinct. No old source projection/NN/raster/FK/native audit is replayed.

ONE151clock constructor derives1,812NEW nonnegative conditional camera error
coefficients: e_i(H)=alpha_i0+sum_j alpha_ij H_j. All H halfwidth symbols remain
UNASSIGNED; no20mm/8mm or other object box is introduced. The bound includes
the exact cache alias, inherited BY-HASH dot-order error constants and the new
cache-add rounding. Finite-RN operator assumptions remain explicit. Independent
per-operand dyadic exponent arithmetic verifies every2,727,966component, and
independent affine-polynomial propagation verifies all1,812coefficients exactly.
Both complete numerator returns,2distinct operator bindings and151ordered
BEFORE/constructor/AFTER pairs are durable/private. Old source data/controls/
roundoff proofs/free-rank maps/motion authority are reused BY HASH, not rerun.

The NEW acceptance protocol seals actual NEW proposals/constructor/solver and
intermediate returns before queries. F zero queries reuse saved cache. Actual
nonzero training k2/full-raster/median/quality outputs stay separate from optional
conditional enclosures. All151hard training restrictions and ALL6ORIGINAL object
motion gates/actual p95 must pass. Original fullpixel k1/matmul source gates
remain POSTconstruction, with original tolerances; no training certificate or
interval substitutes for their actual returns. A rejected fullpixel/heldout
point is not refitted or promoted. Global KD/compiler/sqrt/raster binary
equivalence and source accuracy remain unresolved outside the stated controls.
The protocol does not require claiming those global identities to run a NEW
declared candidate through the actual unmodified operators.

NEXT declare ONE genuinely NEW bounded hard-training-source-filtered full-clock
motion/geometry construction, after NEW complete-construction/conditioning and
actual-return acceptance contracts plus independent NEW matrix binding. Use
actual changed-rank/nearest returns and discrete full-triangle training source
tests as the hard filter; no invented classical IoU derivative or fixed-rank/
vertex/source-face truth. Conditional enclosures may retain unknowns and may
not falsely accept a reference or exclude the family. No method/solver/pose is
yet selected or queued by this phase. Derive from original definitions and
saved training authority, never old rejected vectors/fullpixel/heldout failures.
No fixed-prefix/axis swap, old algorithm retry/warmstart/tolerance/weight/profile/
limit sweep, finite pose/vertex-rank/axis bank/cut/partition loop or budget
continuation. F/separate components/all missing commands/source uncertainty/
parameters/gates stay unchanged. Any real changed reference needs whole151
source/URDF/all object/root/finger motion, FULL1mm native hand/object/support AND
self, full swept, actual motor/grasp and whole-source qualification before
GPU/PPO. Object changes alone cannot resolve8right-self failures. The owner
trajectory goal remains unachieved; this checkpoint does not complete it.
'''
(A/'README.md').write_text('# Conditional source numerical operators\n\n'+note)
for name in ['docs/team/dex3-isaac-review/SOURCE_OBJECT_POSE_DEPTH_20261019.md','docs/team/dex3-isaac-review/WRIST_RETARGET_CONTACT_20261019.md','scripts/dex3_training/SOURCE_RAY_TEMPORAL.md']:
 p=R/name;p.write_text(p.read_text().rstrip()+'\n\n'+note)
p=C/'source-numerical-archive-preparation-v1.json';assert not p.exists();p.write_text(json.dumps(dict(schema='v2d.source-numerical-archive-preparation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=str(A),exact_sources_plans_public_reference_metadata=len(files),manifest_sha256=sha(A/'source-manifest.json'),guards=guards,all3jobs_serial=True,process_successes=3,process_failures=0,NEWcontracts=10,scientific_status='conditional numerical bounds and exact cache aliases; no pose or trajectory qualified'),indent=2)+'\n');p.chmod(0o600)
print(json.dumps(dict(archive=str(A),exact_files=len(files),CPUjobs=3,NEWcontracts=10,trajectory_qualified=False)))

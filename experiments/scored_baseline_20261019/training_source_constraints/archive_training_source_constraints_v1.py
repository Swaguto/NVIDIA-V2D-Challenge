"""Archive completed NEW training-only source work; never execute producers."""
from pathlib import Path
import datetime, hashlib, json, os, shutil

S = Path(__file__).resolve().parents[1]
C = S / 'compute'
R = Path('/Users/rohithiruku/v2d_hand_continuity_fix')
A = R / 'experiments/scored_baseline_20261019/training_source_constraints'
now = datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_private(path, value):
    path = Path(path)
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    path.chmod(0o600)
    with path.open('rb') as stream:
        os.fsync(stream.fileno())

jobs = [
    ('e037-training-source-contract-v1', 'training-source-contract-CPU-plan-v1.json'),
    ('e037-training-source-constraints-v1', 'training-source-constraints-CPU-plan-v1.json'),
    ('e037-training-source-independent-v1', 'training-source-independent-CPU-plan-v1.json'),
    ('e037-training-constraint-oracle-contract-v1', 'training-constraint-oracle-CPU-plan-v1.json'),
]
guards = {}
previous_end = None
for folder, plan_name in jobs:
    output = S / folder
    guard = json.loads((output / 'guard.json').read_text())
    plan_path = C / plan_name
    plan = json.loads(plan_path.read_text())
    assert guard['worker_exit_code'] == 0
    assert guard['worker_sha256'] == sha(plan['worker']) == plan['worker_sha256']
    assert guard['plan_sha256'] == sha(plan_path)
    assert guard['wall_limit_seconds'] == 240
    assert guard['resident_limit_bytes'] == 2147483648
    assert guard['minimum_free_disk_bytes'] == 1610612736
    assert guard['maximum_observed_resident_bytes'] < guard['resident_limit_bytes']
    assert guard['wall_seconds'] < 240 and not guard['GPU_work']
    if previous_end:
        assert previous_end < guard['started_utc']
    previous_end = guard['completed_utc']
    for path, expected in plan['dependencies_sha256'].items():
        assert sha(path) == expected, path
    guards[folder] = guard

write_private(C / 'training-source-cache-fixture-scope-clarification-v1.json', dict(
    schema='v2d.training-source-cache-fixture-scope-clarification.v1', created_utc=now,
    executed_source_sha256=sha(C/'verify_training_constraint_oracle_v1.py'),
    actual_returns_unchanged=True, controls_NOT_rerun=True,
    synthetic_zero_cache_payload_is_opaque=True,
    synthetic_quality_payload_NOT_a_consistent_source_measurement=True,
    scope='The synthetic zero-cache fixture tests transparent reuse only. Its arbitrary quality payload is not derived from its I/U and median. No source-functional, pose or measurement claim uses it. The actual151 baseline quality identities were independently verified in the saved training-source binding job.',
    actual151_baseline_quality_binding_report=str(S/'e037-training-source-independent-v1/report.json'),
    actual151_baseline_quality_binding_report_sha256=sha(S/'e037-training-source-independent-v1/report.json')))

old = json.loads((C/'provider-status-after-object-dynamic-v1.json').read_text())
attempt_path = C/'provider-status-after-training-source-v1.json'
attempt = json.loads(attempt_path.read_text())
assert old['current_provider_status_verified'] and old['provider_status'] == 'STOPPED'
assert attempt['read_only'] and not attempt['current_provider_status_verified']
write_private(C/'provider-status-training-source-reconciliation-v1.json', dict(
    schema='v2d.provider-status-reconciliation.v1', recorded_utc=now,
    current_provider_status_verified=False, current_provider_status='UNVERIFIED',
    last_actual_provider_status=old['provider_status'],
    last_actual_observed_utc=old['checked_utc'],
    last_actual_receipt=str(C/'provider-status-after-object-dynamic-v1.json'),
    last_actual_receipt_sha256=sha(C/'provider-status-after-object-dynamic-v1.json'),
    latest_read_only_attempt=str(attempt_path), latest_attempt_sha256=sha(attempt_path),
    latest_attempt_started_utc=attempt['attempt_started_utc'],
    latest_attempt_finished_utc=attempt['attempt_finished_utc'],
    actual_attempt_error=attempt['error'],
    raw_attempt_contains_stale_fallback_timestamp=True,
    stale_fallback_does_NOT_replace_later_actual_STOPPED_observation=True,
    raw_attempt_UNALTERED=True, VM_mutations=0, paid_work_queued=False,
    future_start_stop_or_auth_reliability_NOT_established=True,
    successful_current_provider_read_required_before_any_future_GPU_work=True))

files = [
    'object_training_source_semantics_v1.py', 'verify_training_source_semantics_v1.py',
    'training-source-contract-CPU-plan-v1.json', 'build_training_source_constraints_v1.py',
    'training-source-constraints-CPU-plan-v1.json', 'source_vertex_bvh_v1.py',
    'verify_training_source_bindings_v1.py', 'training-source-independent-CPU-plan-v1.json',
    'object_training_constraint_oracle_v1.py', 'verify_training_constraint_oracle_v1.py',
    'training-constraint-oracle-CPU-plan-v1.json', 'thumb_conic_geometry_v1.py',
    'guard_right_self_contact_job_v1.py', 'archive_training_source_constraints_v1.py',
]
assert not A.exists()
A.mkdir(parents=True)
manifest = []
for name in files:
    origin, target = C/name, A/name
    shutil.copyfile(origin, target)
    assert sha(origin) == sha(target)
    manifest.append(dict(file=name, restore_relative_path='compute/'+name,
                         executed_or_declared_source_sha256=sha(origin)))
(A/'source-manifest.json').write_text(json.dumps(dict(
    schema='v2d.exact-code-only-research-archive.v1', files=manifest,
    private_inputs_arrays_geometry_results_NOT_included=True,
    no_completed_producer_contract_or_audit_reexecution=True), indent=2)+'\n')

note = '''## Training-only source constraints, 2026-10-20

Four NEW serial guarded CPU processes succeed. Fourteen exact source/plan/helper
files are archived code-only under `training_source_constraints`; all source
pixels, geometry, arrays, numerical returns and certificates remain private.
Production runtime/tests are unchanged. No actual nonzero episode pose or solver
is selected/evaluated, and no trajectory/source accuracy/physical score qualifies.

The NEW source construction uses ORIGINAL functional definitions on a declared
EVEN training domain: eligibility precedes deterministic raster-order sampling
(cap4096), full source-visual triangle rasterization preserves ORIGINAL clip then
integer truncation, silhouette counts use only training pixels, and the median
is distance to the SOURCE VISUAL MESH VERTICES with original even-rank averaging.
These source vertices are the original source operator; they are not a substitute
for compiled MuJoCo support vertices in native collision checks. Quality uses
IoU*exp(-median/.02). The NEW training statistics/nominal hard nonregression
predicate are construction restrictions, not original full-pixel gate changes.
Original source comparison1e-10 and object motion comparison1e-9 stay explicit.
The full-pixel/heldout gates remain postconstruction verification. The rejected
old point, its60postcheck failures and heldout values are not construction inputs.

Nineteen NEW controls pass:12functional,4independent BVH,3cache/provenance/unknown.
ONE151clock training dataset has572,956depth points and no insufficient-support
or zero-central-distance unknown frame. Nearest-vertex/rank ties in42frames
remain explicit. Current nearest vertices/ranks may change; none becomes a
measured pose/mandatory anchor. Positive-depth/projection/zero-distance/empty
support cases fail or remain UNKNOWN explicitly. No missing observation is filled.

Independent bindings verify1,818,644quantizer coordinates,302intersection/union
counts,572,956point lifts and nearest pairs using an independent source-vertex
BVH (39,343,817leaf point/vertex pairs). Maximum projection/point/nearest/gradient
identity errors are2.274e-13px/2.843e-14px/4.441e-16m/2.221e-16. These are numerical
identity diagnostics, not source-accuracy proofs or candidate tolerance changes.

A NEW hard-source oracle retains real raster discontinuities (no invented IoU
gradient) and local median derivatives with explicit rank/NN uncertainty. Its151
actual F queries reuse the saved training baseline BY HASH, without re-rendering
or rescoring it; no actual nonzero episode query ran. One synthetic zero-cache
payload is arbitrary and not a consistent source measurement: its scope is cache
transparency only. Private clarification preserves executed source/returns and
does not rerun controls. Actual151quality bindings are independently verified.

The latest read-only provider attempt times out, so CURRENT Brev state is
UNVERIFIED. Last actual STOPPED observation remains2026-10-20T03:42:30.871525UTC;
the raw attempt's older fallback timestamp is retained but superseded by that
later actual receipt. No VM mutation or paid job is queued. No future lifecycle
reliability is inferred. F/separate components/missing136/heldout31of32/original
source authority/geometry/physics/boxes/caps/centers/tolerances/gates stay unchanged.

NEXT derive one NEW construction that uses the saved training hard-source
predicate during geometric/motion restoration, with explicit discrete-raster
acceptance and meaningful new sign/model/acceptance contracts and independent
canonical bindings before any bounded evaluation. No method is yet selected or
queued. Reuse all completed training datasets/19controls/BVH/oracle evidence BY
HASH; never replay them or the rejected soft method. No heldout/full-pixel
feedback, failed-point warmstart, framewise tuning, budget extension or gate
change is allowed. Object changes alone do not repair8right-self failures.
Actual whole151source/URDF/root+finger+object motion, fresh FULL1mm native
hand/object/support ANDself, full swept, actual motor/grasp and whole-source
qualification still precede GPU/PPO. Owner trajectory goal remains unachieved;
this research checkpoint is not goal completion.
'''
(A/'README.md').write_text('# Training-only source constraints\n\n'+note+'\n')
for path in [R/'docs/team/dex3-isaac-review/SOURCE_OBJECT_POSE_DEPTH_20261019.md',
             R/'docs/team/dex3-isaac-review/WRIST_RETARGET_CONTACT_20261019.md',
             R/'scripts/dex3_training/SOURCE_RAY_TEMPORAL.md']:
    with path.open('a') as stream:
        stream.write('\n'+note+'\n')
write_private(C/'training-source-archive-preparation-v1.json', dict(
    schema='v2d.training-source-archive-preparation.v1', created_utc=now,
    archive=str(A), source_plan_helper_count=len(files),
    archive_manifest_sha256=sha(A/'source-manifest.json'),
    guarded_jobs=guards, guarded_jobs_serial=True,
    worker_plan_dependency_hashes_verified=True,
    all_completed_work_reused_WITHOUT_reexecution=True,
    no_new_nonzero_episode_pose_solver_or_physics=True,
    source_inputs_geometry_arrays_results_publicly_uploaded=False))
print(json.dumps(dict(archive=str(A), exact_files=len(files), jobs=len(guards),
                      original_gates_unchanged=True, trajectory_qualified=False)))

"""Exact pinned native compiler and full archived controls on the Mac CPU."""
import hashlib
import json
import sys
import time
from pathlib import Path

import mujoco
import newton
import numpy as np
import warp as wp

S = Path(__file__).resolve().parents[1]
R = Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0] = [str(S/'cpu-native-runtime-v1/src'),str(R/'scripts/dex3_training')]
import audit_hand_reference_contacts as hands
import audit_reference_contacts as objects
from check_native_scene import compile_scene

P = S/'scored-cohort-14-results/policy-e037'
inputs = P/'frozen'
OUT = S/'local-native-contact-control-v1'
OUT.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
started = time.monotonic()
report = dict(schema='v2d.local-native-cpu-control.v1',episode=37,
              producer_sha256=sha(Path(__file__)), source_inputs_modified=False,
              physics_stepped=False, physical_task_accepted=False,
              runtime=dict(mujoco=mujoco.__version__,newton=newton.__version__,warp=wp.__version__),
              frozen_manifest_sha256=sha(inputs/'manifest.json'),
              code_source='ef21fe0c6c9db0154cdbfaed96cc4b44ac5e67dd',
              cpu_runtime_source_proof_sha256=sha(S/'cpu-native-runtime-v1/source-copy-proof.json'))
save = lambda: (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save()
try:
    compiled = compile_scene(inputs)
    scene,reference,solver,manifest_sha,runtime = compiled
    model = solver.mj_model
    mujoco.mj_saveModel(model,str(OUT/'native-model.mjb'),None)
    original = np.load(P/'readback/native-model-readback.npz',allow_pickle=False)
    comparisons = []
    for key in original.files:
        value = getattr(model.opt,key[7:]) if key.startswith('option_') else getattr(model,key)
        value,expected = np.asarray(value),original[key]
        same_shape = value.shape == expected.shape
        error = float(np.max(abs(value.astype(float)-expected.astype(float)),initial=0)) if same_shape else None
        comparisons.append(dict(array=key,same_shape=same_shape,exact=bool(same_shape and np.array_equal(value,expected)),maximum_absolute_difference=error))
    report['native_array_comparisons'] = comparisons
    report['all_recorded_native_arrays_exact'] = all(a['exact'] for a in comparisons)
    save()
    # Reuse the one exact compiler product. Auditing functions and source
    # contracts remain unchanged, including their every-body FK assertions.
    hands.compile_scene = lambda candidate: compiled if candidate.resolve() == inputs.resolve() else (_ for _ in ()).throw(ValueError('Unexpected source'))
    objects.compile_scene = hands.compile_scene
    hand = hands.audit(inputs.resolve())
    obj = objects.audit(inputs.resolve())
    (OUT/'hand-reference-contacts.json').write_text(json.dumps(hand,indent=2)+'\n')
    (OUT/'object-reference-contacts.json').write_text(json.dumps(obj,indent=2)+'\n')
    prior_hand = json.loads((S/'all-scored-hand-audit-results-v1/e037.json').read_text())
    prior_object = json.loads((P/'reference-contacts.json').read_text())

    def summarize_frames(h,o):
        result = {}
        for frame in h['frames']:
            for c in frame['contacts']:
                key = (frame['reference_frame'],c['hand'],c['target'])
                result[key] = max(result.get(key,0.),max(0.,-c['signed_gap_m']))
        for frame in o['frames']:
            for c in frame['contact_pairs']:
                key = (frame['reference_frame'],*c['bodies'])
                result[key] = c['maximum_penetration_m']
        return result

    actual,expected = summarize_frames(hand,obj),summarize_frames(prior_hand,prior_object)
    differences = [dict(frame=k[0],pair=k[1:],local_penetration_m=actual.get(k,0.),archived_penetration_m=expected.get(k,0.)) for k in sorted(set(actual)|set(expected))]
    report['frame_pair_comparisons'] = differences
    report['maximum_penetration_difference_m'] = max(abs(row['local_penetration_m']-row['archived_penetration_m']) for row in differences)
    report['same_over_1mm_frame_pairs'] = {k for k,v in actual.items() if v > .001} == {k for k,v in expected.items() if v > .001}
    report['all_full_reference_controls_passed'] = bool(report['maximum_penetration_difference_m'] < 1e-5 and report['same_over_1mm_frame_pairs'] and hand['all_native_hand_body_fk_verified'])
    report['full_native_checks_remain_required'] = True
    report['quality_gate_passed'] = False  # This control intentionally has known failures.
    manifest = json.loads((inputs/'manifest.json').read_text())
    for name,h in manifest['files'].items():
        assert sha(inputs/name) == h
    report['frozen_source_closure_verified_after'] = True
except BaseException as error:
    report['error'] = repr(error)
    raise
finally:
    report['wall_seconds'] = time.monotonic()-started
    save()
    print(json.dumps({k:v for k,v in report.items() if k not in ['native_array_comparisons','frame_pair_comparisons']}),flush=True)

"""Derive a GT-free geometry-only control from a frozen fixed-axis spin curve."""
from pathlib import Path
import copy
import hashlib
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

base = Path(__file__).resolve().parent
repo = base.parents[1] / 'issue5-repairs'
sys.path.insert(0, str(repo/'scripts/track3'))
import register_poses as reg
from lid_spin import spin_rotation

baseline_path = base/'portable-pose-candidate-support/probe_results.json'
curve_path = base/'portable-lid-spin-final/probe_results.json'
out = base/'lid-spin-geometry-replay.json'
if out.exists():
    raise ValueError('fresh output required')
baseline_raw, curve_raw = baseline_path.read_bytes(), curve_path.read_bytes()
report, curve = json.loads(baseline_raw), json.loads(curve_raw)
assert report['gt_use'] == curve['gt_use'] == 'none'
assert curve['observation_input_sha256'] == hashlib.sha256(baseline_raw).hexdigest()
report = copy.deepcopy(report)
report.update(producer='derived geometry-only fixed-axis coarse-sweep control',
              baseline_sha256=hashlib.sha256(baseline_raw).hexdigest(),
              curve_prediction_sha256=hashlib.sha256(curve_raw).hexdigest(),
              worker_sha256_lf=hashlib.sha256(Path(__file__).read_text().encode()).hexdigest())
report['params']['geometry_only_lid_spin'] = {'selection': 'minimum saved coarse geometric residual',
                                            'translation_fixed': True, 'axis_tilt_fixed': True,
                                            'gt_use': 'none'}
axis = np.asarray(curve['params']['lid_spin']['axis_in_cad'])
for row, samples in zip(report['frames'], curve['frames'], strict=True):
    assert (row['episode'], row['frame_index']) == (samples['episode'], samples['frame_index'])
    best = min(samples['lid_spin_diagnostic']['coarse_curve'], key=lambda s:s['geometric_residual_m'])
    fit = row['fits']['white_pot_lid']
    R = Rotation.from_quat(np.asarray(fit['rotation_wxyz'])[[1,2,3,0]]).as_matrix()
    fit['rotation_wxyz'] = reg._mat_to_wxyz(spin_rotation(R, axis, best['angle_deg'])).tolist()
    fit['baseline_geometric_residual_m'] = fit['residual_m']
    fit['residual_m'] = best['geometric_residual_m']
    fit['spin_observability'] = 'geometry_only_not_verified'
    if fit['residual_m'] > .05:
        fit.update(status='rejected', reason='geometry_only_residual_gate')
    row['geometry_only_spin_diagnostic'] = best
out.write_text(json.dumps(report,indent=2))
assert baseline_path.read_bytes() == baseline_raw and curve_path.read_bytes() == curve_raw
print(out)

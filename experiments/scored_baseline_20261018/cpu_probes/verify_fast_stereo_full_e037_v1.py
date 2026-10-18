"""Independent closure/clock check; never rerun stereo inference."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parents[1]
OUT = S / 'fast-stereo-cpu-full-e037-v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
report_path = OUT / 'depth-report.json'
guard_path = OUT / 'guard.json'
r = json.loads(report_path.read_text())
g = json.loads(guard_path.read_text())
assert r['all_frames_complete'] and r['frozen_source_closure_verified_after']
assert g['worker_exit_code'] == 0 and 'completed_utc' in g and 'error' not in g
assert r['producer_sha256'] == g['worker_sha256']
assert r['source_frames'] == 151
assert [row['frame'] for row in r['complete']] == list(range(151))
assert r['providers'] == ['CPUExecutionProvider']
assert r['source_inputs_modified'] is False and r['source_confidence_modified'] is False
for name, h in r['source_files_sha256'].items():
    assert sha(S / name) == h
with np.load(S / 'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz', allow_pickle=False) as z:
    axis, timestamps = z['frame_indices'].copy(), z['timestamps_s'].copy()
np.testing.assert_array_equal(axis, np.arange(151))
np.testing.assert_allclose(np.diff(timestamps), .05, atol=1e-8, rtol=0)
prior = S / 'fast-stereo-cpu-seven-v1'
prior_report = json.loads((prior / 'depth-report.json').read_text())
assert sha(prior / 'depth-report.json') == r['previous_seven_report_sha256']
samples = [0, 25, 50, 75, 100, 125, 150]
assert [row['frame'] for row in r['complete'] if row['reused_exact_verified_sample']] == samples
files = {}
for row in r['complete']:
    frame = row['frame']
    p = OUT / row['path']
    assert sha(p) == row['sha256']
    with np.load(p, allow_pickle=False) as z:
        assert set(z.files) == {'depth_m', 'left_right_consistent', 'frame_index', 'timestamp_s'}
        depth, valid = z['depth_m'], z['left_right_consistent']
        assert depth.shape == valid.shape == (800, 1280)
        assert depth.dtype == np.float32 and valid.dtype == np.bool_
        assert np.isfinite(depth).all()
        assert ((depth[valid] > .05) & (depth[valid] < 3.)).all()
        assert int(z['frame_index']) == frame and int(valid.sum()) == row['valid_pixels']
        np.testing.assert_array_equal(z['timestamp_s'], timestamps[frame])
        if frame in samples:
            old = prior / f'{frame:03d}-learned.npz'
            assert sha(old) == next(x['output_sha256'] for x in prior_report['complete'] if x['frame'] == frame)
            with np.load(old, allow_pickle=False) as sample:
                np.testing.assert_array_equal(depth, sample['depth_m'])
                np.testing.assert_array_equal(valid, sample['left_right_consistent'])
    files[row['path']] = dict(bytes=p.stat().st_size, sha256=row['sha256'])
verification = dict(schema='v2d.full-cpu-depth-closure-verification.v1',
    completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    producer_sha256=sha(Path(__file__)), depth_report_sha256=sha(report_path),
    guard_sha256=sha(guard_path), source_frames=151, source_fps=20,
    verified_files=files, identical_reused_sample_frames=samples,
    reused_sample_source_pixels_verified_in_sequential_worker=True,
    source_closure_verified=True, source_clock_verified=True,
    all_estimated_depth_arrays_finite=True, valid_mask_metric_range_verified=True,
    inference_rerun=False, depth_accuracy_ground_truth_verified=False,
    trajectory_promoted=False, training_steps=0, physical_task_accepted=False,
    passed=True)
with (OUT / 'verification-report.json').open('x') as stream:
    stream.write(json.dumps(verification, indent=2)+'\n')
print(json.dumps({k: v for k, v in verification.items() if k != 'verified_files'}, indent=2))

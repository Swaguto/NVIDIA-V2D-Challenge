"""Bounded smooth camera normal/height hypothesis from background observations.

Only training-side plane observations enter the fit. Held-out pixels are checked
after solving. Missing plane observations remain explicitly inferred, and this
does not promote cameras, object poses, robot targets or a policy.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation

S = Path(__file__).resolve().parents[1]
R = Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0, str(R / 'scripts/track3'))
from temporal_pose_path import motion_summary

P = S / 'full-static-support-observations-v1'
ROOT = S / 'scored-cohort-14-results/reconstruction-e037'
OUT = S / 'smooth-plane-camera-alignment-v2'
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
observation_report = json.loads((P / 'report.json').read_text())
assert observation_report['all_frames_complete'] and observation_report['source_closures_verified_after']
assert json.loads((P / 'guard.json').read_text())['worker_exit_code'] == 0
assert sha(P / 'background-plane-observations.npz') == observation_report['output_sha256']
with np.load(P / 'background-plane-observations.npz', allow_pickle=False) as z:
    observations = {k: z[k].copy() for k in z.files}
bundle = ROOT / 'bundles/e037/episode.npz'
assert sha(bundle) == observation_report['source_bundle_sha256']
with np.load(bundle, allow_pickle=False) as z:
    source = {k: z[k].copy() for k in z.files}
camera = source['camera_c2w']
plane = np.asarray(json.loads((ROOT / 'support/e037/report.json').read_text())['plane_world'])
n = plane[:3]
u = np.cross(n, np.eye(3)[np.argmin(abs(n))]); u /= np.linalg.norm(u)
basis = np.stack([u, np.cross(n, u)], axis=1)
scales = np.array([.02, .02, .005])
count = len(camera)
qualified = observations['plane_qualified']
y = np.zeros((count, 3))
weight = np.zeros(count)
for frame in np.flatnonzero(qualified):
    lp = observations['plane_world'][frame]
    cross = np.cross(lp[:3], n)
    angle = np.arctan2(np.linalg.norm(cross), np.dot(lp[:3], n))
    rotvec = cross/max(np.linalg.norm(cross), 1e-15)*angle
    y[frame, :2] = basis.T@rotvec
    # Pivot at the original optical centre. The proposed camera translation is
    # normal-only; plane evidence supplies no tangential translation or yaw.
    center = camera[frame, :3, 3]
    y[frame, 2] = np.dot(lp[:3]-n, center)+lp[3]-plane[3]
    points = observations['training_points_world'][frame, :observations['training_point_counts'][frame]]
    residual = abs(np.einsum('ni,i->n', points, lp[:3], optimize=False)+lp[3])
    weight[frame] = np.clip((.001/max(.0005, float(np.median(residual))))**2, .25, 4.)
normalized = y/scales
report = dict(schema='v2d.smooth-plane-camera-alignment.v2', episode=37,
    producer_sha256=sha(Path(__file__)), observation_report_sha256=sha(P / 'report.json'),
    source_bundle_sha256=sha(bundle), observation_npz_sha256=sha(P / 'background-plane-observations.npz'),
    source_camera_estimates_are_not_ground_truth=True,
    camera_drift_versus_depth_bias_resolved=False,
    translation_gauge='Normal-only camera-centre displacement; tangential translation and yaw unchanged.',
    smoothing=dict(first_difference_weight=20., second_difference_weight=100.,
        rotation_scale_rad=.02, normal_position_scale_m=.005,
        max_rotation_norm_rad=.05, max_normal_displacement_m=.025,
        max_adjacent_rotation_component_rad=.02, max_second_rotation_component_rad=.02,
        max_adjacent_normal_displacement_m=.0015, max_second_normal_displacement_m=.0015),
    heldout_used_for_optimization=False, source_flags_changed=False,
    raw_source_files_changed=False, source_clock_changed=False,
    missing_measurements_promoted_to_observed=False,
    support_changed=False, object_geometry_changed=False,
    source_alignment_stage_passed=False, all_frames_complete=False,
    native_contact_or_object_motion_certificate=False,
    physics_stepped=False, training_steps=0, trajectory_promoted=False, physical_task_accepted=False)
save = lambda: (OUT / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
started = time.monotonic()
save()
try:
    I = sparse.eye(count*3, format='csc')
    first = sparse.kron(sparse.diags([-np.ones(count-1), np.ones(count-1)], [0, 1], shape=(count-1, count)), sparse.eye(3), format='csc')
    second = sparse.kron(sparse.diags([np.ones(count-2), -2*np.ones(count-2), np.ones(count-2)], [0, 1, 2], shape=(count-2, count)), sparse.eye(3), format='csc')
    W = sparse.diags(np.repeat(weight, 3), format='csc')
    H = sparse.triu(2*(W+20*first.T@first+100*second.T@second+1e-10*I), format='csc')
    gradient = -2*np.repeat(weight, 3)*normalized.ravel()
    absolute = np.tile(np.array([.05/np.sqrt(2), .05/np.sqrt(2), .025])/scales, count)
    step = np.array([.02, .02, .0015])/scales
    A = sparse.vstack([I, first, second], format='csc')
    upper = np.r_[absolute, np.tile(step, count-1), np.tile(step, count-2)]
    # Add source-centroid constraints at every qualified frame. Only the
    # training-side plane inliers define them; held-out cells remain untouched.
    centroids, tolerances = {}, {}
    for frame in np.flatnonzero(qualified):
        points = observations['training_points_world'][frame, :observations['training_point_counts'][frame]]
        lp = observations['plane_world'][frame]
        inside = abs(np.einsum('ni,i->n', points, lp[:3], optimize=False)+lp[3]) < .004
        assert inside.sum() >= 10
        centroids[frame] = points[inside].mean(0)
        original_residual = abs(np.einsum('ni,i->n', points[inside], n, optimize=False)+plane[3])
        tolerances[frame] = min(.001, max(.00001, .5*float(np.median(original_residual))))
    def centroid_height(frame, normalized_parameters):
        p = normalized_parameters*scales
        rotation = Rotation.from_rotvec(basis@p[:2]).as_matrix()
        center = camera[frame, :3, 3]
        transformed = center+rotation@(centroids[frame]-center)+n*p[2]
        return float(np.dot(n, transformed)+plane[3])
    current = np.zeros((count, 3))
    report['centroid_constraints'] = dict(training_only=True, frames=len(centroids),
        tolerance='min(1mm, half original training-inlier median absolute height), with 10um numerical floor',
        source_or_motion_gates_relaxed=False)
    report['iterations'] = []
    for iteration in range(5):
        rr, cc, vv, lows, highs = [], [], [], [], []
        for index, frame in enumerate(centroids):
            x = current[frame]
            height = centroid_height(frame, x)
            jacobian = np.zeros(3)
            for component in range(3):
                delta = np.eye(3)[component]*1e-4
                jacobian[component] = (centroid_height(frame, x+delta)-centroid_height(frame, x-delta))/(2e-4)
            jacobian /= .005
            rr.extend([index]*3); cc.extend((frame*3+np.arange(3)).tolist()); vv.extend(jacobian.tolist())
            offset = float(np.dot(jacobian, x))-height/.005
            lows.append(offset-tolerances[frame]/.005); highs.append(offset+tolerances[frame]/.005)
        source_A = sparse.csc_matrix((vv, (rr, cc)), shape=(len(centroids), count*3))
        combined_A = sparse.vstack([A, source_A], format='csc')
        solver = osqp.OSQP()
        solver.setup(P=H, q=gradient, A=combined_A, l=np.r_[-upper, lows], u=np.r_[upper, highs],
                     eps_abs=1e-8, eps_rel=1e-8, max_iter=100000, time_limit=30, polishing=True, verbose=False)
        solution = solver.solve(raise_error=False)
        assert solution.info.status == 'solved'
        current = solution.x.reshape(count, 3)
        maximum_excess = max(abs(centroid_height(f, current[f]))-tolerances[f] for f in centroids)
        report['iterations'].append(dict(iteration=iteration, status=solution.info.status,
            maximum_exact_training_centroid_excess_m=float(maximum_excess), solve_seconds=float(solution.info.run_time)))
        if maximum_excess <= 1e-8:
            break
    report['centroid_hard_gates_passed'] = bool(maximum_excess <= 1e-8)
    report['solver'] = dict(status=solution.info.status, iterations=int(solution.info.iter), seconds=float(solution.info.run_time))
    parameters = current*scales
    alignments = np.repeat(np.eye(4)[None], count, axis=0)
    candidate = camera.copy()
    rows = []
    max_projection_change = 0.
    max_tangential_displacement = 0.
    K = source['intrinsics']
    for frame in range(count):
        r = basis@parameters[frame, :2]
        rotation = Rotation.from_rotvec(r).as_matrix()
        center = camera[frame, :3, 3]
        transform = alignments[frame]
        transform[:3, :3] = rotation
        transform[:3, 3] = center-rotation@center+n*parameters[frame, 2]
        candidate[frame] = transform@camera[frame]
        displacement = candidate[frame, :3, 3]-center
        max_tangential_displacement = max(max_tangential_displacement, float(np.linalg.norm(displacement-n*np.dot(n, displacement))))
        row = dict(frame=frame, plane_directly_qualified=bool(qualified[frame]), alignment_inferred_only=not bool(qualified[frame]))
        if qualified[frame]:
            points = observations['heldout_points_world'][frame, :observations['heldout_point_counts'][frame]]
            before = abs(np.einsum('ni,i->n', points, n, optimize=False)+plane[3])
            transformed = np.einsum('ni,ji->nj', points, rotation, optimize=False)+transform[:3, 3]
            after = abs(np.einsum('ni,i->n', transformed, n, optimize=False)+plane[3])
            row.update(original_heldout_median_m=float(np.median(before)), candidate_heldout_median_m=float(np.median(after)),
                original_heldout_p95_m=float(np.quantile(before, .95)), candidate_heldout_p95_m=float(np.quantile(after, .95)),
                heldout_source_gate_passed=bool(np.median(after) <= .004 and np.median(after) < np.median(before)))
        for si in range(2):
            if not(source['hand_valid'][frame, si] and source['hand_directly_observed'][frame, si]):
                continue
            points = source['hand_joints_world'][frame, si]
            transformed = np.einsum('ni,ji->nj', points, rotation, optimize=False)+transform[:3, 3]
            before = np.einsum('ni,ij->nj', points-center, camera[frame, :3, :3], optimize=False)
            after = np.einsum('ni,ij->nj', transformed-candidate[frame, :3, 3], candidate[frame, :3, :3], optimize=False)
            finite = np.isfinite(before).all(1) & np.isfinite(after).all(1) & (before[:, 2] > .01) & (after[:, 2] > .01)
            if finite.any():
                error = abs((before[finite, :2]/before[finite, 2:3]-after[finite, :2]/after[finite, 2:3])*[K[0, 0], K[1, 1]])
                max_projection_change = max(max_projection_change, float(error.max()))
        rows.append(row)
    m = report['smoothing']
    bounds = dict(maximum_rotation_norm_rad=float(np.linalg.norm(parameters[:, :2], axis=1).max()),
        maximum_normal_displacement_m=float(abs(parameters[:, 2]).max()),
        maximum_adjacent_rotation_component_rad=float(abs(np.diff(parameters[:, :2], axis=0)).max()),
        maximum_second_rotation_component_rad=float(abs(np.diff(parameters[:, :2], n=2, axis=0)).max()),
        maximum_adjacent_normal_displacement_m=float(abs(np.diff(parameters[:, 2])).max()),
        maximum_second_normal_displacement_m=float(abs(np.diff(parameters[:, 2], n=2)).max()))
    bounds_pass = (bounds['maximum_rotation_norm_rad'] <= .05+1e-8 and bounds['maximum_normal_displacement_m'] <= .025+1e-8
        and bounds['maximum_adjacent_rotation_component_rad'] <= .02+1e-8 and bounds['maximum_second_rotation_component_rad'] <= .02+1e-8
        and bounds['maximum_adjacent_normal_displacement_m'] <= .0015+1e-8 and bounds['maximum_second_normal_displacement_m'] <= .0015+1e-8)
    report.update(rows=rows, measured_plane_frames=int(qualified.sum()), inferred_only_frames=np.flatnonzero(~qualified).tolist(),
        motion_bounds=bounds, correction_motion_bounds_passed=bool(bounds_pass),
        maximum_source_hand_projection_change_px=max_projection_change,
        maximum_unobserved_tangential_camera_translation_m=max_tangential_displacement,
        source_gate_failed_frames=[r['frame'] for r in rows if r.get('heldout_source_gate_passed') is False],
        camera_motion=dict(original=motion_summary(camera, source['timestamps_s']), candidate=motion_summary(candidate, source['timestamps_s'])))
    assert max_projection_change < 1e-8 and max_tangential_displacement < 1e-10
    report['source_alignment_stage_passed'] = bool(bounds_pass and report['centroid_hard_gates_passed'] and not report['source_gate_failed_frames'])
    np.savez_compressed(OUT / 'inferred-world-alignment.npz', world_alignments=alignments,
        candidate_camera_c2w=candidate, original_camera_c2w=camera, correction_parameters=parameters,
        source_plane_qualified=qualified, alignment_inferred_only=~qualified,
        frame_indices=source['frame_indices'], timestamps_s=source['timestamps_s'])
    report['candidate_sha256'] = sha(OUT / 'inferred-world-alignment.npz')
    assert sha(bundle) == report['source_bundle_sha256']
    assert sha(P / 'report.json') == report['observation_report_sha256']
    report['source_closures_verified_after'] = True
    report['all_frames_complete'] = True
except BaseException as error:
    report['error'] = repr(error)
    raise
finally:
    report['wall_seconds'] = time.monotonic()-started
    save()
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)

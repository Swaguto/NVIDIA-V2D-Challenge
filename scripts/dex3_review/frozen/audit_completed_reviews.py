"""Audit published replay masks, source axes and simulator evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np

R=Path('/workspace/hand-continuity-repair/dex3-isaac-review')
rows=[]
for root in sorted((R/'episodes').glob('e*')):
    sim=root/('sim-v2' if root.name=='e011' else 'sim-v1')
    if not (sim/'complete.json').is_file():continue
    receipt=json.loads((sim/'complete.json').read_text())
    with np.load(root/'bundle/episode.npz') as data, np.load(sim/'simulator-readback.npz') as readback:
        np.testing.assert_array_equal(readback['frame_indices'],data['frame_indices'])
        np.testing.assert_array_equal(readback['simulator_display_valid'],data['hand_valid'])
        np.testing.assert_array_equal(readback['source_metric_valid'],data['hand_metric_valid'])
        assert not readback['robot_motion_target_validated'].any()
        assert np.isnan(readback['robot_qpos'][~data['hand_valid']]).all()
        maxima=[]
        for side in range(2):
            unseen=~data['hand_valid'][:,side]
            colors=readback['hand_review_color_pixels'][unseen,side]
            maxima.append(int(colors.max()) if len(colors) else 0)
        # Colors are strictly separated from the neutral review CAD materials.
        assert max(maxima)<=20,(root.name,'unknown hand is visible',maxima)
        row={'episode':root.name,'frames':len(data['frame_indices']),
             'visual_counts':data['hand_valid'].sum(axis=0).tolist(),
             'stereo_counts':data['hand_metric_valid'].sum(axis=0).tolist(),
             'unknown_hand_maximum_assigned_color_pixels':maxima,
             'source_axis_and_masks_match':True,
             'physical_task_accepted':False,
             'root_error_m':receipt['maximum_root_position_error_m'],
             'joint_error_rad':receipt['maximum_joint_error_rad'],
             'camera_position_error_m':receipt['maximum_camera_position_error_m'],
             'camera_projection_error_px':receipt['maximum_camera_intrinsics_error_px'],
             'renderer_sha256':receipt['source_sha256']}
    rows.append(row)
(R/'completed-review-audit.json').write_text(json.dumps({'episodes':rows,'complete_count':len(rows),'user_grasp_acceptance':False},indent=2))
print('AUDITED_RENDERERS',len(rows),sum(row['frames'] for row in rows),flush=True)

"""Compare fixed source pairs with geometric diagnostics, never hidden labels."""
from pathlib import Path
import ast
import hashlib
import json
import sys

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

S = Path(__file__).resolve().parents[1]
R = Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0, str(R/'scripts/track3'))
from select_pose_hypothesis import observed_fit

inputs = S/'learned-stereo-inputs-v1'
results = S/'fast-stereo-cpu-seven-v1'
OUT = S/'fast-stereo-source-comparison-seven-v1'
OUT.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
meta = json.loads((inputs/'manifest.json').read_text())
inference = json.loads((results/'depth-report.json').read_text())
assert inference['all_seven_complete'] and inference['training_steps'] == 0
assert inference['source_manifest_sha256'] == sha(inputs/'manifest.json')
for name, proof in meta['files'].items():
    assert sha(inputs/name) == proof['sha256']
source_audit = S/'compute/spoon_source_support_audit_v1.py'
nodes = [node for node in ast.parse(source_audit.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in {'stats', 'points', 'local_plane'}]
assert len(nodes) == 3
namespace = dict(np=np)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_audit), 'exec'), namespace)
stats, points, local_plane = [namespace[name] for name in ['stats', 'points', 'local_plane']]
K = np.asarray(meta['rectified_intrinsics'])
namespace['K'] = K
root = S/'scored-cohort-14-results/reconstruction-e037'
plane = np.asarray(json.loads((root/'support/e037/report.json').read_text())['plane_world'])
mesh_path = root/'geometry/e037/wooden_spoon/mesh.glb'
mesh = trimesh.load(mesh_path, force='mesh', process=False)
vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
report = dict(schema='v2d.fast-stereo-cpu-source-geometric-comparison.v1', episode=37, rows=[],
              source_manifest_sha256=sha(inputs/'manifest.json'),
              inference_report_sha256=sha(results/'depth-report.json'),
              producer_sha256=sha(Path(__file__)),
              audit_functions_source_sha256=sha(source_audit),
              source_mesh_sha256=sha(mesh_path),
              source_inputs_modified=False, trajectory_promoted=False,
              reference_labels_read=False, depth_accuracy_against_ground_truth_known=False,
              physical_task_accepted=False,
              scope='Measured stereo estimates; geometric consistency is diagnostic, not a depth-accuracy certificate.')
for frame in meta['frames']:
    with np.load(inputs/f'{frame:03d}-control.npz', allow_pickle=False) as z:
        control, cv, camera = z['depth_m'].copy(), z['valid'].copy(), z['camera_c2w'].copy()
    with np.load(results/f'{frame:03d}-learned.npz', allow_pickle=False) as z:
        learned, lv = z['depth_m'].copy(), z['left_right_consistent'].copy()
    mask = cv2.imread(str(inputs/f'{frame:03d}-mask.png'), 0) > 127
    interior = cv2.erode(mask.astype(np.uint8), np.ones((5,5), np.uint8)) > 0
    distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    pose_path = root/f'source/e037/poses/e037/ego_object_poses/e037/wooden_spoon/object_to_cam/ego_cam_c/{frame:06d}.json'
    record = json.loads(pose_path.read_text())
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(np.asarray(record['rotation'])[[1,2,3,0]]).as_matrix()
    pose[:3, 3] = record['translation']
    row = dict(frame=frame, source_pose_sha256=sha(pose_path), backends=[])
    panels = [cv2.imread(str(inputs/f'{frame:03d}-left.png'))]
    for label, depth, valid in [('SGBM-control', control, cv), ('C-FastFoundationStereo-CPU-320x512-LR1px', learned, lv)]:
        metric = valid & np.isfinite(depth) & (depth > .15) & (depth < 1.4)
        p, yy, xx = points(depth, metric)
        world = np.einsum('ni,ji->nj', p, camera[:3,:3], optimize=False)+camera[:3,3]
        h = np.einsum('ni,i->n', world, plane[:3], optimize=False)+plane[3]
        fg = mask[yy,xx]
        eroded = interior[yy,xx]
        near = (distance[yy,xx] > 26) & (distance[yy,xx] < 100) & (abs(h) < .035)
        split = ((xx//8+yy//8) % 2) == 0
        ids = np.flatnonzero(near & split)[::3]
        fitted = local_plane(world[ids], plane)
        measures = dict(backend=label, mask_coverage=float((metric & mask).sum()/mask.sum()),
                        foreground_global_plane_height=stats(h[fg]),
                        interior_global_plane_height=stats(h[eroded]),
                        near_background_global_plane_height=stats(h[near]),
                        shape_fits=[])
        for scale in [1.,1.3]:
            v = vertices*scale
            measures['shape_fits'].append(dict(uniform_scale=scale, **observed_fit(v,faces,cKDTree(v),pose,depth,valid,mask,K)))
        if fitted is not None:
            lp, inliers = fitted
            local_h = np.einsum('ni,i->n', world, lp[:3], optimize=False)+lp[3]
            measures['local_plane'] = dict(plane_world=lp.tolist(),fit_points=len(ids), fit_inliers=int(inliers.sum()),
                                          heldout_background_abs_residual=stats(abs(local_h[near & ~split])),
                                          foreground_signed_height=stats(local_h[fg]),
                                          interior_signed_height=stats(local_h[eroded]))
        image = panels[0].copy()//3
        colors = cv2.applyColorMap(np.clip((h+.035)/.07*255,0,255).astype(np.uint8)[:,None], cv2.COLORMAP_TURBO)[:,0]
        roi = cv2.dilate(mask.astype(np.uint8), np.ones((201,201),np.uint8)) > 0
        selected = roi[yy,xx]
        image[yy[selected],xx[selected]] = colors[selected]
        cv2.drawContours(image,cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,0),2)
        cv2.rectangle(image,(0,0),(1280,36),(0,0,0),-1)
        cv2.putText(image,f'e037 f{frame}: {label}; global height +/-35mm', (12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        panels.append(image)
        row['backends'].append(measures)
    common = cv & lv & mask & (control > .15) & (control < 1.4) & (learned > .15) & (learned < 1.4)
    row['same_valid_foreground_depth_difference_m'] = stats(abs(control[common]-learned[common]))
    cv2.imwrite(str(OUT/f'frame-{frame:03d}.jpg'),np.concatenate([cv2.resize(im,(640,400)) for im in panels],axis=1))
    report['rows'].append(row)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(frame=frame, backends=[dict(name=b['backend'],coverage=b['mask_coverage'],
         interior_below_local_table=b.get('local_plane',{}).get('interior_signed_height',{}).get('below_minus_1mm_fraction'),
         local_background_residual_m=b.get('local_plane',{}).get('heldout_background_abs_residual',{}).get('quantiles_m',[None]*5)[2]) for b in row['backends']])),flush=True)

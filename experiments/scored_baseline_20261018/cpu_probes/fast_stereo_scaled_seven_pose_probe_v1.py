"""Repeat the retained rigid-pose probe with only the depth backend changed.

Three predetermined source frames, the same two geometry controls, same fitter,
same original support, cameras, masks and rejection logic. SGBM comparisons are
reported separately, so a new estimate is not represented as old observations.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(R/'scripts/track3'),str(S/'compute')]
from measured_rigid_pose_probe_v2 import refine
from measured_pose_refinement import finite_support_overlap
from foundationpose_track import pose_rejection
from select_pose_hypothesis import observed_fit
from camera_prior import transport

ROOT=S/'scored-cohort-14-results/reconstruction-e037'
depth_root=S/'fast-stereo-cpu-seven-v1';frozen=S/'learned-stereo-inputs-v1'
OUT=S/'fast-stereo-scaled-seven-pose-probe-v1';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inference=json.loads((depth_root/'depth-report.json').read_text());assert inference['all_seven_complete']
meta=json.loads((frozen/'manifest.json').read_text());K=np.asarray(meta['rectified_intrinsics'])
for name,proof in meta['files'].items():assert sha(frozen/name)==proof['sha256']
with np.load(ROOT/'bundles/e037/episode.npz',allow_pickle=False) as z:camera=z['camera_c2w'].copy();np.testing.assert_allclose(K,z['intrinsics'],atol=1e-6,rtol=0)
world_plane=np.asarray(json.loads((ROOT/'support/e037/report.json').read_text())['plane_world'])
with np.load(ROOT/'support/e037/inferred-support-sheet.npz',allow_pickle=False) as z:sheet=z['vertices_world_m'].copy()
mesh_path=ROOT/'geometry/e037/wooden_spoon/mesh.glb';mesh=trimesh.load(mesh_path,force='mesh',process=False)
vertices,faces=np.asarray(mesh.vertices),np.asarray(mesh.faces)
poses=ROOT/'source/e037/poses/e037/ego_object_poses/e037/wooden_spoon/object_to_cam/ego_cam_c'
def load(frame):
    j=json.loads((poses/f'{frame:06d}.json').read_text());T=np.eye(4)
    T[:3,:3]=Rotation.from_quat(np.asarray(j['rotation'])[[1,2,3,0]]).as_matrix();T[:3,3]=j['translation'];return T
report=dict(schema='v2d.fast-stereo-scaled-seven-source-probe.v1',episode=37,frames=[25,50,75,150],rows=[],
    producer_sha256=sha(Path(__file__)),refiner_sha256=sha(S/'compute/measured_rigid_pose_probe_v2.py'),
    source_mesh_sha256=sha(mesh_path),source_manifest_sha256=sha(frozen/'manifest.json'),
    depth_inference_report_sha256=sha(depth_root/'depth-report.json'),
    original_depth_control_report_sha256=sha(S/'rigid-spoon-source-probe-v2/report.json'),
    changed_factor='Pretrained C-FastFoundationStereo depth/LR validity instead of SGBM; source pixels and calibrations unchanged.',
    original_source_files_modified=False,original_acceptance_logic_relaxed=False,
    original_support_changed=False,source_confidence_modified=False,
    physics_stepped=False,training_steps=0,trajectory_promoted=False,physical_task_accepted=False)
save=lambda:(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
started=time.monotonic();save()
prior_path=S/'fast-stereo-rigid-pose-probe-v1/report.json'
prior=json.loads(prior_path.read_text())
for row in prior['rows']:
    if row['arm']=='uniform-1.3-shape':
        row['reused_report_sha256']=sha(prior_path);report['rows'].append(row)
report['all_source_sample_frames']=[0,25,50,75,100,125,150];save()
try:
    for frame in report['frames']:
        assert time.monotonic()-started<300,'Bounded CPU rigid-pose probe'
        file=depth_root/f'{frame:03d}-learned.npz'
        expected=next(x['output_sha256'] for x in inference['complete'] if x['frame']==frame);assert sha(file)==expected
        with np.load(file,allow_pickle=False) as z:depth,valid=z['depth_m'].copy(),z['left_right_consistent'].copy()
        with np.load(frozen/f'{frame:03d}-control.npz',allow_pickle=False) as z:control_depth,control_valid=z['depth_m'].copy(),z['valid'].copy()
        mask=cv2.imread(str(frozen/f'{frame:03d}-mask.png'),0)>127
        P=camera[frame];plane=np.r_[np.einsum('ji,j->i',P[:3,:3],world_plane[:3],optimize=False),world_plane[3]+np.dot(world_plane[:3],P[:3,3])]
        footprint=np.einsum('ni,ij->nj',sheet-P[:3,3],P[:3,:3],optimize=False)
        T=load(frame);prior=transport(load(frame-1),np.linalg.inv(camera[frame-1]),np.linalg.inv(P)) if frame else None
        for label,scale in [('uniform-1.3-shape',1.3)]:
            v=vertices*scale;tree=cKDTree(v)
            candidate,fit=refine(v,faces,T,depth,valid,mask,K,plane,footprint)
            rejection,support=pose_rejection(candidate,depth,valid,(mask*255).astype(np.uint8),K,np.stack([v.min(0),v.max(0)]),True,prior)
            old_rejection,old_support=pose_rejection(candidate,control_depth,control_valid,(mask*255).astype(np.uint8),K,np.stack([v.min(0),v.max(0)]),True,prior)
            row=dict(frame=frame,arm=label,fit=fit,learned_depth_gate_rejection=rejection,learned_depth_support=support,
                     retained_sgbm_depth_gate_rejection=old_rejection,retained_sgbm_depth_support=old_support,
                     retained_sgbm_fit_before=observed_fit(v,faces,tree,T,control_depth,control_valid,mask,K),retained_sgbm_fit_after=observed_fit(v,faces,tree,candidate,control_depth,control_valid,mask,K),
                     finite_overlap_before=finite_support_overlap(v,faces,T,plane,footprint),finite_overlap_after=finite_support_overlap(v,faces,candidate,plane,footprint),candidate_camera_from_object=candidate.tolist())
            report['rows'].append(row);save()
            print(json.dumps(dict(frame=frame,arm=label,fit_accepted=fit['accepted'],learned_depth_gate_passed=rejection is None,retained_sgbm_depth_gate_passed=old_rejection is None,finite_minimum_height_m=None if row['finite_overlap_after'] is None else row['finite_overlap_after']['minimum_height_m'])),flush=True)
    for name,proof in meta['files'].items():assert sha(frozen/name)==proof['sha256']
    assert sha(mesh_path)==report['source_mesh_sha256'];report['frozen_source_closure_verified_after']=True
except BaseException as error:report['error']=repr(error);raise
finally:report['wall_seconds']=time.monotonic()-started;save()

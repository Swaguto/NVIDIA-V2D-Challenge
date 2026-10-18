"""Audit image-only bidirectional consistency on the frozen seven stereo pairs.

The one-pixel threshold is fixed before inspecting heights. Failed cycle checks
become missing estimates in a diagnostic copy; no source or production reference
is edited. Geometry, camera and support remain fixed.
"""
import ast
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

S=Path(__file__).resolve().parents[1]
inputs=S/'learned-stereo-inputs-v1'
OUT=S/'sgbm-left-right-source-audit-v2';OUT.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta=json.loads((inputs/'manifest.json').read_text())
for name,proof in meta['files'].items():assert sha(inputs/name)==proof['sha256']
K=np.asarray(meta['rectified_intrinsics']);factor=K[0,0]*meta['baseline_m']
root=S/'scored-cohort-14-results/reconstruction-e037'
plane=np.asarray(json.loads((root/'support/e037/report.json').read_text())['plane_world'])
source=S/'compute/spoon_source_support_audit_v1.py'
nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'stats','points','local_plane'}]
scope=dict(np=np,K=K);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
stats,points,local_plane=[scope[n] for n in ['stats','points','local_plane']]
report=dict(schema='v2d.sgbm-image-cycle-source-audit.v2',producer_sha256=sha(Path(__file__)),source_manifest_sha256=sha(inputs/'manifest.json'),rows=[],source_or_confidence_modified=False,production_depth_replaced=False,source_or_hidden_labels_used=False,depth_accuracy_proven=False,physical_task_accepted=False,new_training_steps=0,cycle_threshold_px=1.,control_protocol='Require exact recovered 1/16-pixel disparities and exact validity mask; use untouched saved control depth for all geometric analysis. The v1 regenerated depth had one-float-ULP arithmetic differences and was rejected before measurement.')
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
started=time.monotonic()
def disparity(left,right):
    left,right=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in [left,right]]
    for nd,bs in [(128,5),(64,7),(32,9)]:
        matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=nd,blockSize=bs,P1=8*bs*bs,P2=32*bs*bs,disp12MaxDiff=1,uniquenessRatio=10,speckleWindowSize=100,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        d=matcher.compute(left,right).astype(np.float32)/16.
        valid=d>0;depth=np.zeros_like(d);depth[valid]=factor/d[valid]
        valid &= (depth>.05)&(depth<3.)
        if valid.sum()>=400:return d,valid,depth
    return d,valid,depth

for frame in meta['frames']:
    assert time.monotonic()-started<180,'Bounded three-minute source-cycle audit'
    left,right=[cv2.imread(str(inputs/f'{frame:03d}-{side}.png')) for side in ['left','right']]
    forward,valid,depth=disparity(left,right)
    reverse,reverse_valid,_=disparity(np.ascontiguousarray(right[:,::-1]),np.ascontiguousarray(left[:,::-1]))
    reverse=reverse[:,::-1].copy();reverse_valid=reverse_valid[:,::-1].copy()
    with np.load(inputs/f'{frame:03d}-control.npz',allow_pickle=False) as z:
        control,cv,camera=z['depth_m'].copy(),z['valid'].copy(),z['camera_c2w'].copy()
    nonzero=control>0
    recovered=np.zeros_like(control);recovered[nonzero]=np.rint((factor/control[nonzero])*16.)/16.
    np.testing.assert_array_equal(forward[nonzero],recovered[nonzero]);np.testing.assert_array_equal(valid,cv)
    regenerated_depth_maximum_difference=float(abs(depth-control).max())
    depth=control
    yy,xx=np.indices(depth.shape,dtype=np.float32);rx=xx-forward
    projected=cv2.remap(reverse,rx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    rv=cv2.remap(reverse_valid.astype(np.uint8),rx,yy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0
    cycle=np.abs(forward-projected)
    consistent=valid&rv&(rx>=0)&(rx<depth.shape[1]-1)&(cycle<=1.)
    mask=cv2.imread(str(inputs/f'{frame:03d}-mask.png'),0)>127
    interior=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8))>0
    distance=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,3)
    metric=valid&np.isfinite(depth)&(depth>.15)&(depth<1.4)
    p,y,x=points(depth,metric)
    world=np.einsum('ni,ji->nj',p,camera[:3,:3],optimize=False)+camera[:3,3]
    heights=np.einsum('ni,i->n',world,plane[:3],optimize=False)+plane[3]
    fg,eroded=mask[y,x],interior[y,x];kept=consistent[y,x]
    near=(distance[y,x]>26)&(distance[y,x]<100)&(abs(heights)<.035)
    split=((x//8+y//8)%2)==0;ids=np.flatnonzero(near&split)[::3]
    fit=local_plane(world[ids],plane)
    row=dict(frame=frame,control_disparity_and_validity_reproduced_exactly=True,original_control_depth_used=True,regenerated_depth_maximum_float_difference_m=regenerated_depth_maximum_difference,mask_pixels=int(mask.sum()),control_metric_mask_pixels=int(fg.sum()),cycle_consistent_metric_mask_pixels=int((fg&kept).sum()),foreground_cycle_pass_fraction=float(kept[fg].mean()),interior_cycle_pass_fraction=float(kept[eroded].mean()),original_foreground_global_heights=stats(heights[fg]),consistent_foreground_global_heights=stats(heights[fg&kept]))
    if fit is not None:
        lp,_=fit;h=np.einsum('ni,i->n',world,lp[:3],optimize=False)+lp[3]
        row['original_source_local_plane']=dict(plane_world=lp.tolist(),original_eroded_heights=stats(h[eroded]),consistent_eroded_heights=stats(h[eroded&kept]),rejected_eroded_heights=stats(h[eroded&~kept]),heldout_background_abs_residual=stats(abs(h[near&~split])))
    np.savez_compressed(OUT/f'{frame:03d}-cycle.npz',left_disparity_px=forward,right_disparity_px=reverse,cycle_error_px=cycle,left_right_consistent=consistent,filtered_depth_m=np.where(consistent,depth,0.))
    report['rows'].append(row);(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(frame=frame,foreground_cycle_pass_fraction=row['foreground_cycle_pass_fraction'],interior_cycle_pass_fraction=row['interior_cycle_pass_fraction'],interior_before_below_fraction=row.get('original_source_local_plane',{}).get('original_eroded_heights',{}).get('below_minus_1mm_fraction'),interior_after_below_fraction=row.get('original_source_local_plane',{}).get('consistent_eroded_heights',{}).get('below_minus_1mm_fraction'))),flush=True)
for name,proof in meta['files'].items():assert sha(inputs/name)==proof['sha256']
report['frozen_source_closure_verified_after']=True;report['wall_seconds']=time.monotonic()-started
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')

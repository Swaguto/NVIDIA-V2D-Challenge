"""Inspect new GPU priors against existing source evidence; never promote a task."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

S=Path(__file__).resolve().parents[2]
R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(R/'scripts/track3'),str(S/'compute')]
from merge_recovered_source_hand import merge,validate_recovery
from numpy_mano_source_geometry_v2 import NumpyMano
from audit_source_hand_surface_depth_v1 import ray_depth

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

request=S/'e037-fresh-source-hand-request-v2/request.json'
recovery=S/'e037-watcloud-fresh-hand-v1/outputs/recovery'
out=S/'e037-watcloud-fresh-hand-source-audit-v1'
assert not out.exists()
data,sources,receipt,accepted,axes=validate_recovery(request,recovery)
out.mkdir()
handoff=merge(request,recovery,out/'unqualified-handoff')
depth_source=S/'e037-source-hand-motion-fit-v1/source-hand-candidate.npz'
gate_source=S/'e037-articulated-source-hand-fit-v1/report.json'
model=S/'models/mano/models/MANO_RIGHT.pkl'
m=NumpyMano(model)
with np.load(data['canonical_bundle'],allow_pickle=False) as z:K=z['intrinsics'].copy()
with np.load(depth_source,allow_pickle=False) as z:depth=z['estimated_depth_m'].copy();depth_uv=z['landmarks_px'].copy()
old=json.loads(gate_source.read_text());gates=old['gates']
heldout=np.isin(np.arange(21),[1,4,7,10,13,16,19])
rows=[]
for key,name,estimate,mask in accepted:
    side,fid=key
    assert side=='right'
    row=next(x for x in data['frames'] if x['frame_idx']==fid)
    points=np.asarray(row['keypoints']);observed=points[:,2]>0
    assert np.array_equal(points[observed,:2],depth_uv[fid,observed])
    record=json.loads((recovery/name).read_text())
    fresh_v,fresh_j,_=m.record(record)
    _,old_j,_=m.record(sources[key][1])
    metrics={}
    for label,joints in [('old_prior',old_j),('fresh_unfitted_prior',fresh_j)]:
        uv=np.c_[K[0,0]*joints[:,0]/joints[:,2]+K[0,2],K[1,1]*joints[:,1]/joints[:,2]+K[1,2]]
        errors=np.linalg.norm(uv-points[:,:2],axis=-1)
        test=observed&heldout;hd=test&np.isfinite(depth[fid])
        median=float(np.median(errors[test])) if test.any() else None
        p95=float(np.quantile(errors[test],.95)) if test.any() else None
        dz=float(np.median(abs(joints[hd,2]-depth[fid,hd]))) if hd.any() else None
        enough=test.sum()>=2 and hd.sum()>=1
        metrics[label]=dict(all_supported_image_median_px=float(np.median(errors[observed])),
            heldout_image_median_px=median,heldout_image_p95_px=p95,
            heldout_estimated_joint_depth_median_error_m=dz,
            historical_source_screen_passed=bool(enough and median<=gates['heldout_median_image_px'] and p95<=gates['heldout_p95_image_px'] and dz<=gates['heldout_median_estimated_depth_m']))
    hd=observed&heldout&np.isfinite(depth[fid]);rays=ray_depth(fresh_v,m.faces,points[hd,:2],K);hits=np.isfinite(rays)
    surface=abs(rays-(depth[fid,hd]-.005))
    kp=np.asarray(json.loads((recovery/estimate).read_text())['keypoints'])
    rows.append(dict(frame=fid,source_points=int(observed.sum()),mask_supported_points=int((kp[:,2]>0).sum()),
        heldout_image_points=int((observed&heldout).sum()),heldout_depth_points=int(hd.sum()),
        fresh_front_surface_hits=int(hits.sum()),fresh_front_surface_rays=len(rays),
        fresh_front_surface_median_error_m=float(np.median(surface[hits])) if hits.any() else None,
        original_directly_observed=row['original_direct_observation'],**metrics))
report=dict(schema='v2d.fresh-source-hand-raw-source-audit.v1',producer_sha256=sha(Path(__file__)),
    recovery_receipt_sha256=sha(recovery/'receipt.json'),request_sha256=sha(request),
    canonical_bundle_sha256=data['canonical_bundle_sha256'],estimated_depth_cache_sha256=sha(depth_source),
    historical_source_gate_report_sha256=sha(gate_source),private_model_sha256=sha(model),
    geometry_helper_sha256=sha(S/'compute/numpy_mano_source_geometry_v2.py'),
    geometric_convention_closure_previously_verified_not_rerun=True,
    thresholds_unchanged=gates,source_points_compared_before_any_new_mask_drops=True,
    source_depth_is_estimated_not_ground_truth=True,new_geometry_fit_executed=False,
    original_confidence_arrays_changed=False,source_model_results_are_estimates=True,
    native_reference_contact_tolerance_m=.001,native_reference_gate_passed=None,
    carry_reference_accepted=False,training_approved=False,physical_task_accepted=False,
    rows=rows)
(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))

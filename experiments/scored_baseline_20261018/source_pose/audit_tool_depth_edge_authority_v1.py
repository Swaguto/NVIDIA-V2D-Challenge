"""Locate contradictory estimated tool-mask depths without changing any validity flag."""
from pathlib import Path
import datetime,hashlib,json,sys
S=Path(__file__).resolve().parents[1];sys.path.insert(0,str(S/'mediapipe-cpu-deps-v1'))
import cv2,numpy as np
cv2.setNumThreads(1);O=S/'e037-tool-depth-edge-authority-v1';O.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
depth_path=S/'fast-stereo-cpu-full-e037-v1/000050-depth.npz'
mask_path=S/'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/000050.png'
image_path=S/'e037-fresh-source-hand-request-v1/images/000050.png'
probe_path=S/'e037-visible-object-pose-probe-v1/unqualified-object-pose-proposals.npz'
with np.load(depth_path) as z:depth=z['depth_m'].copy();consistent=z['left_right_consistent'].copy()
mask=cv2.imread(str(mask_path),0)>0;valid=mask&consistent&np.isfinite(depth)&(depth>0)
inside_distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
median=float(np.median(depth[valid]))
rows=[]
with np.load(probe_path) as z:
 for name,pixels in [('train',z['train_pixels_C']),('heldout',z['heldout_pixels_C'])]:
  values=depth[pixels[:,1],pixels[:,0]];edge=inside_distance[pixels[:,1],pixels[:,0]]
  far=values>1.;offset=np.abs(values-median)>.05
  rows.append(dict(partition=name,count=len(values),estimated_depth_quantiles_m=np.quantile(values,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),over_1m_count=int(far.sum()),over_1m_source_pixels_C=pixels[far].tolist(),over_1m_inside_mask_boundary_distance_px=edge[far].tolist(),over_50mm_from_mask_valid_median_count=int(offset.sum()),over_50mm_from_mask_valid_median_inside_boundary_distance_quantiles_px=np.quantile(edge[offset],[0,.5,1]).tolist() if offset.any() else [],all_samples_retained=True))
source=cv2.imread(str(image_path));panel=source.copy();far=valid&(depth>1.);y,x=np.nonzero(far)
for px,py in zip(x,y):cv2.circle(panel,(int(px),int(py)),1,(0,0,255),-1)
contours,_=cv2.findContours(mask.astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(panel,contours,-1,(0,200,255),1)
cv2.imwrite(str(O/'source-mask-and-retained-over-1m-estimates.png'),panel[160:390,350:900])
np.savez_compressed(O/'diagnostic-pixel-cohorts.npz',over_1m_pixels_C=np.c_[x,y],over_1m_estimated_depth_m=depth[far],over_1m_inside_mask_distance_px=inside_distance[far])
report=dict(schema='v2d.tool-estimated-depth-edge-authority-audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),frame=50,source_hashes={str(p.relative_to(S)):sha(p) for p in [depth_path,mask_path,image_path,probe_path]},inferred_mask_pixels=int(mask.sum()),inferred_mask_consistent_depth_pixels=int(valid.sum()),median_estimated_mask_depth_m=median,all_mask_valid_depth_quantiles_m=np.quantile(depth[valid],[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist(),over_1m_valid_pixels=int(far.sum()),over_1m_inside_mask_distance_quantiles_px=np.quantile(inside_distance[far],[0,.5,.95,1]).tolist(),fixed_thresholds_are_descriptive_cohorts_not_rejection_rules=True,source_masks_are_inferred_not_ground_truth=True,left_right_consistency_is_not_semantic_foreground_truth=True,low_resolution_model_input_wh=[512,320],output_wh=[1280,800],interpolation_or_boundary_error_is_hypothesis_not_proven=True,rows=rows,no_samples_deleted_or_relabelled=True,source_validity_and_confidence_flags_unchanged=True,old_failed_candidates_preserved=True,new_training_steps=0,new_GPU_work=False,physical_task_accepted=False,required_native_reference_contact_gate_m=.001)
for rel,h in report['source_hashes'].items():assert sha(S/rel)==h,rel
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

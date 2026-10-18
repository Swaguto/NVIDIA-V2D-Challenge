"""Export the existing tracker's one-sided evidence as INFERRED, never direct.

Bidirectional agreement remains 1.5 px when both sides track a point. A sole
available side is retained, as in issue6.image_tracks; it is explicitly marked
one-sided and cannot pass the failed full-hand stereo gate by construction.
"""
from pathlib import Path
import hashlib,json
import cv2,numpy as np
S=Path(__file__).resolve().parents[1];I=S/'e037-source-hand-image-motion-v3';O=S/'e037-inferred-hand-image-evidence-v1';O.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();r=json.load(open(I/'report.json'));raw=json.load(open(S/'mediapipe-source-hand-candidates-full-v1/report.json'))
with np.load(I/'image-observations.npz') as z: a={k:z[k].copy() for k in z.files}
# Recreate byte-equivalent within-view de-dup ordering, no detector execution.
cs=[]
for row in raw['rows']:
 unique=[]
 for c in row['candidate_views'][0]['candidates']:
  p=np.array(c['landmarks_px'],float)
  if not any(np.median(np.linalg.norm(p-o,axis=1))<=5 for o in unique):unique.append(p)
 cs.append(unique)
out=np.full((151,21,2),np.nan);rows=[]
for f,row in enumerate(r['rows']):
 forward=a['forward_px'][f];backward=a['backward_px'][f];av=np.isfinite(forward).all(-1);bv=np.isfinite(backward).all(-1);both=av&bv;agree=both&(np.linalg.norm(forward-backward,axis=-1)<=1.5)
 fresh=[(k,s) for k,s in [(row['forward_kind'],row['forward_candidate']),(row['backward_kind'],row['backward_candidate'])] if k in [1,2] and s>=0]
 if fresh and len({s for k,s in fresh})==1:
  out[f]=cs[f][fresh[0][1]];kind='identity_associated_source_model_prediction';one_sided=len(fresh)==1
 else:
  out[f,av&~bv]=forward[av&~bv];out[f,bv&~av]=backward[bv&~av];out[f,agree]=(forward[agree]+backward[agree])/2;kind='checked_inferred_image_flow';one_sided=bool((av^bv).any())
  if np.isfinite(out[f]).all(-1).sum()<4:out[f]=np.nan;kind='unknown'
 rows.append(dict(frame=f,kind=kind,one_sided_identity_or_flow=one_sided,supported_joints=np.flatnonzero(np.isfinite(out[f]).all(-1)).tolist(),full_metric_hand_qualified=False))
np.savez_compressed(O/'image-observations.npz',frame_indices=a['frame_indices'],timestamps_s=a['timestamps_s'],landmarks_px=out,landmark_supported=np.isfinite(out).all(-1),original_hand_valid=a['original_hand_valid'],original_hand_directly_observed=a['original_hand_directly_observed'],original_hand_metric_valid=a['original_hand_metric_valid'])
for f in [25,35,40,45,50,55,60,75,99]:
 p=I/f'{f:06d}-source.jpg'
 if not p.exists():continue
 im=cv2.imread(str(p));cv2.rectangle(im,(0,0),(1280,50),(0,0,0),-1)
 for point in out[f]:
  if np.isfinite(point).all():cv2.circle(im,tuple(np.rint(point).astype(int)),4,(0,165,255),-1)
 cv2.putText(im,f'frame{f}: INFERRED {rows[f]["kind"]}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,165,255),2);cv2.imwrite(str(O/f'{f:06d}-inferred.jpg'),im)
rpt=dict(schema='v2d.inferred-hand-image-evidence.v1',producer_sha256=sha(Path(__file__)),input_report_sha256=sha(I/'report.json'),input_arrays_sha256=sha(I/'image-observations.npz'),raw_sha256=sha(S/'mediapipe-source-hand-candidates-full-v1/report.json'),fresh_model_estimates_are_not_ground_truth=True,one_sided_flow_supported_by_existing_method=True,original_source_flags_changed=False,failed_full_hand_stereo_gate_retained=True,training_approved=False,rows=rows,supported_frames=[f for f in range(151) if np.isfinite(out[f]).all(-1).sum()>=4],missing_carry_frames=[f for f in range(25,100) if np.isfinite(out[f]).all(-1).sum()<4],observations_sha256=sha(O/'image-observations.npz'))
(O/'report.json').write_text(json.dumps(rpt,indent=2)+'\n');print(json.dumps({k:v for k,v in rpt.items() if k!='rows'},indent=2))

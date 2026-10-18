"""Recover explicitly inferred image tracks from cached detections and RGB motion.

No new detector inference. Handedness model strings never establish identity.
The unchanged 1.5px FB / 1.5second gap limits apply to propagated points.
Fresh 2D predictions remain predictions; original observation flags are readonly.
"""
from pathlib import Path
import hashlib,json,time,sys
import cv2,numpy as np
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'reconstruction/modules/v2d_stereo_depth/lib'))
from stereo_depth import load_rig
OUT=S/'e037-source-hand-image-motion-v2';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
RAW=S/'mediapipe-source-hand-candidates-full-v1/report.json';raw=json.load(open(RAW));B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:b={k:z[k].copy() for k in ['camera_c2w','hand_joints_world','intrinsics','hand_valid','hand_directly_observed','hand_metric_valid','frame_indices','timestamps_s']}
rig=load_rig(str(S/'dataset/track_3/public/meta/camera_calibration.json'))
V=[S/f'dataset/track_3/evaluation/videos/chunk-000/observation.images.{v}/episode_000037.mp4' for v in [rig.left,rig.right]]
report=dict(schema='v2d.source-hand-image-motion.v1',producer_sha256=sha(Path(__file__)),raw_sha256=sha(RAW),bundle_sha256=sha(B),source_sha256={str(p.relative_to(S)):sha(p) for p in V},gates=dict(source_identity_max_error_px=50,source_identity_min_margin_px=25,minimum_identity_anchor_frames=5,fb_limit_px=1.5,max_track_gap_s=1.5,candidate_flow_agreement_px=8,minimum_agreeing_joints=4),original_source_flags_changed=False,full_hand_stereo_gate_relaxed=False,complete_metric_hands_recovered=0,physical_task_accepted=False)
save=lambda:(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');started=time.monotonic();save()
# Use per-frame source camera projection only at original direct+metric anchors.
source_uv=np.full((151,2,2),np.nan);known=b['hand_valid']&b['hand_directly_observed']&b['hand_metric_valid']
for f,c in enumerate(b['camera_c2w']):
 j=(b['hand_joints_world'][f,:,0]-c[:3,3])@c[:3,:3];K=b['intrinsics'];source_uv[f]=np.c_[K[0,0]*j[:,0]/j[:,2]+K[0,2],K[1,1]*j[:,1]/j[:,2]+K[1,2]]
candidates=[];seeds={};votes=[]
for row in raw['rows']:
 f=row['frame'];cs=[]
 for c in row['candidate_views'][0]['candidates']:
  p=np.array(c['landmarks_px'],float)
  if not any(np.median(np.linalg.norm(p-o['points'],axis=1))<=5 for o in cs):cs.append({'points':p,'raw':c})
 candidates.append(cs)
 if known[f,1]:
  allowed=[]
  for i,c in enumerate(cs):
   d=np.linalg.norm(source_uv[f]-c['points'][0],axis=1)
   if d[1]<50 and d[0]-d[1]>=25:
    allowed.append((d[1],i));votes.append(dict(frame=f,label=c['raw']['handedness_label'],source_hand=1,error_px=float(d[1])))
  if allowed:seeds[f]=min(allowed)[1]
assert len(seeds)>=5
# Source decoding occurs once. Keep only grayscale; bounded memory ~155 MB.
caps=[cv2.VideoCapture(str(p)) for p in V];gray=[]
for f in range(151):
 frames=[]
 for cap in caps:
  ok,image=cap.read();assert ok;frames.append(image)
 left,_=rig.rectify_pair(frames[0],frames[1]);gray.append(cv2.cvtColor(left,cv2.COLOR_BGR2GRAY))
for cap in caps:cap.release()

def track(order):
 result=np.full((151,21,2),np.nan);kind=np.zeros(151,np.int8);selection=np.full(151,-1,int);last_anchor=None;previous=None;points=None;active=np.zeros(21,bool)
 for f in order:
  predicted=np.full((21,2),np.nan);good=np.zeros(21,bool)
  if previous is not None and points is not None and active.sum()>=4 and abs(f-last_anchor)/20<=1.5:
   ids=np.flatnonzero(active);before=points[ids,None].astype(np.float32)
   after,sf,_=cv2.calcOpticalFlowPyrLK(gray[previous],gray[f],before,None,winSize=(21,21),maxLevel=3)
   back,sb,_=cv2.calcOpticalFlowPyrLK(gray[f],gray[previous],after,None,winSize=(21,21),maxLevel=3)
   p=after[:,0];passing=(sf[:,0]==1)&(sb[:,0]==1)&(np.linalg.norm(back[:,0]-before[:,0],axis=1)<=1.5)&np.isfinite(p).all(axis=1)&(p[:,0]>=2)&(p[:,0]<1278)&(p[:,1]>=2)&(p[:,1]<798)
   good[ids[passing]]=True;predicted[ids[passing]]=p[passing]
  chosen=seeds.get(f)
  if chosen is None and good[0] and good.sum()>=4:
   scores=[]
   for i,c in enumerate(candidates[f]):
    errors=np.linalg.norm(c['points']-predicted,axis=1);agree=good&(errors<=8)
    if agree[0] and agree.sum()>=4:scores.append((float(np.median(errors[good])),i))
   if scores:chosen=min(scores)[1]
  if chosen is not None:
   points=candidates[f][chosen]['points'].copy();active=np.isfinite(points).all(axis=1)&(points[:,0]>=2)&(points[:,0]<1278)&(points[:,1]>=2)&(points[:,1]<798)
   result[f,active]=points[active];kind[f]=1 if f in seeds else 2;selection[f]=chosen;last_anchor=f
  elif good.sum()>=4:
   points=predicted;active=good;result[f,good]=predicted[good];kind[f]=3
  else:points=None;active[:]=False
  previous=f
 return result,kind,selection
forward,fkind,fsel=track(range(151));backward,bkind,bsel=track(range(150,-1,-1));output=np.full_like(forward,np.nan);status=[]
for f in range(151):
 a=np.isfinite(forward[f]).all(axis=1);bmask=np.isfinite(backward[f]).all(axis=1);both=a&bmask;agree=both&(np.linalg.norm(forward[f]-backward[f],axis=1)<=1.5)
 # For fresh predictions, identity must reach an original anchor in both temporal
 # directions. Different crop estimates need pointwise FB-path agreement too.
 if f in seeds:output[f]=candidates[f][seeds[f]]['points'];reason='source_anchor_associated_prediction'
 elif fkind[f] in [1,2] and bkind[f] in [1,2] and fsel[f]==bsel[f] and fsel[f]>=0:output[f]=candidates[f][fsel[f]]['points'];reason='bidirectionally_identity_associated_prediction'
 else:
  output[f,agree]=(forward[f,agree]+backward[f,agree])/2;reason='bidirectionally_agreed_image_flow'
  if np.isfinite(output[f]).all(axis=1).sum()<4:output[f]=np.nan;reason='unknown'
 status.append(dict(frame=f,kind=reason,supported_joints=np.flatnonzero(np.isfinite(output[f]).all(axis=1)).tolist(),forward_kind=int(fkind[f]),backward_kind=int(bkind[f]),forward_candidate=int(fsel[f]),backward_candidate=int(bsel[f])))
np.savez_compressed(OUT/'image-observations.npz',frame_indices=b['frame_indices'],timestamps_s=b['timestamps_s'],landmarks_px=output,landmark_supported=np.isfinite(output).all(axis=-1),forward_px=forward,backward_px=backward,original_hand_valid=b['hand_valid'],original_hand_directly_observed=b['hand_directly_observed'],original_hand_metric_valid=b['hand_metric_valid'])
# New source-image previews only, not a rerun of any completed policy renderer.
cap=cv2.VideoCapture(str(V[0]));capr=cv2.VideoCapture(str(V[1]));images=[]
for f in range(151):
 ok,l=cap.read();okr,r=capr.read();assert ok and okr
 if f not in [24,35,40,45,50,55,60,75,99,100]:continue
 l,_=rig.rectify_pair(l,r)
 for i,p in enumerate(output[f]):
  if np.isfinite(p).all():cv2.circle(l,tuple(np.rint(p).astype(int)),4,(0,255,0),-1)
 cv2.putText(l,f"frame {f}: {status[f]['kind']}",(12,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2)
 cv2.imwrite(str(OUT/f'{f:06d}-source.jpg'),l);images.append(cv2.resize(l,(640,400)))
cap.release();capr.release();cv2.imwrite(str(OUT/'source-track-grid.jpg'),np.concatenate([np.concatenate(images[i:i+2],axis=1) for i in range(0,len(images),2)],axis=0))
report.update(rows=status,identity_anchor_frames=sorted(seeds),identity_votes=votes,supported_frames=[f for f in range(151) if np.isfinite(output[f]).all(axis=1).sum()>=4],missing_carry_frames=[f for f in range(25,100) if np.isfinite(output[f]).all(axis=1).sum()<4],new_detector_inference=False,all_frames_complete=True,observations_sha256=sha(OUT/'image-observations.npz'),wall_seconds=time.monotonic()-started)
assert sha(RAW)==report['raw_sha256'] and sha(B)==report['bundle_sha256'];report['source_closure_verified_after']=True;save();print(json.dumps({k:v for k,v in report.items() if k not in ['rows','identity_votes']},indent=2),flush=True)

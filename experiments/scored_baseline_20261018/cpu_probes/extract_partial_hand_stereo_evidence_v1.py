"""Separate per-joint evidence; the failed full-hand gate remains unchanged.

A supported wrist/knuckle tuple is not a complete source hand, grasp, or training
reference. Unknown points stay NaN and original bundle flags are read-only.
"""
import hashlib,json,sys,time
from pathlib import Path
import cv2,numpy as np
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'reconstruction/modules/v2d_stereo_depth/lib'))
from stereo_depth import load_rig
OUT=S/'partial-hand-stereo-evidence-v1';OUT.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
RAW=S/'mediapipe-source-hand-candidates-full-v1/report.json';raw=json.load(open(RAW));assert raw['all_selected_frames_complete'] and raw['source_closure_verified_after']
OLD=S/'mediapipe-stereo-hand-candidates-v1/report.json';old=json.load(open(OLD));assert old['qualified_candidate_pairs']==0
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:b={k:z[k].copy() for k in ['camera_c2w','hand_joints_world','intrinsics','hand_valid','hand_directly_observed','hand_metric_valid','frame_indices','timestamps_s']}
rig=load_rig(str(S/'dataset/track_3/public/meta/camera_calibration.json'));D=S/'fast-stereo-cpu-full-e037-v1';depthlock=json.load(open(D/'depth-report.json'))
gates={'per_joint_epipolar_px':1.,'independent_estimated_depth_error_m':.01,'positive_disparity_px':3.,'depth_range_m':[.15,1.4],'within_view_dedup_median_px':5.,'palm_triangle_min_area_m2':1e-5,'wrist_knuckle_distance_range_m':[.025,.14],'maximum_passing_wrist_disagreement_m':.01,'source_identity_maximum_pixel_error':50.,'source_identity_minimum_margin_px':25.}
report={'schema':'v2d.partial-stereo-hand-evidence.v1','producer_sha256':sha(Path(__file__)),'raw_report_sha256':sha(RAW),'retained_failed_full_hand_report_sha256':sha(OLD),'bundle_sha256':sha(B),'gates':gates,'original_source_flags_changed':False,'full_hand_gate_relaxed':False,'complete_hands_recovered':0,'partial_points_are_training_targets':False,'physical_task_accepted':False,'rows':[]};save=lambda:(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');started=time.monotonic();save()
allpairs=[];identity_votes=[];checked_depths={}
try:
 for row in raw['rows']:
  f=row['frame'];views=[]
  for v in row['candidate_views']:
   unique=[]
   for c in sorted(v['candidates'],key=lambda c:-c['handedness_probability']):
    uv=np.array(c['landmarks_px'])
    if any(np.median(np.linalg.norm(uv-np.array(o['landmarks_px']),axis=1))<=5 for o in unique):continue
    unique.append(c)
   views.append(unique)
  rec={'frame':f,'partial_pairs':[]}
  if all(views):
   dp=D/f'{f:06d}-depth.npz';assert sha(dp)==depthlock['complete'][f]['sha256'];checked_depths[str(dp.relative_to(S))]=sha(dp)
   with np.load(dp) as z:depth=z['depth_m'].copy();valid=z['left_right_consistent'].copy()
   for li,lhs in enumerate(views[0]):
    for ri,rhs in enumerate(views[1]):
     if lhs['handedness_label']!=rhs['handedness_label']:continue
     l=np.array(lhs['landmarks_px']);r=np.array(rhs['landmarks_px']);epi=abs(l[:,1]-r[:,1]);disp=l[:,0]-r[:,0]-(rig.P1[0,2]-rig.P2[0,2]);h=cv2.triangulatePoints(rig.P1,rig.P2,l.T,r.T);points=(h[:3]/h[3]).T
     inside=(l[:,0]>=0)&(l[:,0]<rig.size[0])&(l[:,1]>=0)&(l[:,1]<rig.size[1])&(r[:,0]>=0)&(r[:,0]<rig.size[0])&(r[:,1]>=0)&(r[:,1]<rig.size[1])
     initial=inside&(epi<=1)&(disp>=3)&np.isfinite(points).all(axis=1)&(points[:,2]>=.15)&(points[:,2]<=1.4)
     ix=np.clip(np.rint(l[:,0]).astype(int),0,rig.size[0]-1);iy=np.clip(np.rint(l[:,1]).astype(int),0,rig.size[1]-1);error=abs(points[:,2]-depth[iy,ix]);supported=initial&valid[iy,ix]&np.isfinite(depth[iy,ix])&(error<=.01)
     palm=supported[[0,5,9]].all();area=float(np.linalg.norm(np.cross(points[5]-points[0],points[9]-points[0]))/2) if np.isfinite(points[[0,5,9]]).all() else 0.
     bones=np.linalg.norm(points[[5,9]]-points[0],axis=1)
     palm=bool(palm and area>=1e-5 and (bones>=.025).all() and (bones<=.14).all())
     entry={'left':li,'right':ri,'model_label':lhs['handedness_label'],'supported_joints':np.flatnonzero(supported).tolist(),'palm_tuple_supported':palm,'wrist_epipolar_px':float(epi[0]),'wrist_depth_error_m':float(error[0]) if np.isfinite(error[0]) else None,'palm_area_m2':area}
     rec['partial_pairs'].append(entry)
     if supported.any():allpairs.append({'frame':f,'label':lhs['handedness_label'],'points':points,'supported':supported,'left_uv':l,'right_uv':r,'palm':palm,'left_index':li,'right_index':ri,'cost':float(np.mean(epi[supported])+100*np.mean(error[supported]))})
     # Read-only identity controls on original supported source anchors.
     cam=b['camera_c2w'][f];known=b['hand_valid'][f]&b['hand_directly_observed'][f]&b['hand_metric_valid'][f]
     if palm and known.any():
      world=b['hand_joints_world'][f,:,0];camera=np.einsum('ni,ij->nj',world-cam[:3,3],cam[:3,:3],optimize=False);K=b['intrinsics'];uv=np.c_[K[0,0]*camera[:,0]/camera[:,2]+K[0,2],K[1,1]*camera[:,1]/camera[:,2]+K[1,2]]
      distances=np.linalg.norm(uv-l[0],axis=1);order=np.argsort(distances);chosen=int(order[0])
      if known[chosen] and distances[chosen]<50 and distances[order[1]]-distances[chosen]>=25:identity_votes.append({'frame':f,'label':lhs['handedness_label'],'source_hand':chosen,'pixel_error':float(distances[chosen])})
  report['rows'].append(rec)
 # Identity must be supported by existing observed source anchors, never guessed
 # from a label's English spelling or from proximity to the lifted spoon.
 mapping={};votes_by_label={}
 for label in sorted(set(p['label'] for p in allpairs)):
  votes=[v for v in identity_votes if v['label']==label];byframe={v['frame']:v['source_hand'] for v in votes};counts=[sum(v==i for v in byframe.values()) for i in [0,1]];winner=int(np.argmax(counts))
  votes_by_label[label]={'unique_anchor_frames':len(byframe),'counts':counts}
  if len(byframe)>=5 and counts[winner]/len(byframe)>=.9:mapping[label]=winner
 outputs=[];ambiguous=[]
 for f in range(151):
  for label,si in mapping.items():
   pairs=[p for p in allpairs if p['frame']==f and p['label']==label and p['palm']]
   if not pairs:continue
   wrists=np.stack([p['points'][0] for p in pairs]);spread=float(np.linalg.norm(wrists[:,None]-wrists[None,:],axis=-1).max())
   if spread>.01:ambiguous.append({'frame':f,'label':label,'wrist_spread_m':spread});continue
   chosen=min(pairs,key=lambda p:p['cost']);chosen['source_hand']=si;outputs.append(chosen)
 np.savez_compressed(OUT/'partial-source-landmarks.npz',frame_indices=np.array([p['frame'] for p in outputs]),source_hand=np.array([p['source_hand'] for p in outputs]),landmarks_camera_m=np.array([np.where(p['supported'][:,None],p['points'],np.nan) for p in outputs]).reshape(-1,21,3),landmark_supported=np.array([p['supported'] for p in outputs]).reshape(-1,21),left_uv_px=np.array([p['left_uv'] for p in outputs]).reshape(-1,21,2),right_uv_px=np.array([p['right_uv'] for p in outputs]).reshape(-1,21,2))
 report.update(identity_votes=identity_votes,identity_vote_summary=votes_by_label,model_label_to_source_hand=mapping,partial_point_pairs=len(allpairs),palm_tuple_pairs=sum(p['palm'] for p in allpairs),unambiguous_identity_supported_palm_tuples=len(outputs),right_palm_frames=[p['frame'] for p in outputs if p['source_hand']==1],right_carry_palm_frames=[p['frame'] for p in outputs if p['source_hand']==1 and 25<=p['frame']<=99],ambiguous_wrists=ambiguous,candidate_sha256=sha(OUT/'partial-source-landmarks.npz'),all_frames_complete=len(report['rows'])==151)
 for name,h in checked_depths.items():assert sha(S/name)==h
 assert sha(RAW)==report['raw_report_sha256'] and sha(B)==report['bundle_sha256'];report['inputs_verified_after']=True
except BaseException as error:report['error']=repr(error);raise
finally:
 report['wall_seconds']=time.monotonic()-started;save();print(json.dumps({k:v for k,v in report.items() if k not in ['rows','identity_votes']},indent=2),flush=True)

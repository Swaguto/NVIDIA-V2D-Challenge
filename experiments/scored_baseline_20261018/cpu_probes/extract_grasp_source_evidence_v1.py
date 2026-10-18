"""New source-only contact evidence; no policy renderer or completed job rerun."""
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');OUT=S/'e037-grasp-source-evidence-v1';OUT.mkdir()
sys.path.insert(0,str(R/'reconstruction/modules/v2d_stereo_depth/lib'))
from stereo_depth import load_rig
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rig=load_rig(str(S/'dataset/track_3/public/meta/camera_calibration.json'))
videos=[S/f'dataset/track_3/evaluation/videos/chunk-000/observation.images.{cam}/episode_000037.mp4' for cam in [rig.left,rig.right]]
report=json.load(open(S/'fast-stereo-cpu-full-e037-v1/depth-report.json'))
for p in videos:assert sha(p)==report['source_files_sha256'][str(p.relative_to(S))]
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B,allow_pickle=False) as z:flags={k:z[k].copy() for k in ['hand_valid','hand_directly_observed','hand_metric_valid']}
selected=[0,25,50,67,68,75,100,125,150];caps=[cv2.VideoCapture(str(p)) for p in videos];panels=[]
try:
 for frame in range(151):
  raw=[]
  for cap in caps:
   ok,img=cap.read();assert ok;raw.append(img)
  if frame not in selected:continue
  image,_=rig.rectify_pair(*raw)
  cv2.imwrite(str(OUT/f'frame-{frame:03d}.jpg'),image)
  display=cv2.resize(image,(640,400));cv2.rectangle(display,(0,0),(640,44),(0,0,0),-1)
  cv2.putText(display,f'e037 f{frame} t={frame/20:.2f}s',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
  cv2.putText(display,f'R source direct={int(flags["hand_directly_observed"][frame,1])} metric={int(flags["hand_metric_valid"][frame,1])}',(8,37),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1)
  panels.append(display)
 cv2.imwrite(str(OUT/'source-clock-grid.jpg'),np.concatenate([np.concatenate(panels[a:a+3],axis=1) for a in [0,3,6]],axis=0))
 for p in videos:assert sha(p)==report['source_files_sha256'][str(p.relative_to(S))]
 (OUT/'receipt.json').write_text(json.dumps({'schema':'v2d.private-grasp-source-evidence.v1','producer_sha256':sha(Path(__file__)),'source_frames':selected,'selection_reason':'Uniform source-clock phases plus independently detected67/68transition; no reference labels or performance selection.','bundle_sha256':sha(B),'source_hashes':{str(p.relative_to(S)):sha(p) for p in videos},'outputs':{p.name:sha(p) for p in OUT.glob('*.jpg')},'source_confidence_changed':False,'policy_renderer_rerun':False,'private':True},indent=2)+'\n')
finally:
 for cap in caps:cap.release()
print('Private source evidence complete.')

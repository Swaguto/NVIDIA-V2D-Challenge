"""Positive-control plus fixed crop/rotation detector diagnosis; no metric repair."""
import hashlib,json,sys,urllib.request,time
from pathlib import Path
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(S/'mediapipe-cpu-deps-v1'))
import cv2,numpy as np,mediapipe as mp
sys.path.insert(0,str(R/'reconstruction/modules/v2d_stereo_depth/lib'))
from stereo_depth import load_rig
OUT=S/'mediapipe-source-hand-candidates-full-v1';OUT.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
url='https://developers.google.com/static/mediapipe/images/solutions/gesture-recognizer.png'
cached_root=S/'mediapipe-source-hand-probe-v2'
control=(cached_root/'public-documentation-control.png').read_bytes()
cached_report=json.loads((cached_root/'report.json').read_text());assert hashlib.sha256(control).hexdigest()==cached_report['public_control_sha256']
assert cached_report['all_selected_frames_complete'] and cached_report['source_closure_verified_after']
cached_rows={r['frame']:r for r in cached_report['rows']}
model=S/'mediapipe-cpu-model-v1/hand_landmarker.task';lock=json.load(open(model.parent/'model-lock.json'));assert sha(model)==lock['sha256']
opts=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model),delegate=mp.tasks.BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
rig=load_rig(str(S/'dataset/track_3/public/meta/camera_calibration.json'));videos=[S/f'dataset/track_3/evaluation/videos/chunk-000/observation.images.{c}/episode_000037.mp4' for c in [rig.left,rig.right]]
depth=json.load(open(S/'fast-stereo-cpu-full-e037-v1/depth-report.json'))
for p in videos:assert sha(p)==depth['source_files_sha256'][str(p.relative_to(S))]
report={'schema':'v2d.mediapipe-full-source-candidates.v1','producer_sha256':sha(Path(__file__)),'model_sha256':sha(model),'public_control_url':url,'public_control_sha256':hashlib.sha256(control).hexdigest(),'thresholds_unchanged':.5,'metric_hand_reconstruction_claimed':False,'original_flags_changed':False,'source_footage_uploaded':False,'physical_task_accepted':False,'rows':[]};save=lambda:(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');started=time.monotonic();save();caps=[cv2.VideoCapture(str(p)) for p in videos]
def detect(detector,img):return detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))))
try:
 with mp.tasks.vision.HandLandmarker.create_from_options(opts) as detector:
  public=cv2.imdecode(np.frombuffer(control,np.uint8),cv2.IMREAD_COLOR);assert public is not None
  report['public_control_detected_hands']=cached_report['public_control_detected_hands'];report['public_control_inference_reused']=True;report['five_prior_frames_exactly_reused']=[];report['cached_report_sha256']=sha(cached_root/'report.json');report['no_outputs_are_metric_observations']=True;save()
  for frame in range(151):
   if time.monotonic()-started>300:raise TimeoutError('Bounded source preprocessing probe')
   raw=[]
   for cap in caps:
    ok,im=cap.read();assert ok;raw.append(im)
   if frame in cached_rows:
    report['rows'].append(cached_rows[frame]);report['five_prior_frames_exactly_reused'].append(frame);save();continue
   row={'frame':frame,'candidate_views':[]}
   for vi,source in enumerate(rig.rectify_pair(*raw)):
    height,width=source.shape[:2];candidates=[]
    for label,box in [('full',(0,0,width,height)),('left_half',(0,0,width//2,height)),('right_half',(width//2,0,width,height))]:
     x0,y0,x1,y1=box;crop=source[y0:y1,x0:x1];h,w=crop.shape[:2]
     for k in range(4):
      image=np.ascontiguousarray(np.rot90(crop,k));result=detect(detector,image);ih,iw=image.shape[:2]
      for landmarks,handedness in zip(result.hand_landmarks,result.handedness,strict=True):
       xy=np.array([[p.x*iw,p.y*ih] for p in landmarks]);u,v=xy.T
       if k==0:points=xy
       elif k==1:points=np.c_[w-1-v,u]
       elif k==2:points=np.c_[w-1-u,h-1-v]
       else:points=np.c_[v,h-1-u]
       points+=np.array([x0,y0]);assert np.isfinite(points).all()
       candidates.append({'crop':label,'rotation_quarters_ccw':k,'landmarks_px':points.tolist(),'handedness_label':handedness[0].category_name,'handedness_probability':handedness[0].score,'identity_and_stereo_metric_qualification':False})
    row['candidate_views'].append({'view':vi,'candidates':candidates})

   report['rows'].append(row);save();print(json.dumps({'frame':frame,'raw_candidates_per_view':[len(v['candidates']) for v in row['candidate_views']]}),flush=True)
 for p in videos:assert sha(p)==depth['source_files_sha256'][str(p.relative_to(S))]
 report['source_files_sha256']={str(p.relative_to(S)):sha(p) for p in videos};report['source_closure_verified_after']=True;report['all_selected_frames_complete']=len(report['rows'])==151
except BaseException as error:report['error']=repr(error);raise
finally:
 for cap in caps:cap.release()
 report['wall_seconds']=time.monotonic()-started;save()

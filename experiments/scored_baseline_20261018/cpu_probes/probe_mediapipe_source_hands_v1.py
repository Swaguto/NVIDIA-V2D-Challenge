"""Independent CPU image detections on fixed source frames; no observation repair.

The model's hand-relative world coordinates are not camera metric depth and
are not used. Missing detections remain empty; original bundle flags stay intact.
"""
import datetime,hashlib,importlib.metadata,json,sys,time
from pathlib import Path
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(S/'mediapipe-cpu-deps-v1'))
import cv2,numpy as np,mediapipe as mp
sys.path.insert(0,str(R/'reconstruction/modules/v2d_stereo_depth/lib'))
from stereo_depth import load_rig
OUT=S/'mediapipe-source-hand-probe-v1';OUT.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
M=S/'mediapipe-cpu-model-v1';lock=json.load(open(M/'model-lock.json'));model=M/'hand_landmarker.task';assert sha(model)==lock['sha256']
rig=load_rig(str(S/'dataset/track_3/public/meta/camera_calibration.json'))
videos=[S/f'dataset/track_3/evaluation/videos/chunk-000/observation.images.{cam}/episode_000037.mp4' for cam in [rig.left,rig.right]]
depth=json.load(open(S/'fast-stereo-cpu-full-e037-v1/depth-report.json'))
for p in videos:assert sha(p)==depth['source_files_sha256'][str(p.relative_to(S))]
selected=[0,25,35,50,67,68,70,75,100,125,150]
report={'schema':'v2d.mediapipe-independent-source-hand-probe.v1','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'producer_sha256':sha(Path(__file__)),'model_sha256':sha(model),'mediapipe_version':mp.__version__,'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(S/'mediapipe-cpu-deps-v1')])},'device':'CPU','running_mode':'IMAGE','num_hands':2,'detection_presence_tracking_thresholds':.5,'original_source_flags_changed':False,'source_footage_uploaded':False,'metric_camera_landmarks_certified':False,'model_world_coordinates_used_as_metric_camera_depth':False,'contact_labels_created':False,'physical_task_accepted':False,'source_frames':selected,'rows':[]}
save=lambda:(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');started=time.monotonic();save()
caps=[cv2.VideoCapture(str(p)) for p in videos]
options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model),delegate=mp.tasks.BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
try:
 with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
  for frame in range(151):
   if time.monotonic()-started>120:raise TimeoutError('Bounded CPU source-hand probe')
   raw=[]
   for cap in caps:
    ok,img=cap.read();assert ok;raw.append(img)
   if frame not in selected:continue
   rectified=rig.rectify_pair(*raw);row={'frame':frame,'views':[]};panels=[]
   for vi,source in enumerate(rectified):
    result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(source,cv2.COLOR_BGR2RGB)))
    hands=[];display=source.copy();height,width=source.shape[:2]
    for hi,(landmarks,handedness) in enumerate(zip(result.hand_landmarks,result.handedness,strict=True)):
     uv=np.array([[p.x*width,p.y*height] for p in landmarks]);assert uv.shape==(21,2) and np.isfinite(uv).all()
     hands.append({'handedness_model_label':handedness[0].category_name,'handedness_probability':handedness[0].score,'landmarks_px':uv.tolist(),'anatomical_identity_verified':False})
     for i,(x,y) in enumerate(uv):
      cv2.circle(display,(int(x),int(y)),4,(0,255,0) if hi==0 else (0,165,255),-1)
      if i in [0,4,8,12]:cv2.putText(display,str(i),(int(x)+4,int(y)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    cv2.rectangle(display,(0,0),(width,32),(0,0,0),-1);cv2.putText(display,f'e037 f{frame} view{vi} independent CPU detections={len(hands)}',(8,23),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
    row['views'].append({'view_index':vi,'hands':hands});panels.append(cv2.resize(display,(640,400)))
   cv2.imwrite(str(OUT/f'frame-{frame:03d}.jpg'),np.concatenate(panels,axis=1));report['rows'].append(row);save();print(json.dumps({'frame':frame,'detected_hands_per_view':[len(v['hands']) for v in row['views']]}),flush=True)
 for p in videos:assert sha(p)==depth['source_files_sha256'][str(p.relative_to(S))]
 report['source_closure_verified_after']=True;report['all_selected_frames_complete']=len(report['rows'])==len(selected)
except BaseException as error:report['error']=repr(error);raise
finally:
 for cap in caps:cap.release()
 report['wall_seconds']=time.monotonic()-started;save()

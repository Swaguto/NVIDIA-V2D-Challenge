"""One new RGB-camera detector probe at the unchanged confidence; no metric claim."""
from pathlib import Path
import hashlib,json,sys,time
S=Path(__file__).resolve().parents[2];sys.path.insert(0,str(S/'mediapipe-cpu-deps-v1'))
import cv2,numpy as np,mediapipe as mp
O=S/'e037-RGB-hand-candidate-probe-v1';O.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=S/'e037-RGB-source-camera-probe-v1/camera-a-frame-000050.png';source_receipt=json.loads((source.parent/'report.json').read_text());assert sha(source)==source_receipt['image_sha256']
model=S/'mediapipe-cpu-model-v1/hand_landmarker.task';lock=json.loads((model.parent/'model-lock.json').read_text());assert sha(model)==lock['sha256']
image=cv2.imread(str(source));assert image is not None;h,w=image.shape[:2]
options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model),delegate=mp.tasks.BaseOptions.Delegate.CPU),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
report=dict(schema='v2d.source-RGB-hand-candidates.v1',frame=50,camera='ego_cam_a',producer_sha256=sha(Path(__file__)),image_sha256=sha(source),source_RGB_probe_sha256=sha(source.parent/'report.json'),model_sha256=sha(model),thresholds_unchanged=.5,candidates=[],handedness_is_not_identity=True,model_world_landmarks_are_hand_centered_not_absolute_camera_depth=True,metric_hand_reconstruction_claimed=False,original_flags_changed=False,physical_task_accepted=False,registered_to_stereo_camera=False)
save=lambda:(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');started=time.monotonic();save()
with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
 for label,(x0,y0,x1,y1) in [('full',(0,0,w,h)),('left_half',(0,0,w//2,h)),('right_half',(w//2,0,w,h))]:
  crop=image[y0:y1,x0:x1];ch,cw=crop.shape[:2];result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))))
  for landmarks,handedness in zip(result.hand_landmarks,result.handedness,strict=True):
   xy=np.array([[p.x*cw+x0,p.y*ch+y0] for p in landmarks]);assert np.isfinite(xy).all()
   report['candidates'].append(dict(crop=label,landmarks_px=xy.tolist(),handedness_label=handedness[0].category_name,handedness_probability=handedness[0].score,identity_and_metric_qualification=False))
report.update(complete=True,wall_seconds=time.monotonic()-started);save();print(json.dumps({'candidate_count':len(report['candidates']),'candidates':[{k:v for k,v in c.items() if k!='landmarks_px'} for c in report['candidates']]},indent=2))

"""Inspect one unused source RGB frame and encoded clocks; no vision inference."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
from PIL import Image
S=Path(__file__).resolve().parents[2];O=S/'e037-RGB-source-camera-probe-v1';O.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=S/'dataset/track_3/evaluation/videos/chunk-000';videos={k:base/f'observation.images.ego_cam_{k}/episode_000037.mp4' for k in ('a','c')}
streams={}
for k,p in videos.items():
    args=['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_frames:frame=best_effort_timestamp_time','-of','json',str(p)]
    streams[k]=json.loads(subprocess.check_output(args,text=True))
target=O/'camera-a-frame-000050.png';subprocess.run(['ffmpeg','-nostdin','-v','error','-noautorotate','-i',str(videos['a']),'-vf',"select=eq(n\\,50)",'-frames:v','1','-threads','1',str(target)],check=True,timeout=30)
im=np.asarray(Image.open(target).convert('RGB'));timestamps={k:[float(f['best_effort_timestamp_time']) for f in v['frames']] for k,v in streams.items()};calibration=S/'dataset/track_3/public/meta/camera_calibration.json'
report=dict(schema='v2d.source-RGB-camera-one-frame-probe.v1',producer_sha256=sha(Path(__file__)),source_video_sha256={k:sha(v) for k,v in videos.items()},calibration_sha256=sha(calibration),streams={k:v['streams'] for k,v in streams.items()},encoded_frame_clocks_equal=timestamps['a']==timestamps['c'],encoded_frame_50_time_s={k:t[50] for k,t in timestamps.items()},image_path=target.name,image_sha256=sha(target),frame=50,rgb_channel_mean_absolute_difference=float(abs(im[:,:,0].astype(float)-im[:,:,1]).mean()),image_shape=list(im.shape),new_model_inference=False,geometry_or_source_flags_modified=False,physical_sensor_exposure_synchrony_verified=False,registered_to_stereo_camera=False,hidden_reference_labels_read=False,physical_task_accepted=False)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

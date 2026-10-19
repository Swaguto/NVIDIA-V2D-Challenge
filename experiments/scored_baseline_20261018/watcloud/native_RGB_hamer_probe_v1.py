"""One native calibrated RGB crop, HaMeR only, no download or physics/training."""
from pathlib import Path
import hashlib,json,sys,time
sys.path.insert(0,'/workspace/pr32')
import cv2,numpy as np,torch
from v2d.hamer.lib._hamer import get_model,run_hamer
I=Path('/probe-input');O=Path('/outputs/native-RGB');O.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=I/'input.json';data=json.loads(p.read_text());im=I/data['image'];assert sha(im)==data['image_sha256']
image=cv2.cvtColor(cv2.imread(str(im)),cv2.COLOR_BGR2RGB);assert image.shape[:2]==(data['camera']['height'],data['camera']['width'])
assert torch.cuda.is_available();start=time.monotonic();model,cfg=get_model('/workspace/hamer-weights');c=data['crop'];out=run_hamer(model,cfg,image,c['cx'],c['cy'],c['size'],is_right=True)
for k,v in out.items():assert np.isfinite(np.asarray(v)).all(),k
np.savez_compressed(O/'prediction.npz',**out)
receipt=dict(schema='v2d.native-RGB-HaMeR-probe.v1',producer_sha256=sha(Path(__file__)),input_sha256=sha(p),image_sha256=sha(im),checkpoint_sha256=sha(Path('/workspace/hamer-weights/_DATA/hamer_ckpts/checkpoints/hamer.ckpt')),prediction_sha256=sha(O/'prediction.npz'),torch=torch.__version__,device=torch.cuda.get_device_name(0),native_RGB_camera_used=True,SAM2_inference=False,calibrated_camera_to_stereo_conversion_not_yet_applied=True,model_subject_betas_not_accepted=True,original_flags_changed=False,metric_hand_qualified=False,reference_accepted=False,new_training_steps=0,physical_task_accepted=False,wall_seconds=time.monotonic()-start)
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

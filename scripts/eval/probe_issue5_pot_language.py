"""Predetermined language controls; no GT, thresholds or default behavior changes."""
from pathlib import Path
import hashlib
import json
import os
import sys

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--models',type=Path,required=True)
parser.add_argument('--baseline',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
REPO=Path(__file__).resolve().parents[2]
DATA, OUT, MODELS=args.data,args.out,args.models
os.environ['V2D_PUBLIC']=str(DATA)
os.environ['V2D_CALIB']=str(DATA/'meta/camera_calibration.json')
os.environ['HF_HUB_OFFLINE']='1'
sys.path.insert(0,str(REPO/'scripts/track3'))
import cv2
import numpy as np
import torch
import register_poses as registration

if OUT.exists():raise ValueError('fresh output required')
baseline=args.baseline
base=json.loads(baseline.read_text())
if base['gt_use']!='none':raise ValueError('GT-free baseline required')
if hashlib.sha256((DATA/'meta/camera_calibration.json').read_bytes()).hexdigest()!=base['calibration_sha256']:
 raise ValueError('calibration changed')
for row in base['inputs']:
 path=DATA/row['path']
 if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('input changed')
for row in base['models']:
 path=MODELS/row['model'].replace('/','--')/row['revision']/row['file']
 if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('model changed')
registration.GDINO=str(MODELS/'IDEA-Research--grounding-dino-base/12bdfa3120f3e7ec7b434d90674b3396eccf88eb')
registration.SAM2=str(MODELS/'facebook--sam2.1-hiera-large/665f8e2ad61cf5f53d65644ff27c8ee525124610')
torch.set_num_threads(4)
seg=registration.Segmenter(['white_pot'])
rig=registration.load_rig()

class Capture:
 def __init__(self,processor):self.processor=processor;self.last=None
 def __call__(self,*args,**kwargs):return self.processor(*args,**kwargs)
 def post_process_grounded_object_detection(self,*args,**kwargs):
  result=self.processor.post_process_grounded_object_detection(*args,**kwargs)
  self.last=[{'label':label,'score':float(score),'box_xyxy':box} for label,score,box in zip(result[0]['text_labels'],result[0]['scores'].tolist(),result[0]['boxes'].tolist())]
  return result
seg.gp=Capture(seg.gp)
controls=[('white_pot','white pot'),('white_pot','cooking pot'),('white_pot','saucepan'),('white_pot_lid','white pot lid'),('white_pot_lid','pot lid')]
OUT.mkdir()
rows=[]
for ep,indices in [(2,[0,98,342]),(0,[0,106,372])]:
 for frame in indices:
  raw=registration.read_frame(DATA/f'videos/chunk-000/observation.images.ego_cam_c/episode_{ep:06d}.mp4',frame)
  image=cv2.remap(raw,*rig._maps_l,cv2.INTER_LINEAR)
  for name,phrase in controls:
   seg.names=[name];seg.prompt=phrase+'.';seg.phrase2idx={phrase:0};seg._order=[phrase]
   masks,assignments=seg(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),.30)
   mask=masks[0]
   prefix=OUT/f'e{ep:03d}_{frame:06d}_{phrase.replace(" ","_")}'
   path=Path(str(prefix)+'_mask.png');cv2.imwrite(str(path),mask.astype(np.uint8)*255)
   overlay=image.copy();overlay[mask]=np.uint8(.65*overlay[mask]+.35*np.array([0,255,0]))
   cv2.putText(overlay,f'ep {ep} frame {frame}: {phrase}',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
   cv2.imwrite(str(prefix)+'_overlay.jpg',overlay)
   row={'episode':ep,'frame_index':frame,'object':name,'prompt':seg.prompt,'detections':seg.gp.last,'assignments':assignments,'candidates':seg.last_candidates,'mask_pixels':int(mask.sum()),'mask_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
   rows.append(row);print(json.dumps(row),flush=True)
report={'baseline_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),'controls':controls,'frames':rows,'gt_use':'none','box_threshold':.30,'text_threshold':.25,'source_sha256_lf':hashlib.sha256(Path(__file__).read_text().encode()).hexdigest(),'limits':['Language controls only, no default changes, no GT pose read.','Source indices selected before evaluation; not full tracking or pose accuracy.']}
(OUT/'alias_results.json').write_text(json.dumps(report,indent=2))

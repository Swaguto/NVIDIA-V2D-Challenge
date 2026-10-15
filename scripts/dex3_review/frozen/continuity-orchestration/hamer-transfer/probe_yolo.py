import os
os.environ['YOLO_CONFIG_DIR']='/workspace/hand-continuity-repair/hamer-transfer/yolo-config'
import cv2,numpy as np,json,pathlib
from ultralytics import YOLO
r=pathlib.Path('/workspace/hand-continuity-repair');model=YOLO(str(r/'hamer-transfer/runtime/models/detector.pt'))
print('CLASSES',model.names,flush=True)
for fid in [170,190]:
 p=r/f'inputs/issue6-quality-cache/episodes/e041/issue6-full-inputs/e041/color/{fid:06d}.png';img=cv2.imread(str(p));print('FRAME',fid,flush=True)
 for name,roi in [('full',[0,0,1280,800]),('left',[430,330,780,680]),('right',[640,250,1030,680])]:
  x0,y0,x1,y1=roi;pred=model(img[y0:y1,x0:x1],conf=0.5,verbose=False,device=0)[0];rows=[]
  for b,s,c in zip(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy(),pred.boxes.cls.cpu().numpy()):rows.append({'class':int(c),'confidence':float(s),'box':(b+[x0,y0,x0,y0]).round(1).tolist()})
  print(name,rows,flush=True)

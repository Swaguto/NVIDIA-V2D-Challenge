import os,inspect,json,pathlib,importlib.util
os.environ['PYOPENGL_PLATFORM']='egl';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
for name,value in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('str',str),('unicode',str)]:
 if name not in np.__dict__:setattr(np,name,value)
if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
import torch,cv2
from PIL import Image
from hamer.models import load_hamer
r=pathlib.Path('/workspace/hand-continuity-repair');root=r/'roi-recovery/e041';out=root/'fresh-observations-v3';out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('crop_hamer',r/'code/reconstruction/modules/v2d_hamer/lib/_hamer.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
model,cfg=load_hamer(str(r/'hamer-transfer/runtime/weights'));model=model.cuda().eval()
manifest=json.load(open(r/'all20-results/e041/runs/e041/manifest.json'));frames={f['id']:f for f in manifest['frames']};gate=json.load(open(r/'inputs/issue6-quality-cache/episodes/e041/temporal-hands/e041/observation_gate.json'))['direct_observation'];summary={}
for side in ['left','right']:
 counts=[]
 for maskpath in sorted((root/'segmentation-v2'/side/'consensus').glob('*.png')):
  fid=int(maskpath.stem)
  mask=np.asarray(Image.open(maskpath))>0;ys,xs=np.nonzero(mask);cx=(xs.min()+xs.max())/2;cy=(ys.min()+ys.max())/2;size=max(xs.max()-xs.min(),ys.max()-ys.min())*1.7
  image=np.asarray(Image.open(frames[fid]['color']).convert('RGB'));h,w=image.shape[:2];pred=helper.run_hamer(model,cfg,image,float(cx),float(cy),float(size),side=='right')
  joints=pred['pred_keypoints_3d']+pred['pred_cam_t_full'];points=joints[:,:2]/joints[:,2:]*pred['scaled_focal_length']+[w/2,h/2];kp=np.c_[points,np.zeros(21)]
  semantic=cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8))
  for k,(x,y) in enumerate(points):
   x,y=round(x),round(y)
   if 0<=x<w and 0<=y<h and semantic[y,x]:kp[k,2]=.5
  n=int(np.count_nonzero(kp[:,2]));counts.append(n)
  if n>=4:
   p=out/side/f'{fid:06d}.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps({'keypoints':kp.tolist(),'mask':str(maskpath),'source':'bidirectional_sam2_consensus_and_fresh_hamer','estimated':True,'original_direct_observation':bool(gate[side][fid]),'visible_landmarks':n,'crop':[float(cx),float(cy),float(size)]},indent=2))
  if fid in [170,190]:
   view=image.copy();view[mask]=(view[mask]*.55+np.array([30,160,230])*.45).astype(np.uint8)
   for x,y,c in kp:
    if c:cv2.circle(view,(round(x),round(y)),4,(255,40,40),-1)
   Image.fromarray(view).save(out/f'{side}-{fid:06d}-evidence.jpg')
 summary[side]={'candidate_frames':len(counts),'accepted_frames':sum(x>=4 for x in counts),'minimum_visible_landmarks':min(counts) if counts else 0,'counts':counts};print('FRESH',side,summary[side],flush=True)
(out/'receipt.json').write_text(json.dumps({'episode':41,'source':'fresh source-image inference, never independent ground truth','original_contacts_unchanged':True,'hands':summary},indent=2));print('FRESH_OBSERVATIONS_DONE',flush=True)

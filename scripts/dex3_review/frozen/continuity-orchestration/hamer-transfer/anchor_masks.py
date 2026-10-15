import pathlib,json,os
os.environ['PYOPENGL_PLATFORM']='egl'
import numpy as np,torch,cv2
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
r=pathlib.Path('/workspace/hand-continuity-repair');out=r/'roi-recovery/e041/anchor-v2';out.mkdir(exist_ok=False);c=r/'inputs/issue6-quality-cache/episodes/e041';t=c/'temporal-hands/e041';audit=json.load(open(r/'hamer-transfer/runtime/metadata/e041/wearer_candidates.json'));manifest=json.load(open(r/'all20-results/e041/runs/e041/manifest.json'));frames={f['id']:f for f in manifest['frames']}
predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(r/'hamer-transfer/runtime/models/sam2.1_hiera_large.pt'),device='cuda',apply_postprocessing=False))
for side,h,fid in [('left',0,158),('left',0,206),('right',1,120),('right',1,182)]:
 kp=np.array(json.load(open(t/'tracks'/f'{h:03d}'/f'{fid:06d}_keypoints.json'))['people'][0]['pose_keypoints_2d']);row=audit['frames'][fid];candidate=row['candidates'][row['selected'][side]['candidate_index']];box=np.array(candidate['box_xyxy'],np.float32);forearm=kp[0,:2]+1.5*(kp[0,:2]-kp[9,:2]);points=np.array([kp[0,:2],kp[1,:2],forearm],np.float32);points=np.clip(points,[0,0],[1279,799]);labels=np.array([1,1,0],np.int32);image=np.array(Image.open(frames[fid]['color']).convert('RGB'))
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  predictor.set_image(image);masks,scores,_=predictor.predict(point_coords=points,point_labels=labels,box=box,multimask_output=False)
 mask=masks[0]>0;p=out/side;p.mkdir(exist_ok=True);Image.fromarray(mask.astype(np.uint8)*255).save(p/f'{fid:06d}.png');view=image.copy();view[mask]=(view[mask]*.55+np.array([30,160,230])*.45).astype(np.uint8)
 for point,label in zip(points,labels):cv2.circle(view,tuple(np.rint(point).astype(int)),6,(30,255,50) if label else (255,40,40),-1)
 Image.fromarray(view).save(p/f'{fid:06d}-evidence.jpg');print(side,fid,'pixels',int(mask.sum()),'score',scores.tolist(),'prompts',points.tolist(),flush=True)
print('ANCHORS_DONE',flush=True)

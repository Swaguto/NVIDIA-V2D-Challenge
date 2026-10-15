import os,json,pathlib,gc
os.environ['PYOPENGL_PLATFORM']='egl'
import numpy as np,torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
r=pathlib.Path('/workspace/hand-continuity-repair');root=r/'roi-recovery/e041/segmentation-v2';root.mkdir(parents=True,exist_ok=False)
source=r/'inputs/issue6-quality-cache/episodes/e041';manifest=json.load(open(r/'all20-results/e041/runs/e041/manifest.json'));frames={f['id']:f for f in manifest['frames']}
predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',str(r/'hamer-transfer/runtime/models/sam2.1_hiera_large.pt'),device='cuda',apply_postprocessing=False)
for side,a,b in [('left',158,206),('right',120,182)]:
 out=root/side;video=out/'frames';video.mkdir(parents=True)
 for i,fid in enumerate(range(a,b+1)):Image.open(frames[fid]['color']).convert('RGB').save(video/f'{i:06d}.jpg',quality=95)
 for direction,anchor,reverse in [('forward',0,False),('backward',b-a,True)]:
  mask=np.asarray(Image.open(r/'roi-recovery/e041/anchor-v2'/side/f'{a+anchor:06d}.png'))>0
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   state=predictor.init_state(str(video),offload_video_to_cpu=True,offload_state_to_cpu=True)
   predictor.add_new_mask(state,frame_idx=anchor,obj_id=1,mask=mask)
   for i,ids,logits in predictor.propagate_in_video(state,start_frame_idx=anchor,max_frame_num_to_track=b-a,reverse=reverse):
    m=(logits[0,0]>0).cpu().numpy();folder=out/direction;folder.mkdir(exist_ok=True);Image.fromarray(m.astype(np.uint8)*255).save(folder/f'{a+i:06d}.png')
  del state;gc.collect();torch.cuda.empty_cache()
 stats=[]
 for fid in range(a,b+1):
  f=np.asarray(Image.open(out/'forward'/f'{fid:06d}.png'))>0;q=np.asarray(Image.open(out/'backward'/f'{fid:06d}.png'))>0;inter=f&q;iou=inter.sum()/max((f|q).sum(),1);ys,xs=np.nonzero(inter);border=bool(len(xs) and (xs.min()<2 or ys.min()<2 or xs.max()>=1278 or ys.max()>=798));ok=iou>=.25 and inter.sum()>=64 and not border
  if ok:
   d=out/'consensus';d.mkdir(exist_ok=True);Image.fromarray(inter.astype(np.uint8)*255).save(d/f'{fid:06d}.png')
  stats.append({'frame_id':fid,'forward_pixels':int(f.sum()),'backward_pixels':int(q.sum()),'iou':float(iou),'accepted':bool(ok),'consensus_pixels':int(inter.sum()),'border':border})
 (out/'stats.json').write_text(json.dumps(stats,indent=2));print('SAM_SEGMENT',side,sum(s['accepted'] for s in stats),'/',len(stats),flush=True)
print('SAM_PROBE_DONE',flush=True)

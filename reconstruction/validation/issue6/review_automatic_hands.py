"""Render source versus automatically fitted hands; rejected fits remain marked."""
import argparse
import json
from pathlib import Path
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw

p=argparse.ArgumentParser(description=__doc__)
for name in ('source','quality','proposals','out'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();source=a.source;root=a.quality;out=a.out;out.mkdir(parents=True,exist_ok=False)
manifest=json.loads((source/'manifest.json').read_text()); c=manifest['color_camera']
rows=json.loads((root/'frame-evidence.json').read_text()); faces=np.load(a.proposals/'faces.npy')
fps=float(1/np.median(np.diff([r['timestamp_s'] for r in rows])))
writer=imageio.get_writer(out/'source-to-automatic-hands.mp4',fps=fps,codec='libx264',quality=8,macro_block_size=2)
thumbs=[]
try:
    for row in rows:
        frame=row['frame']; image=cv2.cvtColor(cv2.imread(str(source/'color'/f'{frame:06d}.png')),cv2.COLOR_BGR2RGB)
        overlay=image.copy(); labels=[]
        for side,item in row['hands'].items():
            z=np.load(root/'geometry'/side/f'{frame:06d}.npz'); vertices=z['vertices']; joints=z['joints']
            uv=vertices[:,:2]/vertices[:,2:]*[c['fx'],c['fy']]+[c['cx'],c['cy']]
            points=joints[:,:2]/joints[:,2:]*[c['fx'],c['fy']]+[c['cx'],c['cy']]
            ok=item['accepted_by_consistency_proxies']; color=(60,140,255) if side=='left' else (255,70,70)
            if not ok:color=(255,190,30)
            mask=np.zeros(image.shape[:2],np.uint8)
            polygons=np.rint(np.clip(uv[faces],-5000,5000)).astype(np.int32)
            cv2.fillPoly(mask,polygons,255)
            overlay[mask>0]=(0.65*overlay[mask>0]+0.35*np.array(color)).astype(np.uint8)
            for point in points:
                if np.isfinite(point).all():cv2.circle(overlay,tuple(np.rint(point).astype(int)),3,color,-1)
            labels.append(side+': '+('consistency pass' if ok else ', '.join(item['reasons'])))
        pair=Image.fromarray(np.concatenate([image,overlay],axis=1)).resize((1280,400))
        canvas=Image.new('RGB',(1280,448),(12,17,24)); canvas.paste(pair,(0,48)); d=ImageDraw.Draw(canvas)
        d.text((8,4),f'Frame {frame} | Source (left), automatic proposal (right) | yellow = rejected | accuracy NOT accepted',fill='white')
        d.text((8,19),' | '.join(labels),fill='white')
        writer.append_data(np.asarray(canvas)); thumbs.append(canvas.resize((640,224)))
        if len(thumbs)==12 or frame==rows[-1]['frame']:
            sheet=Image.new('RGB',(1280,224*6),(0,0,0))
            for i,thumb in enumerate(thumbs):sheet.paste(thumb,((i%2)*640,(i//2)*224))
            sheet.save(out/f'sheet-{frame-len(thumbs)+1:03d}.jpg'); thumbs=[]
finally:writer.close()
print(out)

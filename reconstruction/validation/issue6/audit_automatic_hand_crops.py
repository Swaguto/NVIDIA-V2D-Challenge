"""Produce enlarged source/mesh crops for a deterministic, risk-focused review.

Review selection is diagnostic only and never changes inference or support.
It includes regular samples, support transitions and worst accepted evidence.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps

p=argparse.ArgumentParser(description=__doc__)
for name in ('source','quality','proposals','out'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
rows=json.loads((a.quality/'frame-evidence.json').read_text())
detections={r['frame']:r for r in json.loads((a.proposals/'detections.json').read_text())}
c=json.loads((a.source/'manifest.json').read_text())['color_camera'];faces=np.load(a.proposals/'faces.npy')
chosen={s:set(range(0,len(rows),30))|{len(rows)-1} for s in ('left','right')}
for side in chosen:
    mask=[r['hands'].get(side,{}).get('accepted_by_consistency_proxies',False) for r in rows]
    for i in range(1,len(rows)):
        if mask[i]!=mask[i-1]:chosen[side].update((i-1,i))
    accepted=[i for i,ok in enumerate(mask) if ok]
    for key,reverse in [('mesh_mask_precision',False),('mask_mesh_recall',False),('median_depth_residual_m',True)]:
        chosen[side].update(sorted(accepted,key=lambda i:rows[i]['hands'][side][key],reverse=reverse)[:5])
tiles=[];selection=[]
for side in ('left','right'):
    for index in sorted(chosen[side]):
        row=rows[index];frame=row['frame'];item=row['hands'].get(side)
        if item is None:
            selection.append({'frame':frame,'side':side,'crop_available':False});continue
        with np.load(a.quality/'geometry'/side/f'{frame:06d}.npz',allow_pickle=False) as z:
            vertices=z['vertices'];joints=z['joints']
        uv=vertices[:,:2]/vertices[:,2:]*[c['fx'],c['fy']]+[c['cx'],c['cy']]
        points=joints[:,:2]/joints[:,2:]*[c['fx'],c['fy']]+[c['cx'],c['cy']]
        detection=detections[frame];candidate=detection['candidates'][detection['selected'][side]['candidate_index']]
        b=np.array(candidate['box_xyxy']);center=(b[:2]+b[2:])/2;size=(b[2:]-b[:2])*.7
        lo=np.floor(np.maximum(0,center-size)).astype(int);hi=np.ceil(np.minimum([c['width'],c['height']],center+size)).astype(int)
        image=cv2.cvtColor(cv2.imread(str(a.source/'color'/f'{frame:06d}.png')),cv2.COLOR_BGR2RGB)
        overlay=image.copy();silhouette=np.zeros(image.shape[:2],np.uint8)
        for polygon in np.rint(np.clip(uv[faces],-5000,5000)).astype(np.int32):
            cv2.fillConvexPoly(silhouette,polygon,255)
        ok=item['accepted_by_consistency_proxies'];color=(60,140,255) if side=='left' else (255,70,70)
        if not ok:color=(255,190,30)
        overlay[silhouette>0]=(.75*overlay[silhouette>0]+.25*np.array(color)).astype(np.uint8)
        for point in points:
            if np.isfinite(point).all():cv2.circle(overlay,tuple(np.rint(point).astype(int)),2,color,-1)
        tile=Image.new('RGB',(640,368),(12,17,24));draw=ImageDraw.Draw(tile)
        draw.text((5,4),f"{side} frame {frame}: {'CONSISTENCY PASS' if ok else 'REJECTED'}; pose accuracy unverified",fill='white')
        draw.text((5,19),f"Precision {item['mesh_mask_precision']:.3f}; recall {item['mask_mesh_recall']:.3f}; depth {item['median_depth_residual_m']}",fill='white')
        draw.text((5,34),'Source (left); estimated mesh/joints (right)',fill='white')
        for x,img in [(0,image),(320,overlay)]:
            crop=Image.fromarray(img[lo[1]:hi[1],lo[0]:hi[0]])
            tile.paste(ImageOps.pad(crop,(320,320),color=(0,0,0)),(x,48))
        selection.append({'frame':frame,'side':side,'crop_available':True,'accepted':ok,'crop_xyxy':[*lo.tolist(),*hi.tolist()]})
        tiles.append(tile)
        if len(tiles)==8:
            sheet=Image.new('RGB',(1280,1472))
            for j,t in enumerate(tiles):sheet.paste(t,((j%2)*640,(j//2)*368))
            sheet.save(a.out/f'audit-{len(list(a.out.glob("audit-*.jpg"))):03d}.jpg');tiles=[]
if tiles:
    sheet=Image.new('RGB',(1280,368*((len(tiles)+1)//2)))
    for j,t in enumerate(tiles):sheet.paste(t,((j%2)*640,(j//2)*368))
    sheet.save(a.out/f'audit-{len(list(a.out.glob("audit-*.jpg"))):03d}.jpg')
(a.out/'selection.json').write_text(json.dumps({'selection_rule':__doc__,'items':selection},indent=2)+'\n')
print(json.dumps({'selected_frames_and_sides':len(selection),'sheets':len(list(a.out.glob('audit-*.jpg')))}))

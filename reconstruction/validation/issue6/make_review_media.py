"""Source-camera video and timestamped sampled panels for visual acceptance."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image,ImageDraw

parser=argparse.ArgumentParser()
parser.add_argument('--episode',type=int,required=True)
args=parser.parse_args()
label=f'e{args.episode:03d}'
run=Path(os.environ.get('V2D_ISSUE6_RUN_ROOT','/workspace/issue6-full-runs/calibrated-temporal'))/label
manifest=json.loads((Path(os.environ.get('V2D_ISSUE6_INPUT_ROOT','/workspace/issue6-full-inputs'))/label/'manifest.json').read_text())
out=Path(os.environ.get('V2D_ISSUE6_REVIEW_ROOT','/workspace/pr32-evidence/review'))/label
out.mkdir(parents=True,exist_ok=True)
mesh=run/'align/hand_mesh_overlay.mp4'
source=out/'hand-camera-overlay.mp4'
subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-i',str(mesh),'-vf','crop=iw/2:ih/2:0:0',
                '-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(source)],check=True)
selected=np.unique(np.linspace(0,len(manifest['frames'])-1,12).round().astype(int)).tolist()
receipt={'episode':args.episode,'original_frame_ids':selected,
         'sampling':'Twelve evenly spaced original frames; this is sampled visual review, not independent labeling of every frame',
         'source_camera_crop':'Top-left panel from the native mesh renderer; no trajectory claim from its three camera-relative side panels',
         'videos':{}}
for name,path in [('mesh',source),('contacts',run/'report/contact_overlay.mp4')]:
    cap=cv2.VideoCapture(str(path))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    assert count==len(manifest['frames']),(path,count,len(manifest['frames']))
    assert fps>0
    canvas=Image.new('RGB',(640*3,428*4),(12,17,24))
    draw=ImageDraw.Draw(canvas)
    for position,index in enumerate(selected):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        okay,frame=cap.read()
        assert okay,(path,index)
        image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((640,400))
        x,y=(position%3)*640,(position//3)*428
        canvas.paste(image,(x,y))
        timestamp=manifest['frames'][index]['timestamp_s']
        draw.text((x+8,y+403),f'e{args.episode:03d} | original frame {index} | {timestamp:.3f} s',fill=(235,242,250))
    cap.release()
    canvas.save(out/f'{name}-montage.jpg',quality=94)
    receipt['videos'][name]={'path':str(path),'frames':count,'fps':fps}
(out/'review-media-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)

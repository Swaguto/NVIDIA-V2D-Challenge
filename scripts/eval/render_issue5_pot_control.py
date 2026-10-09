"""Render frozen pot/lid camera hypotheses against their input masks; no GT."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--observations',type=Path,required=True)
parser.add_argument('--masks',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
if args.out.exists(): parser.error('fresh output required')
repo=Path(__file__).resolve().parents[2]
os.environ['V2D_PUBLIC']=str(args.data)
os.environ['V2D_CALIB']=str(args.data/'meta/camera_calibration.json')
sys.path.insert(0,str(repo/'scripts/track3'))
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import register_poses as reg
from v2d.world_calib.vm.fit_ego_rig import load_mesh
from v2d.world_calib.vm.visualize_object_pose import render_depth

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
raw=args.observations.read_bytes()
report=json.loads(raw)
if (report['gt_use']!='none' or report['camera_frame']!='ego_cam_c'
        or report['coordinate_frame']!='rectified_camera'):
    raise ValueError('GT-free rectified-left observations required')
if sha(args.data/'meta/camera_calibration.json')!=report['calibration_sha256']:
    raise ValueError('calibration changed')
for item in report['inputs']:
    if sha(args.data/item['path'])!=item['sha256']:raise ValueError('RGB/CAD input changed')
rig=reg.load_rig()
meshes={name:load_mesh(args.data/f'mesh/{name}/{name}.glb') for name in report['objects']}
args.out.mkdir(parents=True)
results=[]
previews=[]
for row in report['frames']:
    episode,frame=row['episode'],row['frame_index']
    image=rig.rectify_pair(*[reg.read_frame(args.data/f'videos/chunk-000/observation.images.{cam}/episode_{episode:06d}.mp4',frame) for cam in (rig.left,rig.right)])[0]
    details={}
    for i,name in enumerate(report['objects']):
        p=args.masks/f'e{episode:03d}/{frame:06d}/{name}_mask.png'
        if sha(p)!=row['views'][rig.left]['objects'][name]['mask_sha256']:raise ValueError('mask changed')
        mask=cv2.imread(str(p),0)>0
        image[mask]=np.uint8(.65*image[mask]+.35*np.array([(0,255,0),(255,0,255)][i%2]))
        fit=row['fits'][name]
        if fit['status']!='observed':
            details[name]={'status':fit['status']};continue
        R=Rotation.from_quat(np.array(fit['rotation_wxyz'])[[1,2,3,0]]).as_matrix()
        mesh=meshes[name]
        vertices=fit['scale']*(np.asarray(mesh.vertices)@R.T)+fit['translation']
        rendered=render_depth(vertices,mesh.faces,rig.P1[:3,:3],image.shape[:2])
        silhouette=np.isfinite(rendered)
        contours=cv2.findContours(silhouette.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(image,contours,-1,(0,255,255),2)
        details[name]={'status':'observed','full_cad_mask_iou':float((mask&silhouette).sum()/max((mask|silhouette).sum(),1)),'mask_sha256':sha(p)}
    cv2.putText(image,f'ep{episode} frame{frame}: pot green / lid magenta / CAD yellow',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
    out=args.out/f'e{episode:03d}_{frame:06d}.jpg'
    cv2.imwrite(str(out),image)
    previews.append(cv2.resize(image,(640,400)))
    results.append({'episode':episode,'frame_index':frame,'objects':details,'overlay_sha256':sha(out)})
    print(json.dumps(results[-1]),flush=True)
if len(previews)%2:previews.append(np.zeros_like(previews[0]))
if previews:cv2.imwrite(str(args.out/'preview.jpg'),np.vstack([np.hstack(previews[i:i+2]) for i in range(0,len(previews),2)]))
(args.out/'render_results.json').write_text(json.dumps({'prediction_sha256':hashlib.sha256(raw).hexdigest(),'renderer_sha256_lf':hashlib.sha256(Path(__file__).read_text().encode()).hexdigest(),'gt_use':'none','frames':results,'limits':['Full CAD versus input SAM masks; occlusion is not independently labelled.','No canonical pose or world-camera accuracy measured by this renderer.']},indent=2))
if args.observations.read_bytes()!=raw:raise ValueError('predictions changed during rendering')

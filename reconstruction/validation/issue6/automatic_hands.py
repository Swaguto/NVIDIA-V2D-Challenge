"""Image-supported automatic hand recovery; no episode/frame-specific crops.

Development outputs are proposals. Geometry checks are image/depth consistency
proxies, not ground-truth pose accuracy. Missing evidence remains unknown.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

import cv2
import numpy as np
from wearer_selection import WearerSelector, candidate_depth


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def method_hash(config):
    return {p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('wearer_selection.py'),
                                  Path(__file__).with_name('automatic_hand_quality.py'),
                                  Path(__file__).with_name('run_automatic_hand_suite.py'),config]}


def iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    return float(inter/max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter,1))


def suppress(candidates, threshold):
    result=[]
    for c in sorted(candidates,key=lambda c:(-c['confidence'],c['class'],c['box_xyxy'])):
        if not any(c['class']==x['class'] and iou(c['box_xyxy'],x['box_xyxy'])>threshold for x in result):
            result.append(c)
    return result


def reject_cross_side_conflict(selected,candidates,threshold):
    a,b=(selected[s]['candidate_index'] for s in ('left','right'))
    if a is not None and b is not None and iou(candidates[a]['box_xyxy'],candidates[b]['box_xyxy'])>threshold:
        continuing=[s for s in ('left','right') if selected[s].get('reason')=='stereo_depth_and_temporal_continuity']
        rejected=[s for s in ('left','right') if s not in continuing] if len(continuing)==1 else ['left','right']
        for side in rejected:
            selected[side]={'candidate_index':None,'reason':'ambiguous_left_right_assignment_to_same_image_region'}
        return rejected
    return []


def flow_crop(previous,current,box,policy):
    """Flow only proposes a search crop; it never supplies a hand pose."""
    mask=np.zeros_like(previous); x0,y0,x1,y1=np.asarray(box,dtype=int)
    mask[max(0,y0):min(mask.shape[0],y1),max(0,x0):min(mask.shape[1],x1)]=255
    p=cv2.goodFeaturesToTrack(previous,100,0.01,5,mask=mask)
    if p is None:return None
    q,s,_=cv2.calcOpticalFlowPyrLK(previous,current,p,None)
    if q is None:return None
    back,t,_=cv2.calcOpticalFlowPyrLK(current,previous,q,None)
    if back is None:return None
    good=(s[:,0]>0)&(t[:,0]>0)&(np.linalg.norm(p-back,axis=(1,2))<=policy['maximum_flow_roundtrip_px'])
    if good.sum()<policy['minimum_flow_points']:return None
    delta=np.median((q-p)[good,0],axis=0)
    return (np.asarray(box)+np.tile(delta,2)).tolist()


def detect(source,manifest,out,args,policy,model=None):
    from ultralytics import YOLO
    if model is None:model=YOLO(str(args.detector)).to('cuda')
    c=manifest['color_camera']; width,height=c['width'],c['height']
    selector=WearerSelector(width,height); previous=None; last={}; rows=[]
    for row in manifest['frames']:
        frame=row['id']; path=source/'color'/f'{frame:06d}.png'
        depth_path=(args.depth_root or source/'depth')/f'{frame:06d}.npy'
        image=cv2.imread(str(path)); depth=np.load(depth_path,allow_pickle=False)
        if image is None or image.shape[:2]!=(height,width):raise ValueError('Missing or incompatible source image')
        validity=source/'valid'/f'{frame:06d}.png'
        if validity.exists():depth=np.where(cv2.imread(str(validity),0)>0,depth,np.nan)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); candidates=[]
        views=[(image,None,False)]
        fraction=(1+policy['grid_overlap_fraction'])/2
        tile_w,tile_h=int(width*fraction),int(height*fraction)
        for y0 in (0,height-tile_h):
            for x0 in (0,width-tile_w):
                crop=np.array([x0,y0,x0+tile_w,y0+tile_h])
                views.append((image[y0:y0+tile_h,x0:x0+tile_w],crop,False))
        for side,(last_frame,box) in last.items():
            if previous is None or frame-last_frame>policy['maximum_search_gap_frames']:continue
            moved=flow_crop(previous,gray,box,policy)
            if moved is None:continue
            last[side]=(last_frame,moved)
            b=np.asarray(moved); center=(b[:2]+b[2:])/2; size=(b[2:]-b[:2])*policy['search_crop_expansion']/2
            crop=np.rint(np.r_[center-size,center+size]).astype(int)
            crop=np.maximum(0,np.minimum(crop,[width,height,width,height]))
            x0,y0,x1,y1=crop
            if min(x1-x0,y1-y0)>=16:views.append((image[y0:y1,x0:x1],crop,False))
        # Keep each view's letterboxing independent of other crop shapes.
        # A horizontal reflection reverses anatomical handedness in image space.
        views=[item for view,crop,_ in views for item in
               ((view,crop,False),(cv2.flip(view,1),crop,True))]
        predictions=[model(view,conf=policy['detector_confidence'],verbose=False)[0] for view,_,_ in views]
        for (view,crop,mirrored),result in zip(views,predictions,strict=True):
            for box,confidence,label in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy()):
                if int(label) not in (0,1):continue
                if mirrored:
                    box=np.array([view.shape[1]-box[2],box[1],view.shape[1]-box[0],box[3]])
                    label=1-int(label)
                if crop is not None:box=box+np.tile(crop[:2],2)
                distance,n,fraction=candidate_depth(box,depth)
                candidates.append({'box_xyxy':box.tolist(),'confidence':float(confidence),'class':int(label),
                    'depth_m':distance,'depth_pixels':n,'depth_valid_fraction':fraction,
                    'view':'automatic_current_image_crop' if crop is not None else 'current_image',
                    'mirrored':mirrored})
        candidates=suppress(candidates,policy['nms_iou'])
        prior_state=copy.deepcopy(selector.previous); selected=selector.select(frame,candidates)
        for side in reject_cross_side_conflict(selected,candidates,policy['maximum_cross_side_box_iou']):
            if side in prior_state:selector.previous[side]=prior_state[side]
            else:selector.previous.pop(side,None)
        for side in ('left','right'):
            index=selected[side]['candidate_index']
            if index is not None:last[side]=(frame,candidates[index]['box_xyxy'])
        rows.append({'frame':frame,'timestamp_s':row['timestamp_s'],'source_image_sha256':sha(path),
                     'depth_sha256':sha(depth_path),'candidates':candidates,'selected':selected})
        previous=gray
        if len(rows)%25==0:
            write(out/'detections.partial.json',rows)
            print(json.dumps({'stage':'detect','frames':len(rows)}),flush=True)
    write(out/'detections.json',rows)
    del model
    import torch
    torch.cuda.empty_cache()
    return rows


def fit(source,manifest,out,args,policy,rows,loaded=None):
    import torch
    sys.path.insert(0,str(args.hamer_source))
    from hamer.models import load_hamer
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    model,cfg=loaded if loaded is not None else load_hamer(str(args.hamer_weights))
    model=model.to('cuda').eval()
    camera=manifest['color_camera']; K=np.array([[camera['fx'],0,camera['cx']],[0,camera['fy'],camera['cy']],[0,0,1]],float)
    faces=model.mano.faces.copy(); np.save(out/'faces.npy',faces)
    for row in rows:
        frame=row['frame']; image_path=source/'color'/f'{frame:06d}.png'
        if sha(image_path)!=row['source_image_sha256']:raise ValueError('Source image changed since detection')
        image=cv2.imread(str(image_path))
        for side in ('left','right'):
            chosen=row['selected'][side]['candidate_index']
            if chosen is None:continue
            candidate=row['candidates'][chosen]; right=side=='right'
            sample=ViTDetDataset(cfg,image,np.array([candidate['box_xyxy']]),np.array([right]),
                                 np.zeros((1,21,3)),rescale_factor=policy['crop_rescale_factor'])[0]
            with torch.inference_mode():
                prediction=model({'img':torch.from_numpy(sample['img'])[None].to('cuda'),
                                  'right':torch.tensor([right],device='cuda')})
            v=prediction['pred_vertices'][0].cpu().numpy().astype(float)
            j=prediction['pred_keypoints_3d'][0].cpu().numpy().astype(float)
            cam=prediction['pred_cam'][0].cpu().numpy().astype(float); cam[1]*=1 if right else -1
            v[:,0]*=1 if right else -1; j[:,0]*=1 if right else -1
            width,height=sample['img_size']; f=float(cfg.EXTRA.FOCAL_LENGTH/cfg.MODEL.IMAGE_SIZE*max(width,height))
            cx,cy=sample['box_center']; size=float(sample['box_size'])
            if not np.isfinite(cam).all() or cam[0]<=0:continue
            z=2*f/(size*cam[0]+1e-6)
            translation=np.array([cam[1]+z/f*(cx-width/2),cam[2]+z/f*(cy-height/2),z])
            virtual=v+translation; uv=virtual[:,:2]/virtual[:,2:]*f+[width/2,height/2]
            ok,rvec,tvec=cv2.solvePnP(v,uv,K,None,flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:continue
            R=cv2.Rodrigues(rvec)[0]; vv=v@R.T+tvec.reshape(3); jj=j@R.T+tvec.reshape(3)
            if not np.isfinite(vv).all() or np.min(vv[:,2])<=0:continue
            projected=vv[:,:2]/vv[:,2:]*[camera['fx'],camera['fy']]+[camera['cx'],camera['cy']]
            dest=out/'raw'/side/f'{frame:06d}.npz'; dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,vertices=vv,joints=jj,box=np.asarray(candidate['box_xyxy']),
                                projection_difference_px=np.linalg.norm(projected-uv,axis=1))
            params=prediction['pred_mano_params'][0]
            flip=np.diag([1 if right else -1,1,1])
            global_rotation=flip@R@flip@np.asarray(params['global_orient']).reshape(3,3)
            with torch.inference_mode():
                regenerated=model.mano(global_orient=torch.tensor(global_rotation,dtype=torch.float32,device='cuda')[None,None],
                    hand_pose=torch.tensor(params['hand_pose'],dtype=torch.float32,device='cuda')[None],
                    betas=torch.tensor(params['betas'],dtype=torch.float32,device='cuda').reshape(1,10),pose2rot=False)
            nv=regenerated.vertices[0].cpu().numpy().astype(float); nv[:,0]*=1 if right else -1
            cam_t=(vv-nv).mean(axis=0)
            centered_error=float(np.linalg.norm(nv+cam_t-vv,axis=1).max())
            if centered_error>1e-4:raise ValueError('Calibrated MANO conversion changed hand shape')
            record={'frame_idx':frame,'inference_frame_idx':frame,'timestamp_s':row['timestamp_s'],
                'track_id':3 if right else 2,'is_right':right,'directly_observed':True,
                'mano':{'global_orient':cv2.Rodrigues(global_rotation)[0].reshape(3).tolist(),
                        'hand_pose':np.concatenate([cv2.Rodrigues(m)[0].reshape(3) for m in np.asarray(params['hand_pose']).reshape(15,3,3)]).tolist(),
                        'betas':np.asarray(params['betas']).reshape(10).tolist()},
                'camera':{'pred_cam_t_full':cam_t.tolist(),'scaled_focal_length':camera['fx'],
                          'scaled_focal_length_y':camera['fy'],'principal_point':[camera['cx'],camera['cy']]},
                'direct_observation':{'kind':'automatic_current_image_detector','candidate':candidate,
                    'source_image_sha256':row['source_image_sha256'],'interpolated':False,
                    'calibrated_camera_centered_error_m':centered_error},'quality_acceptance':False}
            write(out/'records'/str(record['track_id'])/f'{frame:06d}.json',record)
        if frame%25==0:print(json.dumps({'stage':'fit','frame':frame}),flush=True)
    del model
    torch.cuda.empty_cache()


def main(argv=None,shared=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','out','detector','hamer-source','hamer-weights','sam-checkpoint'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_suffix('.json'))
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--depth-root',type=Path,help='Metric depth on the same rectified source pixel grid')
    parser.add_argument('--stage',choices=['detect','fit','all'],default='all')
    args=parser.parse_args(argv); policy=json.loads(args.config.read_text());shared=shared or {}
    manifest=json.loads((args.source/'manifest.json').read_text()); hashes=method_hash(args.config)
    if manifest['episode'] in policy['holdout_episodes']:
        if args.freeze is None:raise ValueError('Holdout requires a previously frozen method')
        frozen=json.loads(args.freeze.read_text())
        if frozen['method_files']!=hashes:raise ValueError('Frozen method changed')
    if args.stage!='fit':args.out.mkdir(parents=True,exist_ok=False)
    else:
        parent=json.loads((args.out/'receipt.json').read_text())
        if parent['method_files']!=hashes:raise ValueError('Detector method changed before fitting')
        write(args.out/'detection-receipt.json',parent)
    snapshot=args.out/('fit-source' if args.stage=='fit' else 'source')
    snapshot.mkdir(exist_ok=False)
    for name in hashes:
        path=args.config if name==args.config.name else Path(__file__).with_name(name)
        shutil.copy2(path,snapshot/name)
    depth_root=args.depth_root or args.source/'depth'
    for row in manifest['frames']:
        for path in (args.source/'color'/f"{row['id']:06d}.png",depth_root/f"{row['id']:06d}.npy"):
            if not path.is_file():raise FileNotFoundError(path)
    import torch
    random.seed(policy['seed']); np.random.seed(policy['seed']); torch.manual_seed(policy['seed'])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    receipt={'episode':manifest['episode'],'method_files':hashes,'policy':policy,
             'source_manifest_sha256':sha(args.source/'manifest.json'),'detector_sha256':sha(args.detector),
             'hamer_checkpoint_sha256':sha(args.hamer_weights/'_DATA/hamer_ckpts/checkpoints/hamer.ckpt'),
             'sam_checkpoint_sha256':sha(args.sam_checkpoint),
             'mano_right_sha256':sha(args.hamer_weights/'_DATA/data/mano/MANO_RIGHT.pkl'),
             'hamer_config_sha256':sha(args.hamer_weights/'_DATA/hamer_ckpts/model_config.yaml'),
             'mano_mean_params_sha256':sha(args.hamer_weights/'_DATA/data/mano_mean_params.npz'),
             'hamer_source_commit':subprocess.check_output(['git','-C',str(args.hamer_source),'rev-parse','HEAD'],text=True).strip(),
             'runtime':{'python':sys.version,'gpu':torch.cuda.get_device_name(0),
                        'packages':{name:importlib.metadata.version(name) for name in ('torch','numpy','ultralytics','smplx','pytorch-lightning','timm','opencv-python')}},
             'freeze_sha256':sha(args.freeze) if args.freeze else None,
             'manual_frame_or_crop_choices':False,'quality_accepted':False,'status':'running'}
    if args.freeze:
        for key,value in json.loads(args.freeze.read_text())['models'].items():
            if receipt.get(key)!=value:raise ValueError('Frozen model differs: '+key)
    if subprocess.check_output(['git','-C',str(args.hamer_source),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('HaMeR source has unrecorded tracked modifications')
    write(args.out/'receipt.json',receipt)
    try:
        rows=detect(args.source,manifest,args.out,args,policy,shared.get('detector')) if args.stage!='fit' else json.loads((args.out/'detections.json').read_text())
        if args.stage!='detect':fit(args.source,manifest,args.out,args,policy,rows,shared.get('hamer'))
        receipt['detections_sha256']=sha(args.out/'detections.json')
        if (args.out/'detection-receipt.json').exists():receipt['detection_receipt_sha256']=sha(args.out/'detection-receipt.json')
        receipt['status']='completed_detections' if args.stage=='detect' else 'completed_proposals_require_accuracy_checks'
        if method_hash(args.config)!=hashes:
            receipt['status']='completed_development_proposals_method_changed_during_run'
    except Exception as exc:
        receipt.update(status='failed',error=str(exc)); raise
    finally:write(args.out/'receipt.json',receipt)


if __name__=='__main__':main()

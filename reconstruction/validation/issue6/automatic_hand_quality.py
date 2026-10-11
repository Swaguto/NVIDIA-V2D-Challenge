"""Check automatic proposals against current image masks, depth and image motion.

No contact labels or episode-specific intervals enter this evaluator. These
checks can reject bad fits but cannot certify ground-truth hand accuracy.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from automatic_hands import method_hash, sha, write


def classify(mask,rendered,depth,mask_score,policy):
    rendered_hand=rendered>0
    measured_scene=np.isfinite(depth)&(depth>0)
    occluded=rendered_hand&measured_scene&(rendered>depth+policy['occlusion_tolerance_m'])
    visible=rendered_hand&~occluded
    intersection=mask&visible
    precision=float(intersection.sum()/max(visible.sum(),1))
    recall=float(intersection.sum()/max(mask.sum(),1))
    support=intersection&np.isfinite(depth)&(depth>0)
    residual=np.abs(rendered[support]-depth[support])
    median=float(np.median(residual)) if residual.size else None
    reasons=[]
    if mask_score<policy['minimum_mask_score']:reasons.append('uncertain_image_mask')
    if precision<policy['minimum_mesh_mask_precision']:reasons.append('mesh_outside_image_hand_mask')
    if recall<policy['minimum_mask_mesh_recall']:reasons.append('mesh_does_not_cover_image_hand_mask')
    if support.sum()<policy['minimum_stereo_pixels']:reasons.append('insufficient_stereo_support')
    if median is None or median>=policy['maximum_depth_residual_m']:reasons.append('depth_mismatch')
    return {'mesh_mask_precision':precision,'mask_mesh_recall':recall,'mask_score':float(mask_score),
            'depth_occluded_mesh_pixels':int(occluded.sum()),'visible_mesh_pixels':int(visible.sum()),
            'stereo_pixels':int(support.sum()),'median_depth_residual_m':median,'reasons':reasons}


def motion_disagreement(old_gray,gray,old_uv,uv,policy):
    p=old_uv.astype(np.float32).reshape(-1,1,2)
    q,s,_=cv2.calcOpticalFlowPyrLK(old_gray,gray,p,None)
    if q is None:return {'points':0,'median_error_px':None}
    back,t,_=cv2.calcOpticalFlowPyrLK(gray,old_gray,q,None)
    if back is None:return {'points':0,'median_error_px':None}
    good=(s[:,0]>0)&(t[:,0]>0)&(np.linalg.norm(p-back,axis=(1,2))<=policy['maximum_flow_roundtrip_px'])
    error=np.linalg.norm(q[:,0]-uv,axis=1)[good]
    return {'points':int(good.sum()),'median_error_px':float(np.median(error)) if len(error) else None}


def run(args,predictor=None):
    import pyrender
    import trimesh
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    policy=json.loads(args.config.read_text()); manifest=json.loads((args.source/'manifest.json').read_text())
    hashes=method_hash(args.config)
    proposal_receipt=json.loads((args.proposals/'receipt.json').read_text())
    if proposal_receipt['source_manifest_sha256']!=sha(args.source/'manifest.json'):
        raise ValueError('Source manifest changed since inference')
    if manifest['episode'] in policy['holdout_episodes']:
        if not proposal_receipt.get('freeze_sha256') or proposal_receipt['method_files']!=method_hash(args.config):
            raise ValueError('Holdout quality evaluation requires unchanged frozen proposals/method')
        if sha(args.sam_checkpoint)!=proposal_receipt['sam_checkpoint_sha256']:
            raise ValueError('Frozen segmentation model differs')
    rows=json.loads((args.proposals/'detections.json').read_text()); c=manifest['color_camera']
    out=args.out; out.mkdir(parents=True,exist_ok=False)
    (out/'source').mkdir()
    for name in hashes:
        src=args.config if name==args.config.name else Path(__file__).with_name(name)
        shutil.copy2(src,out/'source'/name)
    if predictor is None:
        predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(args.sam_checkpoint),device='cuda'))
    renderer=pyrender.OffscreenRenderer(c['width'],c['height'])
    camera=pyrender.IntrinsicsCamera(c['fx'],c['fy'],c['cx'],c['cy'])
    faces=np.load(args.proposals/'faces.npy',allow_pickle=False)

    def project(points):return points[:,:2]/points[:,2:]*[c['fx'],c['fy']]+[c['cx'],c['cy']]

    def render(vertices,right):
        scene=pyrender.Scene(); scene.add(camera,pose=np.eye(4))
        mesh=trimesh.Trimesh(vertices*[1,-1,-1],faces if right else faces[:,::-1],process=False)
        scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=False))
        return renderer.render(scene,flags=pyrender.RenderFlags.DEPTH_ONLY|pyrender.RenderFlags.SKIP_CULL_FACES)

    previous={}; results=[]; frames=[]
    try:
        for row in rows:
            frame=row['frame']; image_path=args.source/'color'/f'{frame:06d}.png'
            if sha(image_path)!=row['source_image_sha256']:raise ValueError('Source image changed since detection')
            image=cv2.imread(str(image_path))
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            depth_path=(args.depth_root or args.source/'depth')/f'{frame:06d}.npy'
            if sha(depth_path)!=row['depth_sha256']:raise ValueError('Depth changed since detection')
            depth=np.load(depth_path,allow_pickle=False)
            validity=args.source/'valid'/f'{frame:06d}.png'
            if validity.exists():depth=np.where(cv2.imread(str(validity),0)>0,depth,np.nan)
            available=[(side,args.proposals/'raw'/side/f'{frame:06d}.npz') for side in ('left','right')]
            available=[(side,p) for side,p in available if p.exists()]
            if available:predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            frame_result={'frame':frame,'timestamp_s':row['timestamp_s'],'hands':{}}
            for side,path in available:
                with np.load(path,allow_pickle=False) as data:
                    vertices=data['vertices'].copy(); joints=data['joints'].copy(); box=data['box'].copy()
                with torch.inference_mode():
                    masks,scores,_=predictor.predict(box=box,multimask_output=False)
                mask=masks[0].astype(bool); score=float(scores[0])
                original_vertices=vertices.copy(); original_joints=joints.copy(); offset=np.zeros(3)
                for _ in range(3):
                    rendered=render(vertices,side=='right')
                    support=mask&(rendered>0)&np.isfinite(depth)&(depth>0)
                    if support.sum()<policy['minimum_stereo_pixels']:break
                    dz=float(np.median(depth[support]-rendered[support]))
                    center=vertices.mean(axis=0)
                    step=center/center[2]*dz
                    if np.linalg.norm(step)>0.5:break
                    vertices+=step; joints+=step; offset+=step
                rendered=render(vertices,side=='right')
                measurement=classify(mask,rendered,depth,score,policy)
                uv=project(joints); motion={'points':0,'median_error_px':None}
                if side in previous and previous[side][0]==frame-1:
                    _,old_gray,old_uv=previous[side]
                    motion=motion_disagreement(old_gray,gray,old_uv,uv,policy)
                    if motion['points']>=policy['minimum_flow_points'] and motion['median_error_px']>policy['maximum_flow_pose_disagreement_px']:
                        measurement['reasons'].append('pose_jump_inconsistent_with_image_motion')
                # Keep the previous measured proposal for diagnostics, not hidden-state filling.
                previous[side]=(frame,gray.copy(),uv.copy())
                accepted=not measurement['reasons']
                dest=out/'geometry'/side/f'{frame:06d}.npz'; dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,vertices=vertices,joints=joints,raw_vertices=original_vertices,
                                    raw_joints=original_joints,translation_offset=offset,accepted=np.array(accepted))
                maskpath=out/'masks'/side/f'{frame:06d}.png'; maskpath.parent.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(maskpath),mask.astype(np.uint8)*255)
                record={'frame':frame,'side':side,'accepted_by_consistency_proxies':accepted,
                        **measurement,'motion':motion,'source_fit_sha256':sha(path),
                        'source_image_sha256':row['source_image_sha256'],'quality_accepted':False}
                native_record=args.proposals/'records'/('3' if side=='right' else '2')/f'{frame:06d}.json'
                if accepted and native_record.exists():
                    aligned=json.loads(native_record.read_text())
                    aligned.update(cam_t=(np.asarray(aligned['camera']['pred_cam_t_full'])+offset).tolist(),
                                   hand_scale=1.0,intrinsics=c,
                                   diagnostics={'metric_valid':True,'n_pixels':measurement['stereo_pixels'],
                                                'median_abs_residual_m':measurement['median_depth_residual_m'],
                                                'maximum_residual_m':policy['maximum_depth_residual_m']},
                                   automatic_consistency=record)
                    write(out/'accepted/records'/str(aligned['track_id'])/f'{frame:06d}.json',aligned)
                    accepted_mask=out/'accepted/masks'/side/f'{frame:06d}.png'
                    accepted_mask.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(maskpath,accepted_mask)
                results.append(record); frame_result['hands'][side]=record
            frames.append(frame_result)
            if len(frames)%25==0:
                write(out/'frame-evidence.partial.json',frames)
                print(json.dumps({'stage':'quality','frames':len(frames)}),flush=True)
    finally:renderer.delete()
    summary={'episode':manifest['episode'],'frames':len(frames),'quality_accepted':False,
             'method_files':hashes,'method_unchanged_during_run':hashes==method_hash(args.config),
             'method':'current_image_mask_and_stereo_depth_and_optical_flow_consistency',
             'source_proposal_receipt_sha256':sha(args.proposals/'receipt.json'),
             'sam_checkpoint_sha256':sha(args.sam_checkpoint),'config_sha256':sha(args.config),
             'hand_counts':{s:sum(r['side']==s and r['accepted_by_consistency_proxies'] for r in results) for s in ('left','right')}}
    for side in ('left','right'):
        mask=np.array([f['hands'].get(side,{}).get('accepted_by_consistency_proxies',False) for f in frames])
        summary.setdefault('visibility_transitions',{})[side]=int(np.count_nonzero(mask[1:]!=mask[:-1]))
        longest=run=0
        for present in mask:
            run=0 if present else run+1;longest=max(longest,run)
        summary.setdefault('longest_unsupported_run_frames',{})[side]=longest
    write(out/'frame-evidence.json',frames); write(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','proposals','out','sam-checkpoint'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('automatic_hands.json'))
    p.add_argument('--depth-root',type=Path)
    run(p.parse_args())

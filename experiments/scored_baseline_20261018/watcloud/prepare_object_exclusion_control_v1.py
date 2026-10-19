"""Bind inferred object ambiguity and test only cached-mask CPU postprocessing."""
import copy,json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
S=Path(__file__).resolve().parents[2];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/track3'))
import recover_source_hand as recovery
from merge_recovered_source_hand import validate_recovery
old_request=S/'e037-fresh-source-hand-request-v2/request.json'
parent=S/'e037-watcloud-fresh-hand-v1/outputs/recovery'
O=S/'e037-source-hand-exclusion-request-v1';O.mkdir()
data=json.loads(old_request.read_text());bindings=[]
for row in data['frames']:
    f=row['frame_idx'];mask=S/f'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/{f:06d}.png'
    row['inferred_exclusion']=dict(path=str(mask),sha256=recovery.digest(mask),kind='inferred_source_object_segmentation',source_image_sha256=row['image_sha256'],ambiguous_regions_are_unknown=True)
    excluded=recovery.exclusion_mask(row,(data['camera']['height'],data['camera']['width']))
    points,_,labels=recovery.mask_prompts(row['keypoints'],excluded)
    assert np.count_nonzero(points[:,2])>=4
    bindings.append(dict(frame=f,inferred_object_mask_sha256=recovery.digest(mask),source_image_sha256=row['image_sha256'],remaining_positive_prompts=int((labels==1).sum()),inferred_negative_prompts=int((labels==0).sum())))
request=O/'request.json';request.write_text(json.dumps(data,indent=2)+'\n')
recovery.canonical_sources(recovery.validate_request(request))
reused,parent_receipt=recovery.validate_pose_reuse(request,old_request,parent)
C=S/'e037-object-conflict-mask-control-v1';C.mkdir()
receipt=copy.deepcopy(parent_receipt);receipt.update(request_sha256=recovery.digest(request),producer_sha256=recovery.digest(Path(__file__)),rows=[],files={},complete=False,pose_source_receipt_sha256=recovery.digest(parent/'receipt.json'),hamer_inference_reused=True,sam2_inference_reused=True,sam2_negative_prompts_executed=False,cached_mask_postprocessing_only=True)
iterator=iter(data['frames']);scores={r['frame_idx']:r['mask_score'] for r in parent_receipt['rows']}
def cached_masks(image,points,box,point_labels):
    row=next(iterator);f=row['frame_idx']
    mask=np.asarray(Image.open(parent/f'image-observations/right/{f:06d}-mask.png'))>0
    return mask[None],np.array([scores[f]])
recovery.infer_frames(data,C,recovery.digest(request),cached_masks,None,receipt,reused)
receipt['files']=recovery.output_files(C);receipt['complete']=True
(C/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
_,_,_,accepted,_=validate_recovery(request,C)
rows=[]
for key,name,estimate,maskname in accepted:
    f=key[1];before=json.loads((parent/name).read_text());after=json.loads((C/name).read_text())
    for field in ('mano','camera','cam_t','hand_scale','crop'):assert before[field]==after[field]
    pixels=np.asarray(Image.open(C/maskname))>0
    row=next(r for r in data['frames'] if r['frame_idx']==f);excluded=recovery.exclusion_mask(row,pixels.shape)
    assert not (pixels&excluded).any()
    kp=np.asarray(json.loads((C/estimate).read_text())['keypoints'])
    rows.append(dict(frame=f,retained_image_points=int((kp[:,2]>0).sum()),excluded_mask_pixels=0,pose_and_camera_unchanged=True))
result=dict(schema='v2d.source-object-exclusion-CPU-control.v1',producer_sha256=recovery.digest(Path(__file__)),parent_request_sha256=recovery.digest(old_request),new_request_sha256=recovery.digest(request),parent_GPU_recovery_receipt_sha256=recovery.digest(parent/'receipt.json'),source_object_masks_are_inferred=True,original_source_flags_changed=False,source_model_inference_repeated=False,new_SAM2_negative_prompt_inference_executed=False,geometry_or_training_performed=False,mask_exclusion_is_not_accuracy_evidence=True,canonical_hand_sources_verified=302,bindings=bindings,rows=rows,reference_accepted=False,training_approved=False,physical_task_accepted=False)
(O/'CPU-control-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

"""Object/hand ambiguity and unchanged-pose reuse through real file consumers."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]))
import recover_source_hand as recovery
import merge_recovered_source_hand as merger
from test_source_hand_recovery_merge import bound_request, fake_inference, recovery_request


def child_request(bound_request,tmp_path,mask=None):
    path,data=bound_request
    child=copy.deepcopy(data)
    mask=np.zeros((64,128),bool) if mask is None else mask
    if not mask.any():mask[5:22,5:22]=True
    p=tmp_path/'object-estimate.png';Image.fromarray(mask.astype(np.uint8)*255).save(p)
    child['frames'][0]['inferred_exclusion']=dict(path=str(p),sha256=recovery.digest(p),
        kind='inferred_source_object_segmentation',source_image_sha256=child['frames'][0]['image_sha256'],
        ambiguous_regions_are_unknown=True)
    out=tmp_path/'refinement-request.json';out.write_text(json.dumps(child))
    return out,child


def test_object_ambiguous_landmark_loses_authority_without_moving_other_pixels():
    points=np.zeros((21,3));points[:5]=[[60,25,.5],[65,25,.5],[65,30,.5],[60,30,.5],[79,32,.5]]
    exclusion=np.zeros((64,128),bool);exclusion[25:41,73:101]=True
    before=points.copy();kept,coords,labels=recovery.mask_prompts(points,exclusion)
    np.testing.assert_array_equal(points,before)
    np.testing.assert_array_equal(kept[:4],points[:4])
    assert kept[4,2]==0 and np.count_nonzero(kept[:,2])==4
    assert np.count_nonzero(labels==1)==4 and np.count_nonzero(labels==0)>0
    assert all(exclusion[int(y),int(x)] for x,y in coords[labels==0])


def test_refinement_reuses_verified_pose_and_real_merger_preserves_unknown_flags(bound_request,tmp_path):
    original_path,_=bound_request
    parent=fake_inference(bound_request,tmp_path/'parent')
    path,data=child_request(bound_request,tmp_path)
    reused,_=recovery.validate_pose_reuse(path,original_path,parent)
    output=tmp_path/'refined';output.mkdir()
    receipt=dict(schema='v2d.source-hand-recovery-result.v1',request_sha256=recovery.digest(path),canonical_manifest_sha256=data['canonical_manifest_sha256'],rows=[],complete=True,original_observation_flags_changed=False,reference_accepted=False,training_approved=False,hamer_inference_reused=True)
    def masks(image,points,box,point_labels):
        assert (point_labels==0).any() and (point_labels==1).sum()==4
        return np.ones((1,*image.shape[:2]),bool),np.array([.9])
    def unexpected_model(*args):raise AssertionError('The unchanged pose must not be inferred again')
    recovery.infer_frames(data,output,recovery.digest(path),masks,unexpected_model,receipt,reused)
    receipt['files']=recovery.output_files(output);(output/'receipt.json').write_text(json.dumps(receipt))
    prior=json.loads((parent/'records/right/000005.json').read_text());current=json.loads((output/'records/right/000005.json').read_text())
    for field in ('mano','camera','cam_t','hand_scale','image_size','crop'):assert current[field]==prior[field]
    assert current['pose_source_request_sha256']==recovery.digest(original_path)
    assert current['source_request_sha256']==recovery.digest(path)
    excluded=recovery.exclusion_mask(data['frames'][0],(64,128))
    mask=np.asarray(Image.open(output/'image-observations/right/000005-mask.png'))>0
    assert not mask[excluded].any()
    report=merger.merge(path,output,tmp_path/'handoff')
    assert not report['training_approved'] and not report['original_observation_flags_changed']
    assert report['reference_contact_gate_tolerance_m']==.001


def test_exclusion_with_too_few_remaining_points_abstains_before_model_calls(bound_request,tmp_path):
    exclusion=np.zeros((64,128),bool);exclusion[22:28,57:63]=True
    path,data=child_request(bound_request,tmp_path,exclusion)
    out=tmp_path/'rejected';out.mkdir();receipt={'rows':[]}
    def never(*args,**kwargs):raise AssertionError('Insufficient source support must abstain')
    recovery.infer_frames(data,out,recovery.digest(path),never,never,receipt)
    assert receipt['rows'][0]['status']=='rejected_mask_support'
    assert not recovery.output_files(out)


def test_pose_reuse_rejects_changed_crop_or_tampered_parent(bound_request,tmp_path):
    original_path,_=bound_request;parent=fake_inference(bound_request,tmp_path/'parent');path,data=child_request(bound_request,tmp_path)
    data['frames'][0]['crop']['cx']+=1;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='identical source'):recovery.validate_pose_reuse(path,original_path,parent)
    data['frames'][0]['crop']['cx']-=1;path.write_text(json.dumps(data))
    p=parent/'records/right/000005.json';r=json.loads(p.read_text());r['cam_t'][0]+=.1;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='output changed'):recovery.validate_pose_reuse(path,original_path,parent)


@pytest.mark.parametrize('field,value',[('ambiguous_regions_are_unknown',False),('source_image_sha256','foreign'),('sha256','changed')])
def test_exclusion_cannot_create_foreign_or_measured_authority(bound_request,tmp_path,field,value):
    path,data=child_request(bound_request,tmp_path);data['frames'][0]['inferred_exclusion'][field]=value;path.write_text(json.dumps(data))
    with pytest.raises(ValueError):recovery.validate_request(path)


def test_exclusion_requires_the_same_calibrated_image_grid(bound_request,tmp_path):
    path,data=child_request(bound_request,tmp_path);control=data['frames'][0]['inferred_exclusion'];p=Path(control['path']);Image.new('L',(64,32),255).save(p);control['sha256']=recovery.digest(p);path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='calibrated source grid'):recovery.validate_request(path)

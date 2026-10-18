"""CPU file-protocol integration with fake models; no pose-quality claim."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0,str(ROOT))
import recover_source_hand as recovery
import merge_recovered_source_hand as merger
from issue6.native_priors import load_native_priors
from issue6.image_observations import load_image_estimate

spec=importlib.util.spec_from_file_location('source_request_fixture',Path(__file__).with_name('test_source_hand_recovery.py'))
fixtures=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
recovery_request=fixtures.recovery_request


@pytest.fixture
def bound_request(recovery_request):
    path,data=recovery_request
    previous_bundle=Path(data['canonical_bundle'])
    with np.load(previous_bundle) as z:arrays={k:z[k].copy() for k in z.files}
    bundle=previous_bundle.parent/'canonical'/previous_bundle.name
    bundle.parent.mkdir()
    data['canonical_bundle']=str(bundle)
    arrays.update(hand_valid=np.array([[True,False]]),hand_metric_valid=np.array([[True,False]]))
    np.savez(bundle,**arrays)
    data['canonical_bundle_sha256']=recovery.digest(bundle)
    sources=[]
    for side,right,track in [('left',False,2),('right',True,3)]:
        p=bundle.parent/f'source/hands/{side}/000005.json';p.parent.mkdir(parents=True)
        old=dict(frame_idx=5,track_id=track,is_right=right,directly_observed=not right,
            mano=dict(betas=[0.]*10,global_orient=[0.]*3,hand_pose=[0.]*45),cam_t=[.2,.1,1.],hand_scale=.9)
        p.write_text(json.dumps(old));sources.append(dict(path=str(p.relative_to(bundle.parent)),sha256=recovery.digest(p)))
    manifest=dict(schema='v2d.track3.episode.v1',payload=dict(path=bundle.name,sha256=recovery.digest(bundle)),sources=sources)
    (bundle.parent/'manifest.json').write_text(json.dumps(manifest))
    data['canonical_manifest_sha256']=recovery.digest(bundle.parent/'manifest.json')
    path.write_text(json.dumps(data))
    return path,data


def fake_inference(bound_request,folder,*,mask_valid=True):
    path,data=bound_request
    recovery.canonical_sources(recovery.validate_request(path))
    folder.mkdir()
    receipt=dict(schema='v2d.source-hand-recovery-result.v1',request_sha256=recovery.digest(path),canonical_manifest_sha256=data['canonical_manifest_sha256'],
        rows=[],complete=True,original_observation_flags_changed=False,reference_accepted=False,training_approved=False,fixture_models_only=True)
    def masks(image,points,box):
        assert len(points)==4
        mask=np.full(image.shape[:2],mask_valid,bool)
        return np.array([mask]),np.array([.9])
    def hand(image,crop,right):
        assert right and mask_valid
        return dict(hand_pose=np.tile(np.eye(3),(15,1,1)),global_orient=np.eye(3),betas=np.ones(10),pred_cam_t_full=[.1,.05,2.],scaled_focal_length=100.)
    recovery.infer_frames(data,folder,recovery.digest(path),masks,hand,receipt)
    receipt['files']=recovery.output_files(folder)
    (folder/'receipt.json').write_text(json.dumps(receipt))
    return folder


def rebind_recovery(folder):
    p=folder/'receipt.json';r=json.loads(p.read_text());r['files']=recovery.output_files(folder);p.write_text(json.dumps(r))


def test_merge_uses_real_loaders_and_preserves_source_flags_and_other_hand(bound_request,tmp_path):
    path,data=bound_request
    original=Path(data['canonical_bundle']).parent/'source/hands'
    before={str(p):p.read_bytes() for p in original.rglob('*.json')}
    bundle_before=Path(data['canonical_bundle']).read_bytes()
    fresh=fake_inference(bound_request,tmp_path/'fake-gpu-output')
    out=tmp_path/'merged';report=merger.merge(path,fresh,out)
    assert all(Path(p).read_bytes()==v for p,v in before.items())
    assert Path(data['canonical_bundle']).read_bytes()==bundle_before
    assert (out/'priors/left/000005.json').read_bytes()==(original/'left/000005.json').read_bytes()
    priors=load_native_priors(out/'priors',[dict(id=999)],source_frame_indices=True)
    assert list(priors['right'])==[5] and list(priors['left'])==[5]
    assert not priors['right'][5]['metric_valid'] and not priors['right'][5]['directly_observed']
    image=load_image_estimate(out/'image-observations','right',5,data['camera'],False)
    assert np.count_nonzero(np.array(image['keypoints'])[:,2])==4
    with np.load(out/'original-source-confidence.npz') as z,np.load(data['canonical_bundle']) as original_arrays:
        for field in ['hand_valid','hand_directly_observed','hand_metric_valid']:
            np.testing.assert_array_equal(z[field],original_arrays[field])
    assert report['reference_contact_gate_tolerance_m']==.001
    assert report['reference_contact_gate_passed'] is None and not report['training_approved']
    assert not report['carry_reference_accepted'] and not report['geometry_fitted']
    with pytest.raises(ValueError,match='fresh output'):merger.merge(path,fresh,out)


def test_failed_fresh_mask_cannot_be_staged_as_a_recovery(bound_request,tmp_path):
    path,_=bound_request;fresh=fake_inference(bound_request,tmp_path/'fresh',mask_valid=False)
    assert not (fresh/'records').exists()
    with pytest.raises(ValueError,match='No recovered'):merger.merge(path,fresh,tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('field,value',[('complete',False),('request_sha256','wrong'),('training_approved',True)])
def test_incomplete_or_foreign_recovery_cannot_write_output(bound_request,tmp_path,field,value):
    path,_=bound_request;fresh=fake_inference(bound_request,tmp_path/'fresh');p=fresh/'receipt.json';r=json.loads(p.read_text());r[field]=value;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='receipt'):merger.merge(path,fresh,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_changed_output_and_rebound_false_authority_are_both_rejected(bound_request,tmp_path):
    path,_=bound_request;fresh=fake_inference(bound_request,tmp_path/'fresh');p=fresh/'records/right/000005.json';r=json.loads(p.read_text());r['directly_observed']=True;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='output changed'):merger.merge(path,fresh,tmp_path/'out')
    rebind_recovery(fresh)
    with pytest.raises(ValueError,match='authority'):merger.merge(path,fresh,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_fresh_full_hand_cannot_invent_an_unobserved_image_joint(bound_request,tmp_path):
    path,_=bound_request;fresh=fake_inference(bound_request,tmp_path/'fresh');p=fresh/'image-observations/right/000005.json';r=json.loads(p.read_text());r['keypoints'][10]=[63,27,.5];p.write_text(json.dumps(r));rebind_recovery(fresh)
    with pytest.raises(ValueError,match='fabricated'):merger.merge(path,fresh,tmp_path/'out')


def test_canonical_hand_prior_is_hash_bound_and_supported_frames_are_protected(bound_request,tmp_path):
    path,data=bound_request;canonical=Path(data['canonical_bundle']).parent/'source/hands/right/000005.json';old=canonical.read_bytes();canonical.write_text('{}')
    with pytest.raises(ValueError,match='prior changed'):recovery.canonical_sources(data)
    canonical.write_bytes(old)
    bundle=Path(data['canonical_bundle'])
    with np.load(bundle) as z:arrays={k:z[k].copy() for k in z.files}
    arrays['hand_metric_valid'][0,1]=True;np.savez(bundle,**arrays)
    data['canonical_bundle_sha256']=recovery.digest(bundle);mp=bundle.parent/'manifest.json';mf=json.loads(mp.read_text());mf['payload']['sha256']=data['canonical_bundle_sha256'];mp.write_text(json.dumps(mf));data['canonical_manifest_sha256']=recovery.digest(mp);path.write_text(json.dumps(data))
    fresh=fake_inference((path,data),tmp_path/'fresh')
    with pytest.raises(ValueError,match='supported canonical'):merger.merge(path,fresh,tmp_path/'out')


def test_gpu_setup_failure_still_saves_an_incomplete_receipt(bound_request,tmp_path,monkeypatch):
    path,_=bound_request;out=tmp_path/'failed-setup'
    monkeypatch.setitem(sys.modules,'torch',None)
    with pytest.raises(ModuleNotFoundError):recovery.run(path,out,None,'unused',None)
    receipt=json.loads((out/'receipt.json').read_text())
    assert not receipt['complete'] and not receipt['reference_accepted']
    assert not receipt['training_approved'] and receipt['files']=={}

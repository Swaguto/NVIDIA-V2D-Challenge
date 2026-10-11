"""Independent disk/video audit of a completed continuous Dex3 renderer run."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from continuous_dex3 import digest, verify


def audit(a):
    if (a.replay/'failure.json').exists():
        raise ValueError('Replay has a failure receipt')
    receipt=json.loads((a.replay/'complete.json').read_text())
    display, method=verify(a.retarget,a.presentation)
    bundle_manifest=json.loads((a.bundle/'manifest.json').read_text())
    retarget_manifest=json.loads((a.retarget/'manifest.json').read_text())
    if digest(a.bundle/'manifest.json')!=retarget_manifest['episode_bundle_manifest_sha256']:
        raise ValueError('Bundle and retarget differ')
    if digest(a.bundle/'episode.npz')!=retarget_manifest['episode_bundle_payload_sha256']:
        raise ValueError('Bundle payload changed')
    if digest(a.replay/'replay.mp4')!=receipt['video_sha256'] or digest(a.replay/'simulator-readback.npz')!=receipt['readback_sha256']:
        raise ValueError('Replay artifact hash differs')
    if digest(a.presentation/'manifest.json')!=receipt['presentation_manifest_sha256']:
        raise ValueError('Presentation receipt differs')
    frames=display['frame_indices']; counts=[]
    if sorted(p.name for p in (a.replay/'frames').glob('*.png')) != [f'{int(frame):06d}.png' for frame in frames]:
        raise ValueError('Missing or extra rendered frames')
    for frame in frames:
        rgb=np.asarray(Image.open(a.replay/'frames'/f'{int(frame):06d}.png').convert('RGB')).astype(float)
        blue=(rgb[...,2]>1.5*rgb[...,0]) & (rgb[...,2]>1.3*rgb[...,1]) & (rgb[...,2]>40)
        orange=(rgb[...,0]>1.5*rgb[...,2]) & (rgb[...,0]>1.2*rgb[...,1]) & (rgb[...,0]>40)
        counts.append([int(blue.sum()),int(orange.sum())])
    counts=np.array(counts)
    if np.any(counts<receipt['minimum_hand_pixels_required']):
        raise ValueError('Rendered hand pixels missing')
    geometry=json.loads((a.replay/'imported-visual-geometry.json').read_text())
    with np.load(a.replay/'simulator-readback.npz') as readback:
        for key in ('frame_indices','display_valid','held','pose_source_frames'):
            if not np.array_equal(readback[key],display[key]):raise ValueError('Readback contract differs: '+key)
        if not np.array_equal(readback['robot_valid'],display['source_robot_valid']):raise ValueError('Missing support promoted')
        if not np.array_equal(readback['hand_review_color_pixels'],counts):raise ValueError('Disk render differs from pixel receipt')
        for s,side in enumerate(('left','right')):
            if not np.all(readback['visible_mesh_counts'][:,s]==len(geometry[side])):raise ValueError('Hidden mesh')
        errors={}
        actual=readback['robot_qpos']; expected=display['robot_qpos']
        if actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError('Bad qpos readback')
        errors['independent_root_position_m']=float(np.max(np.linalg.norm(actual[:,:,:3]-expected[:,:,:3],axis=-1)))
        errors['independent_joint_angle_rad']=float(np.max(np.abs(actual[:,:,7:]-expected[:,:,7:])))
        errors['independent_root_rotation_rad']=float(np.max((Rotation.from_quat(actual[:,:,3:7].reshape(-1,4)).inv()*Rotation.from_quat(expected[:,:,3:7].reshape(-1,4))).magnitude()))
        if any(v>1e-4 for v in errors.values()):raise ValueError('Actual state differs from presentation target')
        with np.load(a.bundle/'episode.npz') as bundle:
            if not np.array_equal(bundle['hand_valid'],display['source_robot_valid']) or not np.array_equal(bundle['frame_indices'],frames):
                raise ValueError('Original source support/timeline changed')
            world_to_stage=np.asarray(receipt['world_to_stage'])
            for obj_id,obj in enumerate(bundle_manifest['objects']):
                index=obj['index']
                expected_object=np.einsum('ij,tjk->tik',world_to_stage,bundle[f'object_{index}_world'])
                expected_object[:,:3,:3]*=bundle[f'object_{index}_scale'][:,None,None]
                actual_object=readback['object_world_transform'][:,obj_id]
                delta=float(np.max(np.abs(expected_object-actual_object)))
                if not np.isfinite(actual_object).all() or delta>1e-6:raise ValueError('Object transform mismatch')
                errors[f'object_{index}_transform_max_error']=delta
            if not readback['object_visible'].all():raise ValueError('Object mesh hidden')
        for key in ('root_position_error_m','root_orientation_error_rad','joint_error_rad','usd_link_position_error_m','physics_link_position_error_m'):
            value=readback[key]
            if not np.isfinite(value).all() or np.max(value)>1e-4:raise ValueError('Bad simulator readback: '+key)
            errors[key]=float(np.max(value))
    decoded=0
    with imageio.get_reader(a.replay/'replay.mp4') as reader:
        for rgb in reader:
            if np.std(rgb)<1:raise ValueError('Blank encoded video frame')
            decoded+=1
    if decoded!=len(frames):raise ValueError('Incomplete encoded timeline')
    report=dict(status='passed model continuity and simulator target audit',frames=decoded,
                per_hand_minimum_pixels=counts.min(axis=0).tolist(),source_counts=method['observed_counts'],
                held_counts=method['held_counts'],display_visible_counts=[decoded,decoded],
                errors=errors,objects_visible_every_frame=True,pose_accuracy_accepted=False,contact_or_training_targets=False,
                scope='Model presence and target playback; does not establish hand accuracy or absence of motion jitter',
                auditor_sha256=digest(Path(__file__)))
    destination=a.replay/'independent-audit.json'
    if destination.exists():raise FileExistsError(destination)
    destination.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bundle','retarget','presentation','replay'):p.add_argument('--'+key,type=Path,required=True)
    print(json.dumps(audit(p.parse_args()),indent=2))

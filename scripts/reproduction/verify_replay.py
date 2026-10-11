"""Independent numerical and decoded-video audit for a completed replay."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def audit(bundle,retarget,out):
    bundle,retarget,out=map(Path,(bundle,retarget,out))
    r=json.loads((out/'complete.json').read_text())
    if (out/'failure.json').exists():
        raise ValueError('Replay failure marker present')
    with np.load(bundle/'episode.npz') as source,np.load(retarget/'retarget.npz') as target,np.load(out/'simulator-readback.npz') as actual:
        valid=target['robot_valid'];n=len(source['frame_indices'])
        if not np.array_equal(valid,source['hand_valid']):
            raise ValueError('Source support mask differs from robot mask')
        if not np.array_equal(target['frame_indices'],source['frame_indices']):
            raise ValueError('Source frame IDs differ from robot frames')
        if r['source_frames']!=n or r['decoded_video_frames']!=n:
            raise ValueError('Incomplete recording')
        for key in ('frame_indices','robot_valid'):
            if not np.array_equal(actual[key],target[key]):
                raise ValueError(f'Readback schema mismatch {key}')
        q,expected=actual['robot_qpos'],target['robot_qpos']
        if q.shape!=expected.shape or expected.shape!=(n,2,14):
            raise ValueError('Joint state schema differs')
        if not np.isfinite(expected[valid]).all() or not np.isnan(expected[~valid]).all():
            raise ValueError('Invalid reference joint states')
        if not np.isnan(q[~valid]).all() or not np.isfinite(q[valid]).all():
            raise ValueError('Unknown/finite state mismatch')
        errors={
            'root_position_m':float(np.linalg.norm(q[valid,:3]-expected[valid,:3],axis=1).max()),
            'joint_angle_rad':float(np.max(np.abs(q[valid,7:]-expected[valid,7:]))),
            'root_rotation_rad':float((Rotation.from_quat(q[valid,3:7]).inv()*Rotation.from_quat(expected[valid,3:7])).magnitude().max())}
        for key in ('usd_link_position_error_m','physics_link_position_error_m'):
            if not np.isfinite(actual[key][valid]).all() or not np.isnan(actual[key][~valid]).all():
                raise ValueError(f'Invalid link mask {key}')
            errors[key]=float(actual[key][valid].max())
        if any(v>1e-4 for v in errors.values()):
            raise ValueError(f'Readback tolerance exceeded: {errors}')
        if actual['hand_review_color_pixels'][~valid].sum()!=0:
            raise ValueError('Unknown hands visible')
        if sha(out/'replay.mp4')!=r['video_sha256'] or sha(out/'simulator-readback.npz')!=r['readback_sha256']:
            raise ValueError('Receipt hashes differ')
        import imageio.v2 as imageio
        reader=imageio.get_reader(out/'replay.mp4')
        decoded=0
        for frame in reader:
            if frame.std()<1:
                raise ValueError(f'Blank video frame {decoded}')
            decoded+=1
        reader.close()
        if decoded!=n or len(list((out/'frames').glob('*.png')))!=n:
            raise ValueError('Video/PNG frame coverage differs')
        return {'source_frames':n,'decoded_frames':decoded,'supported_states':int(valid.sum()),'max_errors':errors,'tolerance':1e-4,'status':'passed','physical_manipulation_test':False}

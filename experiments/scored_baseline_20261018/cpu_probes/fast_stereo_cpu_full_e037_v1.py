"""Full source-clock CPU depth hypothesis; reuse the seven verified samples.

Only pretrained depth inference runs. These are estimated observations in a
separate private folder, not a promoted reconstruction, policy or physics run.
"""
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np

S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(S/'cpu-stereo-deps-v1'),str(R/'reconstruction/modules/v2d_stereo_depth/lib')]
import onnxruntime as ort
from stereo_depth import load_rig

OUT=S/'fast-stereo-cpu-full-e037-v1';OUT.mkdir(exist_ok=True)
I=S/'learned-stereo-inputs-v1';prior=S/'fast-stereo-cpu-seven-v1'
model=S/'fast-stereo-model-cache-v1/c_fast_foundationstereo_bp2_dynbatch_dynhw.onnx'
calib=S/'dataset/track_3/public/meta/camera_calibration.json'
bundle=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta=json.loads((I/'manifest.json').read_text());old=json.loads((prior/'depth-report.json').read_text())
assert old['all_seven_complete'] and old['synthetic_shift_control']['passed']
assert sha(model)==old['model_sha256']=='2c79bbb274dc0aef687a8732b987077e9e3aa26ff9dd653141cda48c57dad5d3'
assert old['source_manifest_sha256']==sha(I/'manifest.json')
for name,proof in meta['files'].items():assert sha(I/name)==proof['sha256']
with np.load(bundle,allow_pickle=False) as z:
    axis=z['frame_indices'].copy();timestamps=z['timestamps_s'].copy()
    assert axis.tolist()==list(range(151));np.testing.assert_allclose(np.diff(timestamps),.05,atol=1e-8,rtol=0)
rig=load_rig(str(calib));np.testing.assert_allclose(rig.P1[:3,:3],meta['rectified_intrinsics'],atol=1e-6,rtol=0)
videos=[S/f'dataset/track_3/evaluation/videos/chunk-000/observation.images.{cam}/episode_000037.mp4' for cam in [rig.left,rig.right]]
source_hashes={str(p.relative_to(S)):sha(p) for p in [*videos,calib,bundle]}
caps=[cv2.VideoCapture(str(p)) for p in videos]
for cap in caps:assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==151 and abs(cap.get(cv2.CAP_PROP_FPS)-20)<1e-8
report=dict(schema='v2d.fast-stereo-private-full-cpu-depth.v1',episode=37,source_frames=151,complete=[],
    producer_sha256=sha(Path(__file__)),source_manifest_sha256=sha(I/'manifest.json'),source_files_sha256=source_hashes,
    model_sha256=sha(model),model_revision=old['model_revision'],previous_seven_report_sha256=sha(prior/'depth-report.json'),
    runtime=dict(onnxruntime=ort.__version__,numpy=np.__version__,cpu_threads=2),
    synthetic_shift_control=old['synthetic_shift_control'],input_resolution=[320,512],
    source_inputs_modified=False,source_confidence_modified=False,source_images_uploaded=False,
    model_depths_are_estimates=True,hidden_reference_labels_read=False,training_steps=0,
    trajectory_promoted=False,physical_task_accepted=False,all_frames_complete=False)
save=lambda:(OUT/'depth-report.json').write_text(json.dumps(report,indent=2)+'\n')
started=time.monotonic();save()
try:
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.enable_cpu_mem_arena=False;options.enable_mem_pattern=False
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.use_deterministic_compute=True;options.log_severity_level=3
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    assert session.get_providers()==['CPUExecutionProvider']
    report['providers']=session.get_providers();save()
    def tensor(image):
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
        rgb=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        return np.ascontiguousarray(rgb.transpose(2,0,1)[None])
    def predict(left,right):
        value=np.asarray(session.run(['disparity'],{'left_image':tensor(left),'right_image':tensor(right)})[0]).squeeze()
        assert value.shape==(320,512) and np.isfinite(value).all()
        return value
    for frame in axis:
        frame=int(frame);assert time.monotonic()-started<2400,'Forty-minute CPU worker cap'
        raw=[]
        for cap in caps:
            ok,image=cap.read();assert ok;raw.append(image)
        full_left,full_right=rig.rectify_pair(*raw)
        assert full_left.shape==full_right.shape==(800,1280,3)
        reused=frame in meta['frames'];tick=time.monotonic()
        if reused:
            for label,image in [('left',full_left),('right',full_right)]:np.testing.assert_array_equal(image,cv2.imread(str(I/f'{frame:03d}-{label}.png')))
            src=prior/f'{frame:03d}-learned.npz';expected=next(v['output_sha256'] for v in old['complete'] if v['frame']==frame)
            assert sha(src)==expected
            with np.load(src,allow_pickle=False) as z:depth,valid=z['depth_m'].copy(),z['left_right_consistent'].copy()
        else:
            left,right=[cv2.resize(image,(512,320),interpolation=cv2.INTER_AREA) for image in [full_left,full_right]]
            forward=predict(left,right)
            reverse=predict(np.ascontiguousarray(right[:,::-1]),np.ascontiguousarray(left[:,::-1]))[:,::-1].copy()
            dl=cv2.resize(forward/.4,(1280,800),interpolation=cv2.INTER_LINEAR).astype(np.float32)
            dr=cv2.resize(reverse/.4,(1280,800),interpolation=cv2.INTER_LINEAR).astype(np.float32)
            yy,xx=np.indices(dl.shape,dtype=np.float32);rx=xx-dl
            sample=cv2.remap(dr,rx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            valid=(rx>=0)&(rx<1279)&(dl>.01)&(sample>.01)&(abs(dl-sample)<=1.)
            depth=np.divide(meta['rectified_intrinsics'][0][0]*meta['baseline_m'],dl,out=np.zeros_like(dl),where=dl>.01)
            valid &= np.isfinite(depth)&(depth>.05)&(depth<3.)
        path=OUT/f'{frame:06d}-depth.npz'
        np.savez_compressed(path,depth_m=depth,left_right_consistent=valid,frame_index=frame,timestamp_s=timestamps[frame])
        report['complete'].append(dict(frame=frame,reused_exact_verified_sample=reused,processing_seconds=time.monotonic()-tick,valid_pixels=int(valid.sum()),path=path.name,sha256=sha(path)))
        save();print('Saved source frame',frame,'reused',reused,'seconds',time.monotonic()-tick,flush=True)
    for cap in caps:ok,_=cap.read();assert not ok
    for name,h in source_hashes.items():assert sha(S/name)==h
    for name,proof in meta['files'].items():assert sha(I/name)==proof['sha256']
    assert [r['frame'] for r in report['complete']]==axis.tolist()
    report['frozen_source_closure_verified_after']=True;report['all_frames_complete']=True
except BaseException as error:report['error']=repr(error);raise
finally:
    for cap in caps:cap.release()
    report['wall_seconds']=time.monotonic()-started;report['maximum_resident_bytes_darwin']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save()

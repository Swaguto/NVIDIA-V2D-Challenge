"""Bounded CPU pretrained stereo check: synthetic geometry, then e037 frame125.

This is a different, lighter model from the unexecuted full FoundationStereo
GPU probe. Source images, flags and selected references stay unchanged.
"""
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np

S=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(S/'cpu-stereo-deps-v1'))
import onnxruntime as ort

OUT=S/'fast-stereo-cpu-probe-v1';OUT.mkdir(exist_ok=True)
inputs=S/'learned-stereo-inputs-v1'
model=S/'fast-stereo-model-cache-v1/c_fast_foundationstereo_bp2_dynbatch_dynhw.onnx'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta=json.loads((inputs/'manifest.json').read_text())
for name,proof in meta['files'].items():assert sha(inputs/name)==proof['sha256']
assert sha(model)=='2c79bbb274dc0aef687a8732b987077e9e3aa26ff9dd653141cda48c57dad5d3'
report=dict(schema='v2d.fast-stereo-private-cpu-probe.v1',episode=37,frame=125,
    producer_sha256=sha(Path(__file__)),source_manifest_sha256=sha(inputs/'manifest.json'),
    model_sha256=sha(model),model_revision='53a84e28a368f37e699776250215fcb450d6a109',
    model_source='https://huggingface.co/nvidia/c-fast-foundationstereo',
    preprocessing_source='https://huggingface.co/nvidia/c-fast-foundationstereo/discussions/2',
    runtime=dict(onnxruntime=ort.__version__,numpy=np.__version__,cpu_threads=2),
    source_inputs_modified=False,source_confidence_modified=False,source_images_uploaded=False,
    model_depths_are_estimates=True,hidden_reference_labels_read=False,training_steps=0,
    trajectory_promoted=False,physical_task_accepted=False,frame_complete=False)
save=lambda:(OUT/'depth-report.json').write_text(json.dumps(report,indent=2)+'\n')
started=time.monotonic();save()
try:
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.enable_cpu_mem_arena=False;options.enable_mem_pattern=False
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.use_deterministic_compute=True;options.log_severity_level=3
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    assert session.get_providers()==['CPUExecutionProvider']
    report['providers']=session.get_providers()
    report['input_contract']=[dict(name=x.name,shape=x.shape,type=x.type) for x in session.get_inputs()]
    assert {v.name for v in session.get_inputs()}=={'left_image','right_image'}
    assert session.get_outputs()[0].name=='disparity'
    report['stage']='runtime_ready';save();print('CPU runtime ready',flush=True)
    def tensor(image):
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
        rgb=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        return np.ascontiguousarray(rgb.transpose(2,0,1)[None])
    def predict(left,right):
        assert left.shape==right.shape and left.shape[0]%32==left.shape[1]%32==0
        t=time.monotonic()
        value=np.asarray(session.run(['disparity'],{'left_image':tensor(left),'right_image':tensor(right)})[0]).squeeze()
        assert value.shape==left.shape[:2] and np.isfinite(value).all()
        return value,time.monotonic()-t
    rng=np.random.default_rng(20261018)
    texture=cv2.resize(rng.integers(0,256,(64,80,3),dtype=np.uint8),(320,256),interpolation=cv2.INTER_CUBIC)
    right=texture.copy();right[:,:-16]=texture[:,16:];right[:,-16:]=0
    synthetic,wall=predict(texture,right)
    center=synthetic[48:-48,64:-64]
    report['synthetic_shift_control']=dict(expected_disparity_px=16.,median_disparity_px=float(np.median(center)),median_absolute_error_px=float(np.median(abs(center-16))),inference_seconds=wall,seed=20261018,input_shape=list(texture.shape))
    assert abs(np.median(center)-16)<1. and np.median(abs(center-16))<1.,'Synthetic positive-parallax/normalization check failed'
    report['synthetic_shift_control']['passed']=True;report['stage']='synthetic_control_passed';save();print(json.dumps(report['synthetic_shift_control']),flush=True)
    full_left,full_right=[cv2.imread(str(inputs/f'125-{side}.png')) for side in ['left','right']]
    assert full_left.shape[:2]==(800,1280)
    height,width=320,512
    left,right=[cv2.resize(im,(width,height),interpolation=cv2.INTER_AREA) for im in [full_left,full_right]]
    forward,wall1=predict(left,right)
    report['stage']='source_forward_complete';report['forward_seconds']=wall1;save();print('Source forward complete',wall1,flush=True)
    reverse,wall2=predict(np.ascontiguousarray(right[:,::-1]),np.ascontiguousarray(left[:,::-1]))
    reverse=reverse[:,::-1].copy()
    dl=cv2.resize(forward/(width/full_left.shape[1]),(1280,800),interpolation=cv2.INTER_LINEAR).astype(np.float32)
    dr=cv2.resize(reverse/(width/full_left.shape[1]),(1280,800),interpolation=cv2.INTER_LINEAR).astype(np.float32)
    yy,xx=np.indices(dl.shape,dtype=np.float32);rx=xx-dl
    sampled=cv2.remap(dr,rx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    consistency=(rx>=0)&(rx<1279)&(dl>.01)&(sampled>.01)&(abs(dl-sampled)<=1.)
    factor=meta['rectified_intrinsics'][0][0]*meta['baseline_m']
    depth=np.divide(factor,dl,out=np.zeros_like(dl),where=dl>.01)
    consistency &= np.isfinite(depth)&(depth>.05)&(depth<3.)
    np.savez_compressed(OUT/'125-learned.npz',depth_m=depth,left_disparity_px=dl,right_disparity_px=dr,left_right_consistent=consistency)
    report['stage']='complete';report['frame_complete']=True
    report['source_inference']=dict(input_height=height,input_width=width,actual_horizontal_scale=width/1280,forward_seconds=wall1,reverse_seconds=wall2,left_right_consistent_pixels=int(consistency.sum()),output_sha256=sha(OUT/'125-learned.npz'))
    for name,proof in meta['files'].items():assert sha(inputs/name)==proof['sha256']
    report['frozen_source_closure_verified_after']=True
except BaseException as error:
    report['error']=repr(error);raise
finally:
    report['wall_seconds']=time.monotonic()-started
    report['maximum_resident_bytes_darwin']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save()

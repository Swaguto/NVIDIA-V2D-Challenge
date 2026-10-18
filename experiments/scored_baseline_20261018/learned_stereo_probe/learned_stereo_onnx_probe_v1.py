"""Seven fixed private stereo pairs, pretrained inference only; no PPO.

Preprocessing follows the repository NVIDIA FoundationStereo wrapper. The
disparity rescale uses the actual integer resize width, so metric parallax is
correct even when aspect-preserving dimensions are rounded.
"""
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path('/job')
OUT = ROOT/'outputs'
OUT.mkdir(exist_ok=True)
source = ROOT/'inputs'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
meta = json.loads((source/'manifest.json').read_text())
for name, proof in meta['files'].items():
    p = source/name
    assert p.stat().st_size == proof['size'] and sha(p) == proof['sha256']
sys.path.insert(0, str(ROOT/'deps'))
import onnxruntime as ort

report = dict(schema='v2d.pretrained-stereo-source-diagnostic.v1', episode=37,
              frames=meta['frames'], complete=[], training_steps=0,
              source_manifest_sha256=sha(source/'manifest.json'),
              producer_sha256=sha(Path(__file__)), source_inputs_modified=False,
              reference_labels_read=False, trajectory_promoted=False,
              model_depths_are_estimates=True, physical_task_accepted=False,
              runtime=dict(onnxruntime=ort.__version__, torch=torch.__version__,
                           cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version()))
save = lambda: (OUT/'depth-report.json').write_text(json.dumps(report, indent=2)+'\n')
save()
try:
    filename = 'deployable_foundationstereo_small_576x960_v2.0.onnx'
    url = 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/foundationstereo/deployable_v2.0/files?redirect=true&path='+filename
    model = ROOT/'models'/filename
    model.parent.mkdir(exist_ok=True)
    if not model.exists():
        partial = model.with_suffix('.partial')
        total = 0
        with urllib.request.urlopen(url, timeout=45) as response, partial.open('wb') as target:
            while chunk := response.read(1024*1024):
                target.write(chunk)
                total += len(chunk)
        assert total == 909161154, 'Unexpected versioned NGC model length'
        partial.rename(model)
    assert model.stat().st_size == 909161154
    report['model'] = dict(url=url, bytes=model.stat().st_size, sha256=sha(model))
    save()
    print('Model downloaded and hashed', flush=True)
    if hasattr(ort, 'preload_dlls'):
        ort.preload_dlls()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(str(model), sess_options=options,
            providers=[('CUDAExecutionProvider', dict(device_id=0, cudnn_conv_algo_search='DEFAULT'))])
    assert 'CUDAExecutionProvider' in session.get_providers(), 'CUDA provider unavailable; CPU fallback refused'
    session.disable_fallback()
    report['providers'] = session.get_providers()
    report['inputs'] = [dict(name=x.name, shape=x.shape, type=x.type) for x in session.get_inputs()]
    report['outputs'] = [dict(name=x.name, shape=x.shape, type=x.type) for x in session.get_outputs()]
    assert {x.name for x in session.get_inputs()} == {'left_image', 'right_image'}
    assert session.get_outputs()[0].name == 'disparity'
    save()

    def preprocess(image):
        h, w = image.shape[:2]
        ratio = min(960/w, 576/h)
        nw, nh = int(w*ratio), int(h*ratio)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if ratio <= 1 else cv2.INTER_CUBIC)
        padded = cv2.copyMakeBorder(resized, 0, 576-nh, 0, 960-nw, cv2.BORDER_REPLICATE)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb-np.array([123.675,116.28,103.53], np.float32))/np.array([58.395,57.12,57.375], np.float32)
        return np.ascontiguousarray(rgb.transpose(2,0,1)[None]), (h,w,nh,nw)

    def infer(left, right):
        li, geometry = preprocess(left)
        ri, other = preprocess(right)
        assert geometry == other
        disp = np.asarray(session.run(['disparity'], {'left_image':li, 'right_image':ri})[0]).squeeze()
        assert disp.shape == (576,960) and np.isfinite(disp).all()
        h,w,nh,nw = geometry
        return cv2.resize(disp[:nh,:nw]/(nw/w), (w,h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    for frame in meta['frames']:
        start = time.monotonic()
        left, right = [cv2.imread(str(source/f'{frame:03d}-{side}.png')) for side in ['left','right']]
        dl = infer(left, right)
        # Mirroring the reversed pair preserves the model's positive-parallax convention.
        dr = np.fliplr(infer(np.ascontiguousarray(np.fliplr(right)), np.ascontiguousarray(np.fliplr(left)))).copy()
        yy, xx = np.indices(dl.shape, dtype=np.float32)
        rx = xx-dl
        sampled = cv2.remap(dr, rx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        consistency = (rx >= 0) & (rx <= dl.shape[1]-1) & (dl > .01) & (sampled > .01) & (abs(dl-sampled) <= 1)
        depth = np.divide(meta['rectified_intrinsics'][0][0]*meta['baseline_m'], dl,
                          out=np.zeros_like(dl), where=(dl > .01))
        consistency &= np.isfinite(depth) & (depth > .05) & (depth < 3)
        np.savez_compressed(OUT/f'{frame:03d}-learned.npz', depth_m=depth,
                            left_disparity_px=dl, right_disparity_px=dr,
                            left_right_consistent=consistency)
        report['complete'].append(dict(frame=frame, inference_seconds=time.monotonic()-start,
                                       left_right_consistent_pixels=int(consistency.sum())))
        save()
        print('Completed bidirectional inference', frame, time.monotonic()-start, flush=True)
    for name, proof in meta['files'].items():
        assert sha(source/name) == proof['sha256']
    report['all_seven_complete'] = True
except BaseException as error:
    report['error'] = repr(error)
    raise
finally:
    save()

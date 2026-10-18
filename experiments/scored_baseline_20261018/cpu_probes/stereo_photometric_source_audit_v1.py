"""Independent RGB evidence for the fixed seven stereo depth comparisons.

Report both methods on identical valid pixels and fixed left-image texture
selection. No hidden depths, plane fit or object pose enters this comparison.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

S=Path(__file__).resolve().parents[1];I=S/'learned-stereo-inputs-v1';D=S/'fast-stereo-cpu-seven-v1'
OUT=S/'stereo-photometric-source-audit-v1';OUT.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta=json.loads((I/'manifest.json').read_text());inference=json.loads((D/'depth-report.json').read_text())
assert inference['all_seven_complete'] and inference['source_manifest_sha256']==sha(I/'manifest.json')
factor=meta['rectified_intrinsics'][0][0]*meta['baseline_m']
lut=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
def census(image):
    h,w=image.shape;result=np.zeros((h,w),np.uint32);bit=0
    for dy in range(-2,3):
        for dx in range(-2,3):
            if not(dx or dy):continue
            value=image[2+dy:h-2+dy,2+dx:w-2+dx]<image[2:h-2,2:w-2]
            result[2:h-2,2:w-2]|=value.astype(np.uint32)<<bit;bit+=1
    assert bit==24
    return result
def stats(values):
    return dict(count=int(len(values)),median=float(np.median(values)),p95=float(np.quantile(values,.95)),mean=float(np.mean(values))) if len(values)>=100 else dict(count=int(len(values)),status='insufficient fixed-region samples')
report=dict(schema='v2d.stereo-independent-photometric-source-audit.v1',producer_sha256=sha(Path(__file__)),source_manifest_sha256=sha(I/'manifest.json'),inference_report_sha256=sha(D/'depth-report.json'),source_or_reference_modified=False,depth_ground_truth_known=False,pose_or_support_used=False,physical_task_accepted=False,training_steps=0,selection='Identical common-valid image pixels; automatic mask eroded5x5, nearby26..100pixel ring; texture from left5x5 standard deviation>=3 gray values, fixed before comparison.',rows=[])
for frame in meta['frames']:
    left,right=[cv2.imread(str(I/f'{frame:03d}-{side}.png'),0) for side in ['left','right']]
    with np.load(I/f'{frame:03d}-control.npz',allow_pickle=False) as z:depth,cv=z['depth_m'].copy(),z['valid'].copy()
    old=np.divide(factor,depth,out=np.zeros_like(depth),where=depth>0)
    old=np.rint(old*16.)/16.
    with np.load(D/f'{frame:03d}-learned.npz',allow_pickle=False) as z:new,lv=z['left_disparity_px'].copy(),z['left_right_consistent'].copy()
    h,w=left.shape;yy,xx=np.indices(left.shape,dtype=np.float32)
    common=cv&lv&(old>.01)&(new>.01)&(xx-old>=2)&(xx-new>=2)&(xx-old<w-3)&(xx-new<w-3)&(yy>=2)&(yy<h-3)
    floating=left.astype(np.float32);mean=cv2.blur(floating,(5,5));std=np.sqrt(np.maximum(0,cv2.blur(floating*floating,(5,5))-mean*mean))
    textured=std>=3.
    mask=cv2.imread(str(I/f'{frame:03d}-mask.png'),0)>127
    interior=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8))>0
    distance=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,3)
    regions=dict(object_interior=common&interior,nearby_mask_ring=common&(distance>26)&(distance<100))
    cl,cr=census(left),census(right);row=dict(frame=frame,common_valid_pixels=int(common.sum()),regions={})
    for name,region in regions.items():
        chosen=region&textured;measurements=[]
        for label,disp in [('SGBM',old),('C-FastFoundationStereoCPU',new)]:
            rx=xx-disp;ix=np.clip(np.rint(rx).astype(np.int32),0,w-1)
            descriptor=np.bitwise_xor(cl[chosen],cr[np.broadcast_to(np.arange(h)[:,None],left.shape)[chosen],ix[chosen]])
            costs=lut[descriptor[:,None].view(np.uint8)].sum(axis=1)/24.
            warped=cv2.remap(right.astype(np.float32),rx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            residual=floating-warped
            bias_removed=abs(residual-cv2.blur(residual,(5,5)))
            measurements.append(dict(backend=label,census_hamming_fraction=stats(costs),raw_absolute_gray_error=stats(abs(residual[chosen])),local_bias_removed_gray_error=stats(bias_removed[chosen])))
        row['regions'][name]=dict(common_pixels=int(region.sum()),fixed_textured_pixels=int(chosen.sum()),measurements=measurements)
    report['rows'].append(row);(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(frame=frame,regions={n:[dict(backend=v['backend'],census_median=v['census_hamming_fraction'].get('median'),bias_removed_median=v['local_bias_removed_gray_error'].get('median')) for v in r['measurements']] for n,r in row['regions'].items()})),flush=True)
for name,proof in meta['files'].items():assert sha(I/name)==proof['sha256']
report['frozen_source_closure_verified_after']=True
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')

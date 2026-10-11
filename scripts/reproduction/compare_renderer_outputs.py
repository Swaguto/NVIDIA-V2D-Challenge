"""Measure simulator and pixel differences between two actual renderer runs."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def compare(a):
    report={}
    with np.load(a.reference/'simulator-readback.npz') as first,np.load(a.candidate/'simulator-readback.npz') as second:
        for key in first.files:
            if key not in second.files:raise ValueError('Missing shared readback: '+key)
            x,y=first[key],second[key]
            if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('Readback shape/dtype mismatch')
            if x.dtype.kind=='f':
                delta=float(np.max(np.abs(x-y)))
                if not np.isfinite(delta) or delta>1e-4:raise ValueError('Readback difference: '+key)
                report[key]=delta
            elif not np.array_equal(x,y):raise ValueError('Discrete readback differs: '+key)
        frames=first['frame_indices']
    changed=0; max_pixel_error=0
    for frame in frames:
        name=f'{int(frame):06d}.png'
        x=np.asarray(Image.open(a.reference/'frames'/name)).astype(np.int16)
        y=np.asarray(Image.open(a.candidate/'frames'/name)).astype(np.int16)
        if x.shape!=y.shape:raise ValueError('Renderer shape differs')
        delta=int(np.max(np.abs(x-y))); max_pixel_error=max(max_pixel_error,delta)
        changed+=int(delta!=0)
    result=dict(status='passed shared simulator target reproduction',frames=len(frames),
                readback_maximum_differences=report,pixel_changed_frames=changed,
                maximum_pixel_channel_difference=max_pixel_error,
                scope='Two runs on the same A40/runtime; not a second-machine test')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','out'):p.add_argument('--'+name,type=Path,required=True)
    compare(p.parse_args())

"""Compare independently rebuilt scene export, IK and gate arrays."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(reference, candidate, atol=1e-8):
    report={}
    for filename in ('bundle/episode.npz','retarget/retarget.npz','gate/penetration.npz'):
        metrics={}
        with np.load(reference/filename,allow_pickle=False) as expected, np.load(candidate/filename,allow_pickle=False) as actual:
            if set(expected.files)!=set(actual.files):raise ValueError('Array keys differ: '+filename)
            for key in expected.files:
                x,y=expected[key],actual[key]
                if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('Shape/dtype differs: '+key)
                if x.dtype.kind in 'fc':
                    if not np.array_equal(np.isnan(x),np.isnan(y)) or not np.array_equal(np.isfinite(x),np.isfinite(y)):
                        raise ValueError('Finite/NaN masks differ: '+key)
                    mask=np.isfinite(x)
                    delta=float(np.max(np.abs(x[mask]-y[mask]))) if mask.any() else 0.0
                    if not np.allclose(x,y,rtol=0,atol=atol,equal_nan=True):raise ValueError('Numeric arrays differ: '+key)
                    metrics[key]=delta
                elif not np.array_equal(x,y):raise ValueError('Discrete array differs: '+key)
        report[filename]=dict(arrays=len(expected.files),maximum_absolute_difference=max(metrics.values(),default=0))
    return dict(status='passed',absolute_float_tolerance=atol,relative_float_tolerance=0,payloads=report,
                scope='Export/IK/gate from the same refined scene; independent inference/refinement not tested')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    result=compare(a.reference,a.candidate)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

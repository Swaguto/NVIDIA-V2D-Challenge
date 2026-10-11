"""Measure saved-array differences without inventing an acceptance tolerance."""
import argparse
import json
from pathlib import Path
import numpy as np

def main(args):
    results={}
    with np.load(args.reference,allow_pickle=False) as a,np.load(args.candidate,allow_pickle=False) as b:
        if set(a.files)!=set(b.files):raise ValueError('Scene array keys differ')
        for key in a.files:
            x,y=a[key],b[key]
            if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError(f'Schema differs: {key}')
            row={'shape':list(x.shape),'dtype':str(x.dtype)}
            if x.dtype.kind in 'fc':
                row['nan_mask_equal']=bool(np.array_equal(np.isnan(x),np.isnan(y)))
                finite=np.isfinite(x)&np.isfinite(y)
                delta=np.abs(x[finite].astype(np.float64)-y[finite].astype(np.float64))
                row.update(exact=bool(np.array_equal(x,y,equal_nan=True)),
                    max_absolute_difference=float(delta.max()) if delta.size else 0.0,
                    rms_difference=float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0)
            else:row['exact']=bool(np.array_equal(x,y))
            results[key]=row
    report={'all_arrays_exact':all(r['exact'] for r in results.values()),
            'arrays':results,'scope':'saved arrays; no rendered-image quality or ground-truth claim'}
    if args.out.exists():raise FileExistsError(args.out)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())

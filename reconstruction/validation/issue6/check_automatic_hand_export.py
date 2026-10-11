"""Verify exported native records preserve the checked camera-space geometry."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from automatic_hands import write,sha

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--quality',type=Path,required=True)
p.add_argument('--mano-assets',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
path=Path(__file__).parents[2]/'modules/v2d_hamer/lib/render_hands_aligned_video.py'
spec=importlib.util.spec_from_file_location('native_hand_renderer',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
mano=module._mano_layer(str(a.mano_assets));rows=[]
for path in sorted((a.quality/'accepted/records').glob('*/*.json')):
    record=json.loads(path.read_text());vertices,_=module._build_hand_mesh(record,mano)
    side='right' if record['is_right'] else 'left'
    with np.load(a.quality/'geometry'/side/(path.stem+'.npz'),allow_pickle=False) as geometry:
        error=float(np.linalg.norm(vertices-geometry['vertices'],axis=1).max())
    rows.append({'frame':int(path.stem),'side':side,'maximum_vertex_difference_m':error,'record_sha256':sha(path)})
maximum=max((r['maximum_vertex_difference_m'] for r in rows),default=None)
passed=maximum is not None and maximum<=1e-5
write(a.out,{'scope':'Export format consistency only; not source-pose accuracy.',
             'records':len(rows),'maximum_vertex_difference_m':maximum,'tolerance_m':1e-5,'passed':passed,'rows':rows})
print(json.dumps({'records':len(rows),'maximum_vertex_difference_m':maximum,'passed':passed}))
if not passed:raise SystemExit('Native export mismatch or no records')

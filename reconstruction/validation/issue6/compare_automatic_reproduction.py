"""Compare frozen automatic runs by frame support and actual hand geometry."""
import argparse
import json
from pathlib import Path
import numpy as np
from automatic_hands import write,sha

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
roots=[a.reference,a.candidate]
receipts=[json.loads((r/'proposals/receipt.json').read_text()) for r in roots]
models=('method_files','episode','source_manifest_sha256','detector_sha256','hamer_checkpoint_sha256',
        'sam_checkpoint_sha256','mano_right_sha256','hamer_config_sha256','mano_mean_params_sha256','hamer_source_commit')
profile_matches=all(receipts[0][k]==receipts[1][k] for k in models)
evidence=[json.loads((r/'quality/frame-evidence.json').read_text()) for r in roots]
support=[{(row['frame'],s):item['accepted_by_consistency_proxies'] for row in rows for s,item in row['hands'].items()} for rows in evidence]
frame_ids_match=[r['frame'] for r in evidence[0]]==[r['frame'] for r in evidence[1]]
keys=set(support[0])|set(support[1]);mismatches=[{'frame':f,'side':s} for f,s in sorted(keys) if support[0].get((f,s))!=support[1].get((f,s))]
geometry=[]
for frame,side in sorted(keys):
    paths=[r/'quality/geometry'/side/f'{frame:06d}.npz' for r in roots]
    if not all(path.exists() for path in paths):continue
    with np.load(paths[0],allow_pickle=False) as ref,np.load(paths[1],allow_pickle=False) as cand:
        geometry.append({'frame':frame,'side':side,**{key:float(np.linalg.norm(ref[key]-cand[key],axis=1).max())
            for key in ('vertices','joints','raw_vertices','raw_joints')}})
maximum=max((r[k] for r in geometry for k in ('vertices','joints','raw_vertices','raw_joints')),default=None)
passed=profile_matches and frame_ids_match and not mismatches and maximum is not None and maximum<=1e-4
result={'scope':'Same input inventory is separately required. Numerical reproducibility is not source-pose accuracy.',
        'profile_matches':profile_matches,'frame_ids_match':frame_ids_match,'support_mismatches':mismatches,
        'maximum_geometry_difference_m':maximum,'tolerance_m':1e-4,'passed':passed,
        'reference_receipt_sha256':sha(a.reference/'proposals/receipt.json'),
        'candidate_receipt_sha256':sha(a.candidate/'proposals/receipt.json'),'geometry':geometry}
write(a.out,result)
print(json.dumps({k:v for k,v in result.items() if k!='geometry'}))
if not passed:raise SystemExit('Reproduction differs')

"""Evaluate frozen camera fits using public relative geometry; never modify predictions."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--predictions',type=Path,required=True)
parser.add_argument('--kit',type=Path,required=True,help='extracted verified participant kit')
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
REPO=Path(__file__).resolve().parents[2]
DATA, INPUT, OUT=args.data,args.predictions,args.out
sys.path.insert(0,str(REPO/'scripts/eval'))
import numpy as np
from official_metric import relative_position_error_cm, rotation_matrix
from reference_loader import load_reference
from v2d.world_calib.vm.fit_ego_rig import js_to_mat

if OUT.exists():raise ValueError('fresh evaluation required')
raw=INPUT.read_bytes()
prediction=json.loads(raw)
if prediction['gt_use']!='none':raise ValueError('expected frozen GT-free camera observations')
kit=args.kit/'metric_code/track_3/RPE.py'
if hashlib.sha256(kit.read_bytes()).hexdigest()!='12315085e9c674e4476e90c56b457854f686ca80bbf86d1f3b06121630bd6f40':
 raise ValueError('RPE scorer differs from verified participant kit')
spec=importlib.util.spec_from_file_location('participant_rpe',kit)
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
expected_gt={2:'a1cd007e34edbdaa51138c168312ad4f0bb6e60ed65b52e72d253d6f949ae00b',0:'cec5ecb4f696e3ecde143afaa4d655f71e8311d4c3dfceb7b264d515a8ef94bb'}
records=[]
gt_sources=[]
for episode in [2,0]:
 path=DATA/f'data/chunk-000/episode_{episode:06d}.parquet'
 gt_sha=hashlib.sha256(path.read_bytes()).hexdigest()
 expected=expected_gt[episode]
 if gt_sha!=expected:raise ValueError('GT bytes changed')
 gt_sources.append({'episode':episode,'parquet_sha256':gt_sha})
 ref=load_reference(episode,DATA)
 by_frame={int(frame):index for index,frame in enumerate(ref.frame_index)}
 names=prediction['objects']
 slots=[ref.object_names.index(n) for n in names]
 rows=[x for x in prediction['frames'] if x['episode']==episode]
 pairs=[]
 achieved=[]
 references=[]
 for row in rows:
  frame=row['frame_index'];index=by_frame[frame]
  entry={'frame_index':frame,'capture_time':float(ref.capture_time[index]),
         'gt_visible':{n:bool(ref.visible[index,s]) for n,s in zip(names,slots)},
         'accepted':{n:row['fits'][n]['status']=='observed' for n in names},'pair_errors':None}
  if all(entry['accepted'].values()) and all(entry['gt_visible'].values()):
   pp=[];gg=[];mm=[];ww=[]
   for n,s in zip(names,slots):
    fit=row['fits'][n]
    q=np.asarray(fit['rotation_wxyz'])
    pp.append(np.r_[fit['translation'],q[[1,2,3,0]]])
    gg.append(ref.pose_xyzw[index,s])
    mm.append(js_to_mat({'translation':fit['translation'],'rotation':q}))
    pose=ref.pose_xyzw[index,s]
    ww.append(js_to_mat({'translation':pose[:3],'rotation':pose[[6,3,4,5]]}))
   achieved.append(pp);references.append(gg)
   errors=[]
   for first,second in [(0,1),(1,0)]:
    fitted=np.linalg.inv(mm[first])@mm[second]
    truth=np.linalg.inv(ww[first])@ww[second]
    delta=np.linalg.inv(truth)@fitted
    angle=float(np.degrees(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1))))
    errors.append({'from_object':names[first],'to_object':names[second],
                   'translation_error_cm':float(np.linalg.norm(fitted[:3,3]-truth[:3,3])*100),
                   'canonical_rotation_error_deg':angle})
   entry['pair_errors']=errors
  pairs.append(entry)
 measured=[x for x in pairs if x['pair_errors']]
 summary={'episode':episode,'sampled_frames':len(rows),'common_visible_sampled':sum(all(x['gt_visible'].values()) for x in pairs),
          'common_visible_accepted':len(measured),'frames':pairs,'official_rpe_kernel_cm':None,
          'median_relative_translation_cm':None,'median_relative_canonical_rotation_deg':None}
 if achieved:
  a=np.asarray(achieved)[:,None];g=np.asarray(references)[:,None]
  kernel=float(official._relative_position_error_cm(a,g))
  local=relative_position_error_cm(a,g)
  if abs(kernel-local)>1e-9:raise ValueError('kernel differs from participant kit')
  # Common moving-camera transforms cancel in this within-frame metric.
  aligned=official._warp_to_reference_start(a,g)
  if abs(kernel-official._relative_position_error_cm(aligned,g))>1e-8:raise ValueError('constant-gauge invariance failed')
  summary.update(official_rpe_kernel_cm=kernel,
                 median_relative_translation_cm=float(np.median([x['pair_errors'][0]['translation_error_cm'] for x in measured])),
                 median_relative_canonical_rotation_deg=float(np.median([x['pair_errors'][0]['canonical_rotation_error_deg'] for x in measured])))
 records.append(summary)
report={'prediction_sha256':hashlib.sha256(raw).hexdigest(),'gt_sources':gt_sources,
        'info_sha256':hashlib.sha256((DATA/'meta/info.json').read_bytes()).hexdigest(),
        'official_rpe_source_sha256':hashlib.sha256(kit.read_bytes()).hexdigest(),
        'evaluator_source_sha256_lf':hashlib.sha256(Path(__file__).read_text().encode()).hexdigest(),
        'object_order':prediction['objects'],'episodes':records,
        'gt_use':'evaluation_only_after_frozen_inference',
        'limits':['Eight sampled frames per public episode; no full coverage or all-20 score.',
                  'RPE uses common GT-visible accepted observations; missing/rejected counts retained.',
                  'Camera motion cancels for within-frame relative transforms; no independent C(t) supplied.',
                  'This is direct participant-kit RPE mathematics, not a policy-rollout score() submission.',
                  'Other five metrics and fixed-world accuracy are not claimed.']}
OUT.write_text(json.dumps(report,indent=2))
if hashlib.sha256(INPUT.read_bytes()).hexdigest()!=report['prediction_sha256']:raise ValueError('predictions changed during evaluation')
print(json.dumps({str(x['episode']):{k:v for k,v in x.items() if k!='frames'} for x in records},indent=2))

"""Trace the source observation gap and exact world-space command completion."""
import hashlib,json
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parents[1];OUT=S/'e037-carry-observation-gap-v1';OUT.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz';P=S/'scored-cohort-14-results/policy-e037/frozen'
with np.load(B) as z:source={k:z[k].copy() for k in ['hand_valid','hand_directly_observed','hand_metric_valid','frame_indices','timestamps_s']}
with np.load(P/'targets.npz') as z:q=z['robot_qpos'].copy();times=z['timestamps_s'].copy()
np.testing.assert_array_equal(times,source['timestamps_s'])
def runs(mask):
 ids=np.flatnonzero(mask);groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
 return [{'first_frame':int(v[0]),'last_frame':int(v[-1]),'frames':len(v),'duration_at20Hz_s':len(v)/20} for v in groups if len(v)]
start,end=35,70;rows=np.arange(start+1,end);alpha=(times[rows]-times[start])/(times[end]-times[start])
expected=q[start,1,:3]+alpha[:,None]*(q[end,1,:3]-q[start,1,:3]);error=float(abs(expected-q[rows,1,:3]).max());assert error<1e-6
v=json.load(open(S/'plane-aligned-grasp-separation-v2/frame-bounds.json'));f50=next(r for r in v['frames'] if r['frame']==50)
report={'schema':'v2d.e037-carry-gap-source-audit.v1','producer_sha256':sha(Path(__file__)),'bundle_sha256':sha(B),'original_robot_manifest_sha256':sha(P/'manifest.json'),'separation_report_sha256':sha(S/'plane-aligned-grasp-separation-v2/report.json'),'original_source_flags_changed':False,'reference_modified':False,'new_training_steps':0,'physical_task_accepted':False,'source_frames':151,'right_hand_missing_validity_runs':runs(~source['hand_valid'][:,1]),'right_hand_missing_direct_observation_runs':runs(~source['hand_directly_observed'][:,1]),'right_hand_missing_metric_runs':runs(~source['hand_metric_valid'][:,1]),'world_interpolation_control':{'start':start,'end':end,'interior_frames':len(rows),'maximum_component_difference_m':error,'reproduces_original_missing_hand_root_commands':True},'frame50':f50,'diagnosis':'Source video visibly carries the spoon in right hand at50 while original hand record is invalid/notdirect/notmetric. Original robot root gap36..69 reproduces straight world-space interpolation35..70, not the lifted spoon path. Later local morphology/contact repairs preserve that large upstream gap. This establishes an unsuitable carry reference, not a complete explanation of every policy failure.','next_required_source_work':'Recover separately qualified source hand observations/metric constraints during carry, or validate a separately marked source-supported object-relative completion. Retain original missing flags, inferred status, strict native/source/motion gates and original morphology bound; do not promote the >25cm gap proposals as small morph corrections.'}
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='frame50'},indent=2))

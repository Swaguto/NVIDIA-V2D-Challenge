"""Test existing conservative gap hypothesis; no inferred contact is certified."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,trimesh
from scipy.spatial.transform import Rotation
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/track3'))
from object_relative_gaps import complete_object_relative_gaps
P=S/'plane-aligned-articulated-native-validation-v2/frozen';OUT=S/'e037-object-relative-gap-probe-v1';OUT.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.load(open(P/'manifest.json'))
with np.load(P/'aligned-source-reference.npz') as z:q=z['robot_qpos'].copy();times=z['timestamps_s'].copy();hv=z['hand_valid'].copy();direct=z['hand_directly_observed'].copy();metric=z['hand_metric_valid'].copy()
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:observed=z['object_0_valid']&~z['object_0_pose_inferred']
with np.load(P/'targets.npz') as z:p=z['object_poses'].copy();indices=z['frame_indices'].copy()
world=np.repeat(np.eye(4)[None],len(q),axis=0);world[:,:3,3]=p[:,0,:3];world[:,:3,:3]=Rotation.from_quat(p[:,0,3:7]).as_matrix()
mesh=P/m['objects'][0]['mesh'];vertices=np.asarray(trimesh.load(mesh,force='mesh',process=False).vertices)
report={'schema':'v2d.e037-source-gap-hypothesis.v1','producer_sha256':sha(Path(__file__)),'algorithm_sha256':sha(R/'scripts/track3/object_relative_gaps.py'),'manifest_sha256':sha(P/'manifest.json'),'source_bundle_sha256':sha(B),'source_flags_changed':False,'object_poses_changed':False,'new_training_steps':0,'physics_stepped':False,'physical_task_accepted':False,'trajectory_promoted':False,'variants':[]}
for name,anchors in [('visual_hand_valid',hv),('direct_metric_only',hv&direct&metric)]:
 out,slots,decision=complete_object_relative_gaps(q,anchors,times,[{'vertices':vertices,'world':world,'observed':observed}])
 change=np.linalg.norm(out[:,:,:3]-q[:,:,:3],axis=-1)
 decision.update(anchor_policy=name,maximum_root_change_m=float(change.max()),original_25mm_morph_gate_passed=bool(change.max()<=.025),native_contact_source_and_motion_qualified=False)
 report['variants'].append(decision)
 if (slots>=0).any():np.savez_compressed(OUT/f'{name}-unqualified-proposal.npz',robot_qpos=out,object_relative_inferred=slots>=0,source_anchor_valid=anchors,frame_indices=indices,timestamps_s=times)
print(json.dumps(report,indent=2));(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')

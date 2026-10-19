"""NEW contact-mode component: preserve hard transition failures and repair later tail."""
from pathlib import Path
import datetime,hashlib,json,sys
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'))
import numpy as np,xml.etree.ElementTree as ET
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();O=S/'e037-right-late-contact-mode-v1';F=S/'e037-left-thumb-independent-validation-v1/frozen';C=S/'e037-right-thumb-bracketed-v2';rep=json.loads((C/'report.json').read_text());assert rep['input_manifest_sha256']==sha(F/'manifest.json')
with np.load(C/'right-thumb-reference-commands.npz') as z:proposal=z['robot_qpos'].copy();base=z['original_robot_qpos'].copy();ts=z['timestamps_s'].copy()
with np.load(F/'targets.npz') as z:np.testing.assert_array_equal(base,z['robot_qpos'])
with np.load(F/'aligned-source-reference.npz') as z:aligned=z['robot_qpos'].copy()
assert sha(C/'right-thumb-reference-commands.npz')==rep['commands_sha256']
assert all(r['single_knot_checks_passed'] for r in rep['rows'][116:])
# Declare the separate late contact-mode block after the rejected source/velocity transition.
# Earlier failed slots remain explicit. A fixed cubic ramp preserves temporal authority.
weights=np.zeros(len(base));weights[116:120]=np.array([.15625,.5,.84375,1.]);weights[120:]=1.
q=base.copy();q[:,1,11:]+=weights[:,None]*(proposal[:,1,11:]-base[:,1,11:]);manifest=json.loads((F/'manifest.json').read_text());motion=[]
for si,side in enumerate(['left','right']):
 lj={j.attrib['name']:j.find('limit').attrib for j in ET.parse(F/f'robot/urdfs/g1/dex3_{side}.urdf').getroot().findall('joint') if j.attrib['type']!='fixed'};names=manifest['joint_names'][side];lo=np.array([float(lj[n]['lower']) for n in names]);hi=np.array([float(lj[n]['upper']) for n in names]);vc=np.array([float(lj[n]['velocity']) for n in names]);v=abs(np.diff(q[:,si,7:],axis=0)/np.diff(ts)[:,None]);d=q[:,si,7:]-aligned[:,si,7:];ok=bool(((q[:,si,7:]>=lo-1e-8)&(q[:,si,7:]<=hi+1e-8)).all() and (v<=vc+1e-7).all() and abs(d).max()<=.25+1e-7 and abs(np.diff(d,axis=0)).max()<=.05+1e-7 and abs(np.diff(d,n=2,axis=0)).max()<=.05+1e-7);motion.append(dict(hand=side,passed=ok,component_rad=float(abs(d).max()),adjacent_added_rad=float(abs(np.diff(d,axis=0)).max()),second_added_rad=float(abs(np.diff(d,n=2,axis=0)).max())))
np.testing.assert_array_equal(q[:,0],base[:,0]);np.testing.assert_array_equal(q[:,:,:7],base[:,:,:7]);np.testing.assert_array_equal(q[:116,1],base[:116,1]);np.testing.assert_array_equal(q[:,1,7:11],base[:,1,7:11]);assert all(m['passed'] for m in motion)
np.savez_compressed(O/'reference-commands.npz',robot_qpos=q,original_robot_qpos=base,timestamps_s=ts,frame_indices=np.arange(len(q)))
report=dict(schema='v2d.right-late-contact-mode-component.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),parent_manifest_sha256=sha(F/'manifest.json'),parent_failed_constructor_report_sha256=sha(C/'report.json'),commands_sha256=sha(O/'reference-commands.npz'),component_half_open=[116,151],fixed_cubic_ramp_half_open=[116,120],ramp_weights=weights[116:120].tolist(),reason='The early/late observed transitions fail source or temporal gates; their failed controls are retained. Later thumb/palm/index contact modes form a separate source-eligible block, with a fixed bounded bridge. This is a partial component, not erasure of earlier failures.',original_motion=motion,verified_left_roots_objects_nonthumb_source_human_flags_exact_unchanged=True,independent_source_self_native_validation_required=True,whole_reference_or_physical_task_qualified=False,new_GPU_jobs=0,new_PPO_steps=0)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

"""NEW saved-bank graph reachability and exact-ray representation limitation."""
from pathlib import Path
import datetime,hashlib,json,sys,time,xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');O=S/'e037-coupled-bank-temporal-graph-v2';sys.path.insert(0,str(R/'scripts/dex3_training'));start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
import numpy as np
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.temporal_pose_graph import finite_temporal_reachability
F=S/'e037-thumb-ray-independent-validation-v1/frozen';D=S/'e037-coupled-ray-pose-bank-v1';r=json.loads((D/'report.json').read_text());assert sha(D/'pose-bank.json')==r['pose_bank_sha256'];assert sha(F/'manifest.json')==r['input_manifest_sha256'];lib=json.loads((S/'e037-temporal-pose-graph-library-v1/report.json').read_text());assert lib['new_library_controls_passed'];bank=json.loads((D/'pose-bank.json').read_text());mf=json.loads((F/'manifest.json').read_text())
with np.load(F/'aligned-source-reference.npz') as z:aligned=z['robot_qpos'].copy()
with np.load(F/'targets.npz') as z:current=z['robot_qpos'].copy();ts=z['timestamps_s'].copy()
frames=np.array([row['frame'] for row in r['rows']]);assert np.array_equal(frames,np.arange(102,120));indices=[row['selected_bank_indices'] for row in r['rows']];banks=[np.array([bank[i]['command'][7:] for i in idx]).reshape(-1,7) for idx in indices];limits={j.attrib['name']:j.find('limit').attrib for j in ET.parse(F/'robot/urdfs/g1/dex3_right.urdf').getroot().findall('joint') if j.attrib['type']!='fixed'};lo=np.array([float(limits[n]['lower']) for n in mf['joint_names']['right']]);hi=np.array([float(limits[n]['upper']) for n in mf['joint_names']['right']]);vc=np.array([float(limits[n]['velocity']) for n in mf['joint_names']['right']]);costs=[]
for f,idx in zip(frames,indices,strict=True):
 row=[]
 for i in idx:
  cmd=np.array(bank[i]['command']);oldR=Rotation.from_quat(current[f,1,3:7]);newR=Rotation.from_quat(cmd[3:7]);row.append(float(np.sum(((cmd[7:]-current[f,1,7:])/.25)**2)+np.sum(((newR*oldR.inv()).as_rotvec()/.08)**2)))
 costs.append(row)
result=finite_temporal_reachability(banks,aligned[frames,1,7:],ts[frames],lo,hi,vc,component_cap=.25,adjacent_added_cap=.05,second_added_cap=.05,node_costs=costs,tolerance=1e-8)
# This bank contains empty rows. Preserve them, without silently skipping frames.
assert result.path_indices is None
seed_verification_path=S/'e037-source-coupled-isolated-independent-v1/report.json';seeds=json.loads(seed_verification_path.read_text());representation=[]
for seed in seeds['rows']:
 errors=np.array(seed['source_image_before_after_px']);lower=abs(errors[1]-errors[0]);prescribed=np.minimum(.0001,.01*errors[0]);records=[]
 for i,digit in [(1,'thumb'),(2,'index'),(3,'middle')]:
  records.append(dict(digit=digit,pixel_motion_lower_bound_px=float(lower[i]),fixed_ray_pixel_motion_px=float(prescribed[i]),not_representable_by_the_generator_ray=bool(lower[i]>prescribed[i]+1e-6)))
 representation.append(dict(frame=seed['frame'],records=records,source_nonregressing_pose_remains_valid_as_isolated_control=True,imported_control_not_regenerated=True))
assert all(r['records'][-1]['not_representable_by_the_generator_ray'] for r in representation)
report=dict(schema='v2d.coupled-bank-temporal-graph.v2',initial_stricter_adapter_failure_retained=True,original_URDF_position_check_tolerance_rad=1e-8,retained_seed113_exceeds_rounded_URDF_lower_by_rad=1.1965426427451575e-9,no_bound_or_source_gate_relaxed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),input_manifest_sha256=sha(F/'manifest.json'),pose_bank_report_sha256=sha(D/'report.json'),pose_bank_sha256=sha(D/'pose-bank.json'),graph_library_sha256=lib['library_sha256'],frames=frames.tolist(),selected_bank_sizes=[len(b) for b in banks],path_found=False,empty_bank_frames=[int(frames[i]) for i in result.empty_bank_indices],forward_pair_counts=list(result.forward_pair_counts),backward_pair_counts=list(result.backward_pair_counts),complete_path_viable_node_counts=[len(v) for v in result.viable_node_indices],joint_motion_only_scope='Necessary finite graph; includes actual joint speed and original added first/second bounds. Root p95/acceleration, continuous collision and dynamics are not certified.',saved_isolated_control_verification_sha256=sha(seed_verification_path),exact_ray_representation_limitation=representation,triangle_inequality_bound_not_a_new_source_or_native_audit=True,empty_finite_bank_not_global_infeasibility=True,no_source_threshold_or_authority_changed=True,no_old_pose_bank_native_or_optimization_jobs_repeated=True,current_three_component_freeze_unchanged=True,new_GPU_jobs=0,new_PPO_steps=0,no_full_reference_or_task_qualified=True,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

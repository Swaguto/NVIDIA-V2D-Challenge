"""NEW exact second-order finite-bank temporal path, no local solver or native rerun."""
from pathlib import Path
import datetime,hashlib,json,time,sys,xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();O=S/'e037-thumb-ray-temporal-path-v1'
import numpy as np

def path(layers,reference,baseline,times,velocity,adjacent=.05,second=.05):
 # Exact dynamic programming over consecutive PAIRS, retaining all feasible predecessors.
 n=len(layers);delta=[v-reference[i] for i,v in enumerate(layers)];cost=[np.sum((v-baseline[i])**2,axis=1) for i,v in enumerate(layers)];back=[]
 def edges(i):
  actual=layers[i][None,:,:]-layers[i-1][:,None,:];added=delta[i][None,:,:]-delta[i-1][:,None,:]
  return (np.max(abs(added),axis=2)<=adjacent+1e-8)&np.all(abs(actual)<=(times[i]-times[i-1])*velocity+1e-7,axis=2)
 score=cost[0][:,None]+cost[1][None,:];score[~edges(1)]=np.inf
 for i in range(2,n):
  valid=edges(i);nextscore=np.full((len(layers[i-1]),len(layers[i])),np.inf);parents=np.full(nextscore.shape,-1,dtype=int)
  for a in range(len(layers[i-1])):
   second_difference=delta[i][None,:,:]-2*delta[i-1][a][None,None,:]+delta[i-2][:,None,:];ok=np.max(abs(second_difference),axis=2)<=second+1e-8;candidate=score[:,a,None]+cost[i][None,:];candidate[~ok]=np.inf;parent=np.argmin(candidate,axis=0);best=candidate[parent,np.arange(len(layers[i]))];best[~valid[a]]=np.inf;nextscore[a]=best;parents[a]=parent
  back.append(parents);score=nextscore
 if not np.isfinite(score).any():return None
 a,b=np.unravel_index(np.argmin(score),score.shape);indices=[None]*n;indices[-2:]=[int(a),int(b)]
 for i in range(n-1,1,-1):indices[i-2]=int(back[i-2][indices[i-1],indices[i]])
 return indices
# Independent small controls: a greedy-nearest branch loses a required predecessor;
# second differences, original-reference deltas and actual speed are separate gates.
controls=[]
L=[np.array([[0.]]),np.array([[0.],[.02]]),np.array([[.04]])];Z=np.zeros((3,1));T=np.arange(3,dtype=float)
r=path(L,Z,Z,T,np.array([1.]),second=.025);assert r==[0,1,0];controls.append('pair-state path keeps nongreedy predecessor')
assert path(L,Z,Z,T,np.array([1.]),second=.005) is None;controls.append('rejects second-added violation')
assert path(L,Z,Z,T,np.array([.001]),second=.05) is None;controls.append('rejects actual speed violation')
assert path([np.array([[.2]]),np.array([[.2]]),np.array([[.2]])],np.array([[0],[.1],[0]]),Z,T,np.array([1.]),adjacent=.05) is None;controls.append('original-reference added motion is not motion from a repaired baseline')
F=S/'e037-right-late-independent-validation-v1/frozen';manifest=json.loads((F/'manifest.json').read_text());prior=S/'e037-thumb-source-ray-branches-v1/report.json';report=json.loads(prior.read_text());bank_path=S/'e037-thumb-source-ray-branches-v1/candidate-bank.json';assert sha(bank_path)==report['candidate_bank_sha256'] and sha(F/'manifest.json')==report['input_manifest_sha256'];bank=json.loads(bank_path.read_text())
with np.load(F/'targets.npz') as z:q=z['robot_qpos'].copy();ts=z['timestamps_s'].copy()
with np.load(F/'aligned-source-reference.npz') as z:original=z['robot_qpos'].copy()
names=manifest['joint_names']['right'][4:];lj={j.attrib['name']:j.find('limit').attrib for j in ET.parse(F/'robot/urdfs/g1/dex3_right.urdf').getroot().findall('joint') if j.attrib['type']!='fixed'};velocity=np.array([float(lj[n]['velocity']) for n in names]);layers=[];provenance=[]
for frame in range(7):
 records=[r for r in bank if r['frame']==frame and r['all_original_source_image_nonregression'] and r['original_position_bounds'] and r['native_self_penetration_m']<=.001]
 if frame>=5:records=[dict(frame=frame,ray='unchanged verified combined command',thumb_joints_rad=q[frame,1,11:].tolist())]
 assert records,(frame,'No finite native/source-qualified node; no fallback promoted');layers.append(np.array([r['thumb_joints_rad'] for r in records]));provenance.append(records)
indices=path(layers,original[:7,1,11:],q[:7,1,11:],ts[:7],velocity);chosen=q.copy();selected=[]
if indices is not None:
 for frame,node in enumerate(indices):chosen[frame,1,11:]=layers[frame][node];selected.append(provenance[frame][node])
np.testing.assert_array_equal(chosen[:,0],q[:,0]);np.testing.assert_array_equal(chosen[:,:,:7],q[:,:,:7]);np.testing.assert_array_equal(chosen[5:,1],q[5:,1]);np.testing.assert_array_equal(chosen[:,1,7:11],q[:,1,7:11]);delta=chosen[:,1,7:]-original[:,1,7:];speed=abs(np.diff(chosen[:,1,7:],axis=0)/np.diff(ts)[:,None]);vc=np.array([12,12,12,12,6.857,12,12]);motion=bool(abs(delta).max()<=.25+1e-7 and abs(np.diff(delta,axis=0)).max()<=.05+1e-7 and abs(np.diff(delta,n=2,axis=0)).max()<=.05+1e-7 and (speed<=vc+1e-7).all());assert motion
np.savez_compressed(O/'reference-commands.npz',robot_qpos=chosen,original_robot_qpos=q,timestamps_s=ts,frame_indices=np.arange(len(q)))
r=dict(schema='v2d.thumb-ray-finite-temporal-path.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),input_manifest_sha256=sha(F/'manifest.json'),bank_sha256=sha(bank_path),source_report_sha256=sha(prior),analytical_controls_passed=controls,layer_counts=[len(x) for x in layers],path_found=indices is not None,selected_node_indices=indices,selected_source_native_seed_records=selected,original_motion_caps_passed=motion,commands_sha256=sha(O/'reference-commands.npz'),method='Exact pair-state dynamic programming in frozen finite bank for frames0:7, with unchanged5/6 boundary; minimum total squared change under original added first/second differences and actual velocity, no local restarts or geometry reruns',no_native_or_source_probe_repeated=True,verified_left_and_late_right_roots_source_data_objects_unchanged=True,finite_bank_failure_not_global_infeasibility=True,independent_full_candidate_source_native_self_motion_checks_required=True,whole_reference_or_physical_task_qualified=False,new_GPU_jobs=0,new_PPO_steps=0,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['analytical_controls_passed','layer_counts','path_found','selected_node_indices','original_motion_caps_passed','wall_seconds']},indent=2))

"""NEW necessary scalar-motion bounds for retained converged coupled poses."""
from pathlib import Path
import datetime, hashlib, json, sys, time, xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');O=S/'e037-coupled-seed-temporal-compatibility-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
import numpy as np
from scipy.optimize import linprog
F=S/'e037-thumb-ray-independent-validation-v1/frozen';P=S/'e037-source-orientation-coupled-fingers-v1/report.json';V=S/'e037-source-coupled-isolated-independent-v1/report.json';verified=json.loads(V.read_text());assert verified['all_two_isolated_source_self_bounds_prerequisites_passed'];assert sha(P)==verified['seed_report_sha256'];assert sha(F/'manifest.json')==verified['input_manifest_sha256'];mf=json.loads((F/'manifest.json').read_text())
with np.load(F/'targets.npz') as z:q=z['robot_qpos'].copy();ts=z['timestamps_s'].copy()
with np.load(F/'aligned-source-reference.npz') as z:aligned=z['robot_qpos'].copy()
seed={r['frame']:np.array(r['command']) for r in json.loads(P.read_text())['candidate_commands']};free=np.arange(104,118);N=len(q);nf=len(free);D1=np.diff(np.eye(N),axis=0);D2=np.diff(np.eye(N),n=2,axis=0);E=np.eye(N)[:,free];limits={j.attrib['name']:j.find('limit').attrib for j in ET.parse(F/'robot/urdfs/g1/dex3_right.urdf').getroot().findall('joint') if j.attrib['type']!='fixed'}
# Three independent analytical controls, including a second-difference conflict.
controls=[]
for name,result,expected in [('bounded_bridge',linprog([0,0],A_ub=[[1,-1],[-1,1]],b_ub=[.05,.05],bounds=[(.1,.1),(.1,.1)],method='highs').success,True),('first_difference_conflict',linprog([0,0],A_ub=[[1,-1],[-1,1]],b_ub=[.05,.05],bounds=[(.1,.1),(-.1,-.1)],method='highs').success,False),('second_difference_conflict',linprog([0,0,0],A_ub=[[1,-2,1],[-1,2,-1]],b_ub=[.05,.05],bounds=[(0,0),(.1,.1),(0,0)],method='highs').success,False)]:
 assert bool(result)==expected;controls.append(name)
rows=[];witness=np.array(q[:,1,7:]);ranges=[]
for j,name in enumerate(mf['joint_names']['right']):
 lim=limits[name];lo=np.maximum(aligned[free,1,7+j]-.25,float(lim['lower']));hi=np.minimum(aligned[free,1,7+j]+.25,float(lim['upper']));fixed=q[:,1,7+j].copy();fixed[free]=0;delta_fixed=fixed-aligned[:,1,7+j];vc=float(lim['velocity']);A=np.vstack([D1@E,-D1@E,D2@E,-D2@E,D1@E,-D1@E]);b=np.r_[.05-D1@delta_fixed,.05+D1@delta_fixed,.05-D2@delta_fixed,.05+D2@delta_fixed,vc*np.diff(ts)-D1@fixed,vc*np.diff(ts)+D1@fixed];bounds=list(zip(lo,hi,strict=True))
 for f,cmd in seed.items():bounds[int(np.flatnonzero(free==f)[0])]=(float(cmd[7+j]),float(cmd[7+j]))
 solve=linprog(np.zeros(nf),A_ub=A,b_ub=b,bounds=bounds,method='highs');row=dict(joint=name,success=bool(solve.success),status=int(solve.status),message=str(solve.message),converged_seed_values={str(f):float(cmd[7+j]) for f,cmd in seed.items()})
 if solve.success:
  x=solve.x;bound=np.array(bounds);res=float(np.max(np.einsum('ij,j->i',A,x,optimize=False)-b));assert res<1e-8 and (x>=bound[:,0]-1e-8).all() and (x<=bound[:,1]+1e-8).all();witness[free,j]=x;row['independent_linear_residual_max_rad']=res
  for f in free:
   objective=np.zeros(nf);objective[np.flatnonzero(free==f)[0]]=1;left=linprog(objective,A_ub=A,b_ub=b,bounds=bounds,method='highs');right=linprog(-objective,A_ub=A,b_ub=b,bounds=bounds,method='highs');assert left.success and right.success;rr=dict(frame=int(f),joint=name,minimum=float(left.fun),maximum=float(-right.fun));ranges.append(rr)
 rows.append(row)
report=dict(schema='v2d.coupled-seed-temporal-compatibility.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),input_manifest_sha256=sha(F/'manifest.json'),converged_seed_report_sha256=sha(P),isolated_verification_sha256=sha(V),free_frames_half_open=[104,118],fixed_seed_frames=[111,113],analytical_controls_passed=controls,rows=rows,necessary_finger_motion_compatible=all(r['success'] for r in rows),ranges=ranges,scope='Independent scalar necessary motion LPs only; a witness is not a source/native/root-motion or physically feasible pose trajectory',source_native_old_path_and_inference_not_repeated=True,current_verified_reference_unchanged=True,no_new_reference_promoted=True,new_GPU_jobs=0,new_PPO_steps=0,wall_seconds=time.monotonic()-start)
if report['necessary_finger_motion_compatible']:np.savez_compressed(O/'scalar-motion-witness-not-reference.npz',right_joint_angles=witness,timestamps_s=ts)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='ranges'},indent=2))

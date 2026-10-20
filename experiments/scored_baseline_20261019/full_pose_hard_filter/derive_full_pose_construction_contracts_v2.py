"""NEW concrete rotating-support/canonical/actual-wrapper contracts only."""
from pathlib import Path
import json,os,hashlib,datetime
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from full_pose_construction_v1 import rotating_support_pair,canonical
from full_pose_training_oracle_v1 import FullPoseTrainingOracle
S=Path(__file__).resolve().parents[1];O=S/'e037-full-pose-construction-contracts-v2'
def cv(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return{str(k):cv(v)for k,v in x.items()}
 if isinstance(x,(tuple,list)):return[cv(v)for v in x]
 return x
def save(n,d):
 with (O/n).open('w')as f:json.dump(cv(d),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def arrays(n,d):
 p=O/(n+'.npz')
 with p.open('wb')as f:np.savez_compressed(f,**d);f.flush();os.fsync(f.fileno())
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def emit(d):
 with (O/'NEW-synthetic-oracle-journal.jsonl').open('a')as f:json.dump(cv(d),f,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
from fractions import Fraction as F
parent=S/'e037-full-pose-construction-contracts-v1'
retained={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in parent.glob('*RETURN.json')}
save('NEW-retained-prefix-and-failed-packing-provenance.json',dict(retained=retained,completed_two_contracts_NOT_replayed=True,failed_third_control_remains_FAILED=True,no_model_constructor_or_solver_rerun=True))
d=json.loads((parent/'NEW-full-pose-canonical-SI-and-whole-SOC-conditioning-RETURN.json').read_text());r=F.from_float(.17);u=F(1,2**53);gamma=2*u/(1-2*u);actual=d['geometry_row'];J=list(map(F.from_float,[.4,-.2,.1,.07,-.03,.02]));proofs=[]
for j in range(6):
 exact=-J[j]*(r if j<3 else 1)/r;returned=F.from_float(actual[426+j]);error=abs(returned-exact);bound=gamma*abs(exact);proofs.append(dict(column=426+j,exact=str(exact),returned=str(returned),error=str(error),derived_two_operator_normal_RN_bound=str(bound),passed=error<=bound))
save('NEW-exact-SAVED-row-operator-roundoff-RETURN.json',dict(proofs=proofs,no_synthetic_model_replay=True,old_bit_identity_expectation_remains_FAILED=True,no_actual_coefficients_or_original_gate_tolerances_repaired=True));assert all(p['passed']for p in proofs)
# Actual NEW pure-rotation synthetic input through unchanged training operators.
verts=np.array([[-.05,-.04,0],[.05,-.04,0],[.03,.05,0],[-.04,.03,.02]]);cam=np.eye(4);cam[:3,3]=[0,0,.6];yy,xx=np.indices((16,16));domain=(xx+yy)%2==0;points=np.column_stack([np.linspace(-.03,.03,120),np.linspace(.02,-.02,120),np.full(120,.6)]);stat=dict(training_intersection=5,training_union=20,training_mesh_vertex_median_m=.1,training_quality=.1);saved=[]
def capture(n,d):
 result=arrays(n,d);saved.append((n,d));return result
D=dict(V=verts,faces=np.array([[0,1,2],[0,2,3]]),K=np.array([[70.,0,8],[0,70.,8],[0,0,1]]),camera=np.array([cam]),Rcw=np.array([np.eye(3)]),stats=[stat]);load=lambda f:dict(training_mask=domain,domain=domain,points_C_m=points);oracle=FullPoseTrainingOracle(D,load,emit,capture);a=np.array([.12,-.08,.03]);out=oracle.query(0,np.zeros(3),a,'NEW-synthetic-full-pose');pose=next(d for n,d in saved if n.endswith('NEW-camera-pose-RETURN'));expectedR=Rotation.from_rotvec(a).as_matrix();e=float(np.max(abs(pose['camera_from_object'][:3,:3]-expectedR)));save('NEW-full-pose-actual-training-wrapper-RETURN.json',dict(actual_RETURN=out,camera_rotation_error=e,passed=out['status']=='NEW_actual_full_pose_TRAINING_ONLY'and e<1e-15,not_episode_or_original_fullpixel_query=True));assert out['status']=='NEW_actual_full_pose_TRAINING_ONLY'and e<1e-15
# Cache-only zero dispatch is tested with an exception-producing data loader;
# it never invokes a baseline raster/NN operation.
zero=FullPoseTrainingOracle(D,lambda _:(_ for _ in ()).throw(RuntimeError('must not load')),emit,capture).query(0,np.zeros(3),np.zeros(3),'NEW-synthetic-zero');save('NEW-zero-full-pose-cache-provenance-RETURN.json',dict(actual_RETURN=zero,passed=zero['status']=='CACHED_F_BY_HASH'));assert zero['status']=='CACHED_F_BY_HASH'
save('report.json',dict(schema='v2d.NEW-full-pose-construction-remaining-contracts.v2',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEW_contracts=3,previous_completed_contracts_reused_BY_HASH=2,previous_failed_bit_identity_control_retained=True,no_episode_pose_or_solver=True));print(json.dumps(dict(NEW_contracts=3,completed_two_reused=True)))

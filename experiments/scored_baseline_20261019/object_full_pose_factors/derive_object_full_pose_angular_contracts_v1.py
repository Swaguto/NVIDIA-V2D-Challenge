"""Two NEW derivative contracts needed by the concrete full-pose construction."""
from pathlib import Path
import datetime,json,hashlib,os
import numpy as np
from scipy.spatial.transform import Rotation
from object_full_pose_maps_v1 import skew,compose
from object_full_pose_angular_v1 import angular_clock_maps
S=Path(__file__).resolve().parents[1];O=S/'e037-object-full-pose-angular-contracts-v1'
def save(n,d):
 def cv(x):
  if isinstance(x,np.ndarray):return x.tolist()
  if isinstance(x,np.generic):return x.item()
  if isinstance(x,dict):return{k:cv(v)for k,v in x.items()}
  return x
 with (O/n).open('w')as f:json.dump(cv(d),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 (O/n).chmod(0o600)
t=np.array([0.,.071,.192,.314]);R=Rotation.from_rotvec([[.2,.1,-.1],[.21,.07,-.03],[.23,.02,.05],[.28,-.06,.12]]).as_matrix();maps=angular_clock_maps(R,t);h=2e-7;FD=[]
for k in range(12):
 i,j=divmod(k,3);e=np.eye(3)[j]*h;P=R.copy();M=R.copy();P[i]=R[i]@Rotation.from_rotvec(e).as_matrix();M[i]=R[i]@Rotation.from_rotvec(-e).as_matrix();p=angular_clock_maps(P,t);m=angular_clock_maps(M,t);FD.append(np.r_[(p['angular_velocity']-m['angular_velocity']).ravel(),(p['angular_acceleration']-m['angular_acceleration']).ravel()]/(2*h))
F=np.stack(FD,axis=1);A=np.vstack([maps['angular_velocity_J'].toarray(),maps['angular_acceleration_J'].toarray()]);error=float(np.max(abs(F-A)));save('NEW-original-angular-and-midclock-Jacobian-RETURN.json',dict(FD=F,analytic=A,maximum_error=error,diagnostic_bound=2e-7,NOT_original_motion_tolerance=True,passed=error<2e-7));assert error<2e-7
T=np.eye(4);T[:3,:3]=R[1];T[:3,3]=[.1,-.2,.6];L=Rotation.from_rotvec([-.2,.1,.4]).as_matrix();p=np.array([.12,-.18,.64]);y=T[:3,:3].T@(p-T[:3,3]);J=np.column_stack([-T[:3,:3].T@L,skew(y)]);fd=[]
for k in range(6):
 d=np.zeros(3);a=np.zeros(3)
 if k<3:d[k]=h
 else:a[k-3]=h
 P=compose(T,d,a,L);M=compose(T,-d,-a,L);fd.append((P[:3,:3].T@(p-P[:3,3])-M[:3,:3].T@(p-M[:3,3]))/(2*h))
F=np.stack(fd,axis=1);error2=float(np.max(abs(F-J)));save('NEW-source-local-full-six-coordinate-Jacobian-RETURN.json',dict(FD=F,analytic=J,maximum_error=error2,diagnostic_bound=2e-9,passed=error2<2e-9,NN_rank_witnesses_NOT_fixed_truth=True));assert error2<2e-9
save('report.json',dict(schema='v2d.NEW-full-pose-angular-source-derivative-contracts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEW_contracts=2,errors=[error,error2],no_episode_pose_source_NN_raster_FK_native_solver=True));print(json.dumps(dict(NEW_contracts=2,errors=[error,error2])))

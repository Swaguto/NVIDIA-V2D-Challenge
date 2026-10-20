"""NEW concrete rotating-support/canonical/actual-wrapper contracts only."""
from pathlib import Path
import json,os,hashlib,datetime
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from full_pose_construction_v1 import rotating_support_pair,canonical
from full_pose_training_oracle_v1 import FullPoseTrainingOracle
S=Path(__file__).resolve().parents[1];O=S/'e037-full-pose-construction-contracts-v1'
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
H=np.array([[-1.,.9,.1],[1.,.8,-.1],[.3,1.1,.03]]);NH=np.array([[1.,0,0]]);V=np.array([[-.1,-.1,-.1],[.12,-.07,.09],[-.02,.11,.04],[.03,.02,-.12]]);NO=np.array([[0.,1.,0.]]);T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[.2,-.1,.3]).as_matrix();T[:3,3]=[.02,.03,.01];stage=Rotation.from_rotvec([.2,.1,-.3]).as_matrix();b=rotating_support_pair(H,NH,V,NO,T,stage);h=2e-7;fd=[]
for j in range(6):
 P=T.copy();M=T.copy()
 if j<3:P[:3,3]+=stage[:,j]*h;M[:3,3]-=stage[:,j]*h
 else:
  e=np.eye(3)[j-3]*h;P[:3,:3]=T[:3,:3]@Rotation.from_rotvec(e).as_matrix();M[:3,:3]=T[:3,:3]@Rotation.from_rotvec(-e).as_matrix()
 fd.append((rotating_support_pair(H,NH,V,NO,P,stage)['gap']-rotating_support_pair(H,NH,V,NO,M,stage)['gap'])/(2*h))
err=float(np.max(abs(np.array(fd)-b['J'])));save('NEW-moving-object-normal-total-derivative-RETURN.json',dict(actual=b,FD=fd,error=err,passed=b['branch_family']==1 and err<2e-9));assert b['branch_family']==1 and err<2e-9
# A branch tie stays explicit; derivative is only the selected search branch.
H2=np.array([[-2.,.7,0],[2.,.7,0]]);T2=np.eye(4);b2=rotating_support_pair(H2,NH,V,NO,T2,np.eye(3));save('NEW-support-tie-and-both-sign-union-RETURN.json',dict(actual=b2,passed=b2['support_tie_counts'][1]==2 and len(b2['all_axis_gaps'])==4,not_native_or_smoothness_certificate=True));assert b2['support_tie_counts'][1]==2 and len(b2['all_axis_gaps'])==4
N=151;r=.17;data=dict(radius=r,world=np.broadcast_to(np.eye(4),(N,4,4)),pairs=np.array([[71,0,1]]),translation_median_gradient=np.tile([.2,-.3,.4],(N,1)),rotation_median_gradient=np.tile([.03,.02,-.01],(N,1)),min_depth_vertex=np.zeros(N,int),source_rotation_J=np.zeros((N,1,3,3)),Rcw=np.broadcast_to(np.eye(3),(N,3,3)),min_C0_z=np.ones(N)*.5,linear_M=sparse.csr_matrix((897,453)),angular_M=sparse.csr_matrix((897,453)),linear_base=np.zeros(897),angular_base=np.zeros(897),motion_caps=[1.,2.,3.,4.,.8,1.7]);geo=dict(gaps=np.array([-.004]),J=np.array([[.4,-.2,.1,.07,-.03,.02]]));m=canonical(data,geo);row=m['A'].getrow(2).toarray().ravel();expected=np.zeros(907);expected[426:432]=-geo['J'][0]*np.r_[np.ones(3),np.full(3,1/r)];expected[-1]=-1;ok=np.array_equal(row,expected)and abs(m['b'][2]-(-.003/r))<1e-16 and len(m['cones'])==900
save('NEW-full-pose-canonical-SI-and-whole-SOC-conditioning-RETURN.json',dict(rows=m['A'].shape[0],columns=m['columns'],nonnegative=m['nonnegative'],SOC4=len(m['cones']),geometry_row=row,expected=expected,configuration=m['config'],passed=ok));assert ok
# Actual NEW pure-rotation synthetic input through unchanged training operators.
verts=np.array([[-.05,-.04,0],[.05,-.04,0],[.03,.05,0],[-.04,.03,.02]]);cam=np.eye(4);cam[:3,3]=[0,0,.6];yy,xx=np.indices((16,16));domain=(xx+yy)%2==0;points=np.column_stack([np.linspace(-.03,.03,120),np.linspace(.02,-.02,120),np.full(120,.6)]);stat=dict(training_intersection=5,training_union=20,training_mesh_vertex_median_m=.1,training_quality=.1);saved=[]
def capture(n,d):
 result=arrays(n,d);saved.append((n,d));return result
D=dict(V=verts,faces=np.array([[0,1,2],[0,2,3]]),K=np.array([[70.,0,8],[0,70.,8],[0,0,1]]),camera=np.array([cam]),Rcw=np.array([np.eye(3)]),stats=[stat]);load=lambda f:dict(training_mask=domain,domain=domain,points_C_m=points);oracle=FullPoseTrainingOracle(D,load,emit,capture);a=np.array([.12,-.08,.03]);out=oracle.query(0,np.zeros(3),a,'NEW-synthetic-full-pose');pose=next(d for n,d in saved if n.endswith('NEW-camera-pose-RETURN'));expectedR=Rotation.from_rotvec(a).as_matrix();e=float(np.max(abs(pose['camera_from_object'][:3,:3]-expectedR)));save('NEW-full-pose-actual-training-wrapper-RETURN.json',dict(actual_RETURN=out,camera_rotation_error=e,passed=out['status']=='NEW_actual_full_pose_TRAINING_ONLY'and e<1e-15,not_episode_or_original_fullpixel_query=True));assert out['status']=='NEW_actual_full_pose_TRAINING_ONLY'and e<1e-15
# Cache-only zero dispatch is tested with an exception-producing data loader;
# it never invokes a baseline raster/NN operation.
zero=FullPoseTrainingOracle(D,lambda _:(_ for _ in ()).throw(RuntimeError('must not load')),emit,capture).query(0,np.zeros(3),np.zeros(3),'NEW-synthetic-zero');save('NEW-zero-full-pose-cache-provenance-RETURN.json',dict(actual_RETURN=zero,passed=zero['status']=='CACHED_F_BY_HASH'));assert zero['status']=='CACHED_F_BY_HASH'
save('report.json',dict(schema='v2d.NEW-full-pose-construction-contracts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEW_contracts=5,moving_normal_derivative_error=err,no_episode_pose_source_NN_raster_solver_native_physics_GPU=True));print(json.dumps(dict(NEW_contracts=5,success=True)))

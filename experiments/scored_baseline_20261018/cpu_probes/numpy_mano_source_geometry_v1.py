"""Private CPU MANO geometry experiment using standard linear blend skinning.

Licensed model arrays are loaded only from the owner's local mount. No model
arrays are exported. Runtime compatibility aliases are confined to this process.
The result must close against saved native geometry before any fitting use.
"""
from pathlib import Path
import inspect,pickle,sys
import numpy as np
from scipy.spatial.transform import Rotation
S=Path(__file__).resolve().parents[1]
class NumpyMano:
 def __init__(self,path):
  sys.path.insert(0,str(S/'mano-cpu-deps-v1'))
  if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
  for k,v in {'bool':np.bool_,'int':int,'float':float,'complex':complex,'object':object,'unicode':str,'str':str}.items():
   if k not in np.__dict__:setattr(np,k,v)
  with open(path,'rb') as f:d=pickle.load(f,encoding='latin1')
  self.template=np.asarray(d['v_template']);self.shape=np.asarray(d['shapedirs']);self.pose=np.asarray(d['posedirs']);self.regressor=d['J_regressor'].toarray();self.weights=d['weights'];self.parents=d['kintree_table'][0].astype(np.int64);self.faces=d['f'];self.reorder=np.array([0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]);self._cache={}
 def shaped(self,betas):
  key=tuple(np.array(betas).ravel())
  if key not in self._cache:
   v=self.template+np.einsum('vci,i->vc',self.shape,betas);self._cache[key]=(v,self.regressor@v)
  return self._cache[key]
 def __call__(self,pose,betas):
  shaped,rest=self.shaped(betas);rots=Rotation.from_rotvec(np.array(pose).reshape(16,3)).as_matrix();posed=shaped+np.einsum('vci,i->vc',self.pose,(rots[1:]-np.eye(3)).ravel());g=np.tile(np.eye(4),(16,1,1))
  for k in range(16):
   relative=np.eye(4);relative[:3,:3]=rots[k];relative[:3,3]=rest[k] if k==0 else rest[k]-rest[self.parents[k]];g[k]=relative if k==0 else g[self.parents[k]]@relative
  a=g.copy();a[:,:3,3]-=np.einsum('kij,kj->ki',g[:,:3,:3],rest);skin=np.einsum('vk,kij->vij',self.weights,a);hom=np.c_[posed,np.ones(778)];vertices=np.einsum('vij,vj->vi',skin,hom)[:,:3];joints=np.concatenate([g[:,:3,3],vertices[[745,317,444,556,673]]])[self.reorder]
  return vertices,joints,g
 def record(self,record,pose=None,translation=None):
  p=np.r_[record['mano']['global_orient'],record['mano']['hand_pose']] if pose is None else pose;v,j,g=self(p,record['mano']['betas']);centre=v.mean(0);scale=record['hand_scale'];t=np.array(record['cam_t'] if translation is None else translation);return (v-centre)*scale+centre+t,(j-centre)*scale+centre+t,g
if __name__=='__main__':
 import hashlib,json
 O=S/'e037-numpy-mano-closure-v1';O.mkdir();B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz';m=NumpyMano(S/'models/mano/models/MANO_RIGHT.pkl');rows=[]
 with np.load(B) as z:
  for f in [0,24,35,70,100,150]:
   record=json.load(open(B.parent/f'source/hands/right/{f:06d}.json'));v,j,_=m.record(record);c=z['camera_c2w'][f];nativej=(z['hand_joints_world'][f,1]-c[:3,3])@c[:3,:3];nativev=(z['hand_vertices_world'][f,1]-c[:3,3])@c[:3,:3];rows.append(dict(frame=f,joint_max_difference_m=float(abs(j-nativej).max()),vertex_max_difference_m=float(abs(v-nativev).max())))
 report=dict(schema='v2d.numpy-mano-source-closure.v1',producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_sha256=hashlib.sha256((S/'models/mano/models/MANO_RIGHT.pkl').read_bytes()).hexdigest(),rows=rows,all_saved_native_geometry_close=all(x['joint_max_difference_m']<1e-6 and x['vertex_max_difference_m']<1e-6 for x in rows));(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['all_saved_native_geometry_close']

"""NEW all-pair rotating compiled face-union runtime; not a native audit."""
import numpy as np
from object_full_pose_maps_v1 import compose
from full_pose_construction_v1 import rotating_support_pair

def geometry_all(d,parameters):
 p=np.asarray(parameters,float)
 if p.shape!=(151,6)or not np.isfinite(p).all():raise ValueError('complete finite declared full-pose parameters required')
 fixed={};objects={};poses=[]
 for f in range(151):
  for g in d['fixed_geoms']:
   T=d['body'][f,g];fixed[f,g]=(d['vertices'][g]@T[:3,:3].T+T[:3,3],d['normals'][g]@T[:3,:3].T)
  row=[]
  for g in d['object_geoms']:
   T=compose(d['body'][f,g],p[f,:3],p[f,3:],d['stage']);objects[f,g]=T;row.append(T)
  poses.append(row)
 gaps=[];jac=[];normal=[];families=[];branches=[];tie=[];supportties=[];iv=[];ih=[];allg=[];ptr=[0]
 for f,a,b in d['pairs']:
  H,NH=fixed[int(f),int(a)];out=rotating_support_pair(H,NH,d['vertices'][int(b)],d['normals'][int(b)],objects[int(f),int(b)],d['stage']);gaps.append(out['gap']);jac.append(out['J']);normal.append(out['normal']);families.append(out['branch_family']);branches.append(out['branch']);tie.append(out['branch_tie_count']);supportties.append(out['support_tie_counts']);iv.append(out['object_support_vertex']);ih.append(out['fixed_support_vertex']);allg.append(out['all_axis_gaps']);ptr.append(ptr[-1]+len(out['all_axis_gaps']))
 return dict(gaps=np.array(gaps),J=np.array(jac),normals=np.array(normal),families=np.array(families,np.int8),branches=np.array(branches,np.int32),branch_tie_counts=np.array(tie,np.int32),support_tie_counts=np.array(supportties,np.int32),object_support_vertex=np.array(iv,np.int32),fixed_support_vertex=np.array(ih,np.int32),all_axis_gaps=np.concatenate(allg),axis_ptr=np.array(ptr,np.int64),NEW_object_body_poses=np.array(poses),parameters=p.copy(),proxy_phase_m=float(np.maximum(0,-.001-np.array(gaps)).max(initial=0)))

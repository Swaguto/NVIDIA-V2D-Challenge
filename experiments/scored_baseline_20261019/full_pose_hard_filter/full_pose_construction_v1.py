"""NEW single full-pose proposal construction; geometry/search != native gates."""
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from object_full_pose_maps_v1 import skew,compose,source_dispatch

def rotating_support_pair(H,NH,V,NO,T,stage):
 """All retained compiled FACE normals, both signs; no edge-SAT/native claim.

Returns a NEW rotation-dependent union proxy and derivative of its deterministic
first active branch. Every support/rank tie remains explicit. H/NH fixed native
body returns; V/NO already include geom-to-body placement.
 """
 R=T[:3,:3];p=T[:3,3];O=V@R.T+p
 nH=np.vstack([NH,-NH]);nH=nH/np.linalg.norm(nH,axis=1)[:,None]
 raw=np.vstack([NO,-NO])@R.T;length=np.linalg.norm(raw,axis=1);nO=raw/length[:,None]
 PO=O@nH.T;PH=H@nH.T;gH=PO.min(axis=0)-PH.max(axis=0)
 OH=H@nO.T;OO=O@nO.T;gO=OH.min(axis=0)-OO.max(axis=0)
 allg=np.r_[gH,gO];k=int(np.argmax(allg));ties=np.flatnonzero(allg==allg[k]);J=np.zeros(6)
 if k<len(nH):
  normal=nH[k];iv=int(np.argmin(PO[:,k]));ih=int(np.argmax(PH[:,k]));J[:3]=normal@stage;J[3:]=normal@(-R@skew(V[iv]));family=0;supportties=[int(np.count_nonzero(PO[:,k]==PO[iv,k])),int(np.count_nonzero(PH[:,k]==PH[ih,k]))]
 else:
  j=k-len(nH);normal=nO[j];iv=int(np.argmax(OO[:,j]));ih=int(np.argmin(OH[:,j]));body_normal=np.vstack([NO,-NO])[j];dn=(np.eye(3)-np.outer(normal,normal))@(-R@skew(body_normal))/length[j];J[:3]=-normal@stage;J[3:]=(H[ih]-O[iv])@dn-normal@(-R@skew(V[iv]));family=1;supportties=[int(np.count_nonzero(OO[:,j]==OO[iv,j])),int(np.count_nonzero(OH[:,j]==OH[ih,j]))]
 return dict(gap=float(allg[k]),J=J,all_axis_gaps=allg,branch=k,branch_family=family,branch_tie_count=len(ties),support_tie_counts=supportties,object_support_vertex=iv,fixed_support_vertex=ih,normal=normal,object_vertices=O,rotating_normals=nO)

def metric_radius(vertices_by_geom):
 return max(float(np.linalg.norm(v,axis=1).max())for v in vertices_by_geom)

def canonical(data,geometry):
 """907 scaled cols: delta/r, theta, rho/r. All rows condition whole cones."""
 r=data['radius'];N=len(data['world']);n=6*N+1;rr=[];cc=[];vv=[];b=[];orig=[]
 def row(entries,rhs,kind):
  i=len(b)
  for j,x in entries:
   if x!=0:rr.append(i);cc.append(j);vv.append(float(x))
  b.append(float(rhs));orig.append(kind)
 row([(n-1,-1)],0.,'rho_lower');row([(n-1,1)],1.,'rho_upper')
 for f,(gap,J)in enumerate(zip(geometry['gaps'],geometry['J'])):
  frame=int(data['pairs'][f,0]);row([(6*frame+j,-J[j]*(r if j<3 else 1)/r)for j in range(6)]+[(n-1,-1)],(gap+.001)/r,'rotating_support_search')
 for f in range(N):
  gm=data['translation_median_gradient'][f];gr=data['rotation_median_gradient'][f]
  row([(6*f+j,gm[j])for j in range(3)]+[(6*f+3+j,gr[j]/r)for j in range(3)],0.,'median_SEARCH')
  # One current minimum-depth branch, never a full-vertex source certificate.
  v=int(data['min_depth_vertex'][f]);Jz=data['source_rotation_J'][f,v,2]
  row([(6*f+j,-data['Rcw'][f,2,j])for j in range(3)]+[(6*f+3+j,-Jz[j]/r)for j in range(3)],(data['min_C0_z'][f]-.01)/r,'depth_SEARCH')
 nonnegative=len(b);cones=[]
 def soc(top,components,kind):
  start=len(b);row([],top,kind)
  for entries,rhs in components:row(entries,rhs,kind)
  cones.append(dict(start=start,size=4,kind=kind))
 M=data['linear_M'].tocsr();ang=data['angular_M'].tocsr();base=data['linear_base'];abase=data['angular_base'];caps=data['motion_caps']
 for group,(A,c0,types)in enumerate([(M,base,(0,2)),(ang,abase,(1,3))]):
  for k in range(299):
   cap=caps[types[0]if k<150 else types[1]];components=[]
   for a in range(3):
    row_index=3*k+a;indices=A.indices[A.indptr[row_index]:A.indptr[row_index+1]];values=A.data[A.indptr[row_index]:A.indptr[row_index+1]];items=[]
    for col,val in zip(indices,values):
     frame,j=divmod(int(col),3);items.append((6*frame+(j if group==0 else j+3),-float(val)*(r if group==0 else 1)/cap))
    components.append((items,float(c0[row_index])/cap))
   soc(1.,components,'linear_motion'if group==0 else'angular_motion_SEARCH')
 for f in range(N):
  soc(1/8,[([(6*f+j,-1.)],0.)for j in range(3)],'translation_norm_CONSTRUCTION')
  soc(np.pi/12,[([(6*f+3+j,-1.)],0.)for j in range(3)],'rotation_norm_CONSTRUCTION')
 A=sparse.coo_matrix((vv,(rr,cc)),shape=(len(b),n)).tocsc();q=np.zeros(n);q[-1]=1.;P=sparse.diags(np.r_[np.full(6*N,1e-3),0.],format='csc')
 return dict(A=A,b=np.array(b),q=q,P=P,nonnegative=nonnegative,cones=cones,row_kinds=orig,radius=r,columns=n,conditioning='delta/r, theta, rho/r; geometric+median+depth rows /r; whole motion SOC /originalcap',config=dict(translation_norm_halfwidth_m=r/8,rotation_norm_halfwidth_rad=float(np.pi/12),rho_upper_m=r,QP_iterations=100,QP_seconds=60.,threads=1,default_tolerances=True,quadratic_dimensionless_regularization=1e-3,episode_QP_calls=1,episode_nonzero_trial_maximum=1,line_search_or_restart=False))

def actual_motion(poses,times,caps):
 t=np.asarray(times);dt=np.diff(t);R=Rotation.from_matrix(poses[:,:3,:3]);v=np.diff(poses[:,:3,3],axis=0)/dt[:,None];w=(R[1:]*R[:-1].inv()).as_rotvec()/dt[:,None];mid=(dt[:-1]+dt[1:])/2;a=np.diff(v,axis=0)/mid[:,None];alpha=np.diff(w,axis=0)/mid[:,None];arrays=dict(linear_velocity=v,angular_velocity=w,linear_acceleration=a,angular_acceleration=alpha);values=[np.linalg.norm(x,axis=1).max(initial=0)for x in [v,w,a,alpha]]+[np.quantile(np.linalg.norm(v,axis=1),.95),np.quantile(np.linalg.norm(w,axis=1),.95)];passes=[bool(x<=cap+1e-9)for x,cap in zip(values,caps)];return dict(arrays=arrays,values=values,passes=passes,comparison=1e-9)

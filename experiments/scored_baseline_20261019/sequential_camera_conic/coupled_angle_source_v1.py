"""New full-clock angle/actual-URDF source model primitives; no optimizer.

Robot angles are derived variables. Source flags/depth and original tolerances
are not inferred, repaired or changed by this representation.
"""
from fractions import Fraction as F
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

def rat(x):
 x=F(x);return dict(numerator=str(x.numerator),denominator=str(x.denominator))
def unrat(r):return F(int(r['numerator']),int(r['denominator']))

class Chain:
 def __init__(self,xml,joint_names):
  self.xml=ET.fromstring(xml);self.names=list(joint_names)
  joints=list(self.xml.findall('joint'));children={j.find('child').attrib['link']:j for j in joints}
  roots={p.attrib['name']for p in self.xml.findall('link')}-set(children)
  if len(roots)!=1 or len(set(self.names))!=len(self.names):raise ValueError('single rooted URDF and unique explicit joints required')
  self.root=next(iter(roots));self.chains={}
  for link in {p.attrib['name']for p in self.xml.findall('link')}:
   chain=[];cur=link;seen=set()
   while cur!=self.root:
    if cur in seen or cur not in children:raise ValueError('disconnected or cyclic URDF')
    seen.add(cur);j=children[cur];kind=j.attrib['type'];origin=j.find('origin');T=np.eye(4)
    if origin is not None:
     T[:3,3]=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ');T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.attrib.get('rpy','0 0 0'),sep=' ')).as_matrix()
    if not np.isfinite(T).all():raise ValueError('finite unchanged URDF origins required')
    axis=None
    if kind!='fixed':
     if kind!='revolute' or j.attrib['name']not in self.names:raise ValueError('explicit fixed/revolute names required')
     axis=np.fromstring(j.find('axis').attrib['xyz'],sep=' ')
     if axis.shape!=(3,)or not np.isfinite(axis).all()or abs(float(axis@axis)-1)>1e-14:raise ValueError('actual unit joint axis required; no morphology repair')
    chain.append(dict(name=j.attrib['name'],kind=kind,origin=T,axis=axis));cur=j.find('parent').attrib['link']
   self.chains[link]=list(reversed(chain))
 def point_jacobian(self,root_pose,angles,link):
  root_pose=np.asarray(root_pose,dtype=float);angles=np.asarray(angles,dtype=float)
  if root_pose.shape!=(7,)or angles.shape!=(len(self.names),)or not np.isfinite(np.r_[root_pose,angles]).all():raise ValueError('finite complete root/joint arrays required')
  if np.linalg.norm(root_pose[3:])==0:raise ValueError('nonzero root quaternion required')
  T=np.eye(4);T[:3,:3]=Rotation.from_quat(root_pose[3:]).as_matrix();T[:3,3]=root_pose[:3];positions=dict(zip(self.names,angles,strict=True));axes={};origins={};records=[]
  for joint in self.chains[link]:
   T=T@joint['origin']
   if joint['kind']=='revolute':
    axes[joint['name']]=T[:3,:3]@joint['axis'];origins[joint['name']]=T[:3,3].copy();motion=np.eye(4);motion[:3,:3]=Rotation.from_rotvec(joint['axis']*positions[joint['name']]).as_matrix();T=T@motion
   records.append(dict(joint=joint['name'],transform=T.copy()))
  p=T[:3,3].copy();J=np.zeros((3,len(self.names)))
  for i,name in enumerate(self.names):
   if name in axes:J[:,i]=np.cross(axes[name],p-origins[name])
  return dict(point=p,jacobian=J,ancestor_joint_names=[v['name']for v in self.chains[link]],world_axes=axes,world_origins=origins,intermediate_transforms=records)

def camera_source_factor(point,jacobian,camera,stage,K,pixel,cap,original_image_tolerance=1e-6,minimum_depth=.01):
 """Smooth squared-cone inequality PLUS a separate STRICT positive-depth gate.

 Radius is the original cap + original pixel tolerance. It is never an
 algorithmic merit threshold. All returned points/Jacobians are diagnostics.
 """
 point=np.asarray(point,dtype=float);jacobian=np.asarray(jacobian,dtype=float);camera=np.asarray(camera,dtype=float);stage=np.asarray(stage,dtype=float);K=np.asarray(K,dtype=float);pixel=np.asarray(pixel,dtype=float)
 if point.shape!=(3,)or jacobian.ndim!=2 or jacobian.shape[0]!=3 or camera.shape!=(4,4)or stage.shape!=(3,3)or K.shape!=(3,3)or pixel.shape!=(2,)or not np.isfinite(np.r_[point,jacobian.ravel(),camera.ravel(),stage.ravel(),K.ravel(),pixel,cap,original_image_tolerance,minimum_depth]).all():raise ValueError('explicit finite source geometry required')
 if cap<0 or original_image_tolerance!=1e-6 or minimum_depth!=.01 or K[0,0]<=0 or K[1,1]<=0:raise ValueError('unchanged original image/depth authority required')
 radius=float(cap+original_image_tolerance);A=camera[:3,:3].T@stage.T;C=A@point-camera[:3,:3].T@camera[:3,3];CJ=A@jacobian
 B=np.array([[K[0,0],0.,K[0,2]-pixel[0]],[0.,K[1,1],K[1,2]-pixel[1]]]);X=B@C;Q=radius*radius*np.outer([0.,0.,1.],[0.,0.,1.])-B.T@B;g=float(radius*radius*C[2]*C[2]-X@X);gradient=2*(Q@C)@CJ;depth=float(C[2]-.01)
 original_pixel_error=None
 if C[2]>.01:original_pixel_error=float(np.linalg.norm(np.array([K[0,0]*C[0]/C[2]+K[0,2],K[1,1]*C[1]/C[2]+K[1,2]])-pixel))
 return dict(camera_point=C,camera_jacobian=CJ,source_cone_value=g,source_cone_jacobian=gradient,strict_depth_value=depth,strict_depth_jacobian=CJ[2].copy(),original_cap_px=float(cap),original_image_tolerance_px=1e-6,constructed_radius_px=radius,original_minimum_depth_m=.01,direct_image_error_px=original_pixel_error,negative_or_zero_depth_is_NOT_source_pass=True)

def linear_angle_rows(original,fixed,timestamps,names,limits,free,emit):
 """All original angle boxes and time rows, exact saved-binary fixed folding.

 Every fixed row is retained and checked by a caller. No fixed boundary is a
 measured joint. Velocity tolerance has rad/s units before multiplication dt.
 """
 original=np.asarray(original);fixed=np.asarray(fixed);ts=list(map(F,timestamps));names=list(names);n=len(ts)
 if original.shape!=fixed.shape or original.shape!=(n,len(names))or n<3 or len(set(names))!=len(names):raise ValueError('complete unique full-clock dimensions required')
 if not(np.isfinite(original).all()and np.isfinite(fixed).all())or any(b<=a for a,b in zip(ts,ts[1:])):raise ValueError('finite values and positive actual clock required')
 free=set(free)
 if any(not(isinstance(t,int)and 0<=t<n and j in names)for t,j in free):raise ValueError('invalid free variable')
 tol=F(float(1e-7));positiontol=F(float(1e-8));component=F(float(.25))+tol;change=F(float(.05))+tol;rows=[]
 def offer(kind,joint,terms,bound,origin):
  proposed=dict(kind=kind,joint=joint,terms=[dict(frame=t,coefficient=rat(k))for t,k in terms],upper=rat(bound),origin=origin)
  emit(dict(kind='BEFORE_NEW_angle_linear_row',proposal=proposed));active=[];folded=F(0)
  for t,k in terms:
   if(t,joint)in free:active.append(dict(variable=f'q_{t}_{joint}',frame=t,coefficient=rat(k)))
   else:folded+=k*F(float(fixed[t,names.index(joint)]))
  returned=dict(id=len(rows),**proposed,active_terms=active,fixed_sum=rat(folded),folded_upper=rat(bound-folded),constant_only=not active,constant_only_passed=bool(folded<=bound)if not active else None)
  emit(dict(kind='AFTER_NEW_angle_linear_row',returned=returned));rows.append(returned)
 for c,joint in enumerate(names):
  lower,upper,velocity=[F(float(x))for x in limits[joint]]
  if lower>=upper or velocity<=0:raise ValueError('actual URDF position/speed authority required')
  for t in range(n):
   baseline=F(float(original[t,c]));offer('actual_URDF_position_upper',joint,[(t,F(1))],upper+positiontol,dict(position_tolerance_rad=rat(positiontol)));offer('actual_URDF_position_lower',joint,[(t,F(-1))],-lower+positiontol,dict(position_tolerance_rad=rat(positiontol)));offer('original_component_upper',joint,[(t,F(1))],baseline+component,dict(original_angle=rat(baseline),component_cap_rad=rat(F(float(.25))),original_motion_tolerance_rad=rat(tol)));offer('original_component_lower',joint,[(t,F(-1))],-baseline+component,dict(original_angle=rat(baseline),component_cap_rad=rat(F(float(.25))),original_motion_tolerance_rad=rat(tol)))
  for t in range(1,n):
   dt=ts[t]-ts[t-1];base=F(float(original[t,c]))-F(float(original[t-1,c]));terms=[(t,F(1)),(t-1,F(-1))]
   for sign in[-1,1]:
    offer('original_adjacent_added',joint,[(a,sign*k)for a,k in terms],sign*base+change,dict(sign=sign,clocks=[t,t-1],original_angles=[rat(F(float(original[a,c])))for a in[t,t-1]],added_cap_rad=rat(F(float(.05))),motion_tolerance_rad=rat(tol)))
    offer('actual_URDF_velocity',joint,[(a,sign*k)for a,k in terms],(velocity+tol)*dt,dict(sign=sign,clocks=[t,t-1],exact_positive_dt_s=rat(dt),actual_URDF_velocity_rad_s=rat(velocity),velocity_tolerance_rad_s=rat(tol)))
  for t in range(2,n):
   terms=[(t,F(1)),(t-1,F(-2)),(t-2,F(1))];base=sum((k*F(float(original[a,c]))for a,k in terms),F(0))
   for sign in[-1,1]:offer('original_second_added',joint,[(a,sign*k)for a,k in terms],sign*base+change,dict(sign=sign,clocks=[t,t-1,t-2],original_angles=[rat(F(float(original[a,c])))for a in[t,t-1,t-2]],added_cap_rad=rat(F(float(.05))),motion_tolerance_rad=rat(tol)))
 return rows

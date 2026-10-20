"""NEW object-translation maps; no pose/source/native truth or solver authority."""
from fractions import Fraction as Q
import numpy as np

def exact_translation_motion_maps(timestamps):
 t=[Q.from_float(float(x)) for x in timestamps]
 if len(t)<3 or any(t[i+1]<=t[i] for i in range(len(t)-1)):raise ValueError('strict original clock')
 velocities=[{i:-1/(t[i+1]-t[i]),i+1:1/(t[i+1]-t[i])} for i in range(len(t)-1)]
 acceleration=[]
 for i in range(len(t)-2):
  scale=2/(t[i+2]-t[i]);row={}
  for j,v in velocities[i+1].items():row[j]=row.get(j,Q(0))+scale*v
  for j,v in velocities[i].items():row[j]=row.get(j,Q(0))-scale*v
  acceleration.append({j:v for j,v in row.items() if v})
 return dict(velocity=velocities,acceleration=acceleration)

def scalar_apply(row,positions):
 return [sum(v*Q.from_float(float(positions[j][k])) for j,v in row.items()) for k in range(3)]

def affine_camera_translation(camera_from_world,world_delta):
 r=np.asarray(camera_from_world,float)[:3,:3];return np.einsum('ij,j->i',r,np.asarray(world_delta,float),optimize=False)

def source_pixel_translation_derivative(camera_point,intrinsics,camera_from_world):
 C=np.asarray(camera_point,float);K=np.asarray(intrinsics,float);r=np.asarray(camera_from_world,float)[:3,:3]
 if not np.isfinite(C).all() or C[2]<=0: return dict(status='UNRESOLVED_nonpositive_or_nonfinite_depth')
 J=np.array([[K[0,0]/C[2],0.,-K[0,0]*C[0]/C[2]**2],[0.,K[1,1]/C[2],-K[1,1]*C[1]/C[2]**2]])
 return dict(status='local_derivative_ONLY',J=np.einsum('ij,jk->ik',J,r,optimize=False))

def translated_triangle_ray_map(triangle,ray):
 tri=np.asarray(triangle,float);r=np.asarray(ray,float);edge=np.array([tri[1]-tri[0],tri[2]-tri[0]]);n=np.cross(edge[0],edge[1]);den=float(np.dot(n,r))
 if not np.isfinite(np.r_[tri.ravel(),r]).all() or den==0:return dict(status='UNRESOLVED_parallel_or_degenerate_or_nonfinite',denominator=den)
 # Root is affine in CAMERA translation. Barycentric membership, positive root,
 # every other face/occlusion, and every miss still require their actual checks.
 root=float(np.dot(n,tri[0])/den)
 return dict(status='ONE_plane_root_NOT_front_surface',depth_constant=root,depth_translation_coefficients=n/den,denominator=den,no_face_coverage_occlusion_or_positive_depth_claimed=True)

def translate_support_halfspace(hand_max,object_min,normal,gate_m=.001):
 n=np.asarray(normal,float)
 if not np.isfinite(n).all() or abs(np.linalg.norm(n)-1)>1e-12:raise ValueError('unit support normal required')
 return dict(normal=n,minimum_dot_translation=float(hand_max-object_min-gate_m),scope='sufficient separating support plane only; no dynamics/grasp/exact-contact certificate')

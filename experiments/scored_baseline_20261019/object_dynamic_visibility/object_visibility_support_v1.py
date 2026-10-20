"""NEW fixed-rotation source visibility-cell/support-plane construction helpers.

All maps concern a sufficient construction family. They are not source truth,
full first-surface visibility, native penetration proofs, or robot commands.
"""
import numpy as np
from scipy import sparse

CONFIG=dict(training_rays_per_clock=24,world_translation_construction_halfwidth_m=.02,source_depth_conditioning_m=.01,translation_damping=1e-3,visibility_cell_positive_depth_m=.010000001,packing_construction_guard_m=1e-10,solver_iterations=100,solver_seconds=60.,threads=1)

def ray_cells(triangle,rays):
 tri=np.asarray(triangle,float);ray=np.asarray(rays,float);a=tri[:,0];e1=tri[:,1]-a;e2=tri[:,2]-a;h=np.cross(ray[:,None],e2[None]);det=np.einsum('fi,rfi->rf',e1,h,optimize=False);nonzero=abs(det)>1e-12;inverse=np.divide(1,det,out=np.zeros_like(det),where=nonzero);origin=-a;cross=np.cross(origin,e1)
 u0=np.einsum('fi,rfi->rf',origin,h,optimize=False)*inverse;v0=np.einsum('ri,fi->rf',ray,cross,optimize=False)*inverse;z0=np.einsum('fi,fi->f',e2,cross,optimize=False)[None]*inverse
 ud=-h*inverse[:,:,None];vd=np.stack([np.einsum('ri,fi->rf',ray,np.cross(-np.eye(3)[j],e1),optimize=False)*inverse for j in range(3)],axis=-1);zd=np.stack([np.einsum('fi,fi->f',e2,np.cross(-np.eye(3)[j],e1),optimize=False)[None]*inverse for j in range(3)],axis=-1)
 return dict(u0=u0,v0=v0,z0=z0,ud=ud,vd=vd,zd=zd,nonzero=nonzero,det=det)

def first_cell_for_ray(triangles,ray):
 c=ray_cells(triangles,np.asarray(ray)[None]);u,v,z=c['u0'][0],c['v0'][0],c['z0'][0];hit=c['nonzero'][0]&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(z>0);ids=np.flatnonzero(hit)
 if not len(ids):return dict(status='MISSING_first_surface_kept_infinite',depth_m=float('inf'),all_eligible_faces_retained=True)
 i=int(ids[np.argmin(z[ids])]);return dict(status='ONE_baseline_front_face_local_cell_ONLY',face=i,depth_m=float(z[i]),u0=float(u[i]),v0=float(v[i]),z0=float(z[i]),ud=c['ud'][0,i],vd=c['vd'][0,i],zd=c['zd'][0,i],baseline_hit_faces=ids,baseline_hit_face_depths=z[ids],nonzero_determinant=float(c['det'][0,i]),visibility_and_occlusion_can_change=True)

def support_axis_plane(hand_vertices,object_vertices,axes,gate_m=.001):
 H=np.asarray(hand_vertices,float);O=np.asarray(object_vertices,float);n=np.asarray(axes,float);norm=np.linalg.norm(n,axis=1)
 if np.any(norm<=0)or not np.isfinite(np.r_[H.ravel(),O.ravel(),n.ravel()]).all():raise ValueError('finite nonzero geometric support axes')
 n=n/norm[:,None];lo=np.min(O@n.T,axis=0);hi=np.max(H@n.T,axis=0);gaps=lo-hi;k=int(np.argmax(gaps));return dict(normal=n[k],hand_support_max=float(hi[k]),object_support_min=float(lo[k]),baseline_support_gap_m=float(gaps[k]),minimum_dot_native_translation=float(-gaps[k]-gate_m+CONFIG['packing_construction_guard_m']),selected_axis=k,all_axis_gaps=gaps,scope='ONE sufficient support direction; not exact/native penetration/global necessity')

def p95_speed_envelope(saved_current_speeds,original_p95,original_maximum):
 s=np.asarray(saved_current_speeds,float);q=float(np.quantile(s,.95));slack=float(original_p95-q)
 if slack<0:raise ValueError('starting original object p95 gate not satisfied')
 return dict(current_p95=q,uniform_slack=slack,per_edge_caps=np.minimum(s+slack,original_maximum),scope='sufficient monotone-quantile construction bound; original p95 still checked after solve')

def sparse_record(A):
 A=A.tocsc();return dict(shape=list(A.shape),data=A.data,indices=A.indices,indptr=A.indptr)

def returned(s):
 out={}
 for k in dir(s):
  if k.startswith('_'):continue
  v=getattr(s,k)
  if callable(v):continue
  out[k]=str(v)if k=='status'else v
 return out

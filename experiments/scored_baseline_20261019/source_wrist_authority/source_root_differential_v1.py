"""Explicit world SO(3) residual derivatives; no inferred observation authority."""
import numpy as np

def skew(v):
 x,y,z=np.asarray(v,float);return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

def left_jacobian(v):
 v=np.asarray(v,float);theta=float(np.linalg.norm(v));W=skew(v)
 if theta<1e-5:
  a=.5-theta**2/24+theta**4/720;b=1/6-theta**2/120+theta**4/5040
 else:a=(1-np.cos(theta))/theta**2;b=(theta-np.sin(theta))/theta**3
 return np.eye(3)+a*W+b*np.einsum('ij,jk->ik',W,W,optimize=False)

def pixel_root_jacobian(points_native,root_translation,residual_rotvec,camera_from_native,K):
 points=np.asarray(points_native);A,offset=camera_from_native;J=left_jacobian(residual_rotvec);out=[]
 for p in points:
  X=np.einsum('ij,j->i',A,p,optimize=False)-offset
  if not np.isfinite(X).all() or X[2]<=.01:raise ValueError('Pixel derivative needs a finite visible positive-depth point')
  root=-np.einsum('ij,jk->ik',skew(p-root_translation),J,optimize=False);P=np.array([[K[0,0]/X[2],0,-K[0,0]*X[0]/X[2]**2],[0,K[1,1]/X[2],-K[1,1]*X[1]/X[2]**2]])
  out.append(np.einsum('ij,jk,kl->il',P,A,root,optimize=False))
 return np.stack(out)

"""Exact function snapshots from the completed independent support audit; no old study executes."""
import math
import numpy as np

def point(a,b):return np.array([.0458*np.cos(a)+.035*np.cos(a+b),.0458*np.sin(a)+.035*np.sin(a+b)])

def support(n,ab,bb):
 n=np.asarray(n,dtype=float);phi=float(np.arctan2(n[1],n[0]));a0,a1=ab;b0,b1=bb;assert 0<b0<=b1<np.pi and np.linalg.norm(n)>0;candidates=[]
 def add(a,b,kind):candidates.append(dict(a=float(a),b=float(b),kind=kind,point=point(a,b),projection=float(n@point(a,b))))
 for a in [a0,a1]:
  for b in [b0,b1]:add(a,b,'corner')
 for b in [b0,b1]:
  shift=float(np.arctan2(.035*np.sin(b),.0458+.035*np.cos(b)))
  for k in range(-4,5):
   a=phi-shift+k*np.pi
   if a0<=a<=a1:add(a,b,'distal_boundary_stationary')
 for a in [a0,a1]:
  for k in range(-4,5):
   b=phi-a+k*np.pi
   if b0<=b<=b1:add(a,b,'proximal_boundary_stationary')
 return min(candidates,key=lambda v:v['projection']),candidates

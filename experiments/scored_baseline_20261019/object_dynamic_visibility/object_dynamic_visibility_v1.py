"""NEW dynamic visibility/union-support object construction primitives.

Robust residuals and semismooth derivatives are optimizer devices, never gates.
"""
import numpy as np

CONFIG=dict(construction_halfwidth_m=.02,training_rays_per_clock=24,depth_scale_m=.01,coverage_scale_px=3.,native_scale_m=.001,native_weight=100.,motion_weight=100.,missing_optimization_equivalent_m=.05,max_nfev=40,method_seconds=170.,threads=1,solver='scipy.least_squares.trf.lsmr',default_solver_tolerances=True)

def robust_signed(x):
 x=float(x)
 if x==0:return 0.,1.
 l=np.log1p(x*x);r=np.copysign(np.sqrt(l),x);d=abs(x)/((1+x*x)*np.sqrt(l));return float(r),float(d)

def projection_jacobian(C,K,Rcw):
 C=np.asarray(C,float)
 if not np.isfinite(C).all()or C[2]<=0:raise ValueError('UNRESOLVED nonpositive source projection')
 J=np.array([[K[0,0]/C[2],0.,-K[0,0]*C[0]/C[2]**2],[0.,K[1,1]/C[2],-K[1,1]*C[1]/C[2]**2]])
 return np.einsum('ij,jk->ik',J,Rcw,optimize=False)

def projected_edge_distance(C,edges,pixel,K,Rcw):
 C=np.asarray(C,float);edges=np.asarray(edges,int);p=np.asarray(pixel,float)
 if np.any(C[:,2]<=0):return dict(status='UNRESOLVED_nonpositive_projection')
 uv=np.column_stack([K[0,0]*C[:,0]/C[:,2]+K[0,2],K[1,1]*C[:,1]/C[:,2]+K[1,2]]);a=uv[edges[:,0]];b=uv[edges[:,1]];e=b-a;den=np.einsum('ni,ni->n',e,e,optimize=False);t=np.divide(np.einsum('ni,ni->n',p-a,e,optimize=False),den,out=np.zeros_like(den),where=den>0);t=np.minimum(1,np.maximum(0,t));near=a+t[:,None]*e;diff=near-p;d2=np.einsum('ni,ni->n',diff,diff,optimize=False);i=int(np.argmin(d2));distance=float(np.sqrt(d2[i]));J=(1-t[i])*projection_jacobian(C[edges[i,0]],K,Rcw)+t[i]*projection_jacobian(C[edges[i,1]],K,Rcw);gradient=np.zeros(3)if distance==0 else diff[i]@J/distance
 return dict(status='NEW_closest_projected_edge_distance',distance_px=distance,world_translation_gradient=gradient,edge_index=i,edge_vertices=edges[i],edge_fraction=float(t[i]),closest_pixel=near[i],zero_length_projected_edges=int(np.count_nonzero(den==0)),nonunique_nearest_edge=bool(np.count_nonzero(d2==d2[i])>1),semismooth_choice_NOT_visibility_or_source_accuracy=True)

def maximum_axis_union(gaps,axes,delta):
 values=np.asarray(gaps)+np.einsum('ni,i->n',axes,delta,optimize=False);i=int(np.argmax(values));return dict(gap_m=float(values[i]),active_axis=i,world_gradient=np.asarray(axes)[i],nonunique_active_axis=bool(np.count_nonzero(values==values[i])>1),scope='max over all retained face directions; sufficient geometry family, not measured/native gap or complete SAT')

def positive_norm_residual(value,cap,weight):
 v=np.asarray(value,float);norm=float(np.linalg.norm(v));raw=max(0.,norm-cap);grad=np.zeros_like(v)if raw==0 or norm==0 else weight*v/(norm*cap)
 return weight*raw/cap,grad

def p95_value_gradient(speeds):
 s=np.asarray(speeds,float);n=len(s)
 if n<2 or not np.isfinite(s).all():raise ValueError('finite full-clock speeds')
 order=np.argsort(s,kind='stable');position=(n-1)*.95;lo=int(np.floor(position));hi=int(np.ceil(position));w=position-lo;i,j=int(order[lo]),int(order[hi]);gradient=np.zeros(n);gradient[i]+=1-w;gradient[j]+=w
 return dict(value=float((1-w)*s[i]+w*s[j]),gradient=gradient,ordered_indices=[i,j],weights=[1-w,w],tie_at_quantile=bool(s[i]==s[j]),scope='piecewise sorting derivative; original actual p95 check remains separate')

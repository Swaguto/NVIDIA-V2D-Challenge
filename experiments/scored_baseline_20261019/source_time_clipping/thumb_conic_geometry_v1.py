"""Planar source-cone conics, robot branches and compiled-parent bounds.

These numerical geometric helpers do not infer measured source/contact truth
or certify an active native contact, a motor, a trajectory or a task.
"""
import json
import math
import os
import numpy as np
from scipy.optimize import linprog


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': repr(value)}
    return value


def journal(file, value):
    file.write(json.dumps(plain(value), allow_nan=False, separators=(',', ':'))+'\n')
    file.flush()
    os.fsync(file.fileno())


def conic_from_camera_plane(camera_plane, intrinsics, source_pixel, radius):
    H=np.asarray(camera_plane,dtype=float);K=np.asarray(intrinsics,dtype=float);uv=np.asarray(source_pixel,dtype=float)
    if H.shape!=(3,3) or K.shape!=(3,3) or uv.shape!=(2,) or not np.isfinite(np.r_[H.ravel(),K.ravel(),uv,radius]).all() or radius<=0:
        raise ValueError('Finite plane/intrinsics/image-disk inputs required')
    B=np.array([[K[0,0],0.,K[0,2]-uv[0]],[0.,K[1,1],K[1,2]-uv[1]]])
    return H.T@(B.T@B-radius**2*np.diag([0.,0.,1.]))@H


def conic_ellipse(Q):
    Q=np.asarray(Q,dtype=float)
    if Q.shape!=(3,3) or not np.isfinite(Q).all() or not np.allclose(Q,Q.T,atol=1e-8,rtol=1e-12):
        raise ValueError('Finite symmetric conic required')
    A=Q[:2,:2];eigenvalues,eigenvectors=np.linalg.eigh(A)
    if eigenvalues.min()<=0:
        return dict(kind='nonelliptic_or_degenerate',eigenvalues=eigenvalues)
    center=-np.linalg.solve(A,Q[:2,2]);r2=float(center@A@center-Q[2,2])
    if r2<=0:
        return dict(kind='empty_or_point',center=center,eigenvalues=eigenvalues,radius_squared=r2)
    factor=eigenvectors@np.diag(np.sqrt(r2/eigenvalues))
    return dict(kind='ellipse',center=center,factor=factor,eigenvalues=eigenvalues,radius_squared=r2)


def ellipse_circle_roots(center, factor, circle_center, radius):
    """All numerical tan-half-angle roots of |c+L(cos t,sin t)-d|^2=r^2."""
    z=np.asarray(center)-np.asarray(circle_center);u=np.asarray(factor)[:,0];v=np.asarray(factor)[:,1]
    f0=float(z@z+.5*(u@u+v@v)-radius**2);fc=float(2*z@u);fs=float(2*z@v);fc2=float(.5*(u@u-v@v));fs2=float(u@v)
    poly=np.array([f0+fc+fc2,2*fs+4*fs2,2*f0-6*fc2,2*fs-4*fs2,f0-fc+fc2]);scale=max(float(abs(poly).max()),1e-20);trim=poly.copy()
    while len(trim)>1 and abs(trim[-1])<1e-13*scale:trim=trim[:-1]
    roots=np.polynomial.polynomial.polyroots(trim) if len(trim)>1 else np.array([],complex)
    angles=[]
    for root in roots:
        if abs(root.imag)<1e-8:
            angle=float(2*np.arctan(root.real)%(2*np.pi))
            if not any(abs(np.angle(np.exp(1j*(angle-a))))<1e-8 for a in angles):angles.append(angle)
    # The half-angle chart omits t=pi; retain an independently checked seam.
    seam_point=z-u;seam_res=float(seam_point@seam_point-radius**2)
    if abs(seam_res)<1e-11*max(float(z@z+u@u+radius**2),1e-20) and not any(abs(a-np.pi)<1e-8 for a in angles):angles.append(float(np.pi))
    return dict(angles=sorted(angles),complex_roots=[[float(x.real),float(x.imag)] for x in roots],polynomial_ascending=poly,trig_coefficients=[f0,fc,fs,fc2,fs2],degenerate_constant=bool(len(trim)==1),seam_residual=seam_res,numerical_roots_not_formal_exclusion=True)


def thumb_planar_inverse(point, offset, lengths, proximal_bounds, distal_bounds):
    x,y=np.asarray(point)-np.asarray(offset);l1,l2=lengths;co=(x*x+y*y-l1*l1-l2*l2)/(2*l1*l2)
    if abs(co)>1+1e-12:return []
    result=[]
    for distal in [-math.acos(float(np.clip(co,-1,1))),math.acos(float(np.clip(co,-1,1)))]:
        proximal=math.atan2(y,x)-math.pi/2-math.atan2(l2*math.sin(distal),l1+l2*math.cos(distal))
        for shift in [-2*math.pi,0.,2*math.pi]:
            p=np.array([proximal+shift,distal])
            if proximal_bounds[0]-1e-8<=p[0]<=proximal_bounds[1]+1e-8 and distal_bounds[0]-1e-8<=p[1]<=distal_bounds[1]+1e-8 and not any(abs(p-v).max()<1e-9 for v in result):result.append(p)
    return result


def sinusoid_bounds(cosine,sine,constant,interval):
    a,b,w=np.broadcast_arrays(cosine,sine,constant);lo,hi=interval;left=a*np.cos(lo)+b*np.sin(lo)+w;right=a*np.cos(hi)+b*np.sin(hi)+w;lower=np.minimum(left,right);upper=np.maximum(left,right);phase=np.arctan2(b,a)
    for k in range(math.floor((lo-np.pi)/(2*np.pi))-1,math.ceil((hi+np.pi)/(2*np.pi))+2):
        peak=phase+2*k*np.pi;trough=peak+np.pi;lower=np.where((lo<=trough)&(trough<=hi),w-np.hypot(a,b),lower);upper=np.where((lo<=peak)&(peak<=hi),w+np.hypot(a,b),upper)
    pad=1e-12*(1+abs(a)+abs(b)+abs(w));return lower-pad,upper+pad


def rotate_z(vertices, angle):
    cc,ss=np.cos(angle),np.sin(angle);R=np.array([[cc,-ss,0.],[ss,cc,0.],[0.,0.,1.]])
    return np.asarray(vertices)@R.T


def common_ball(palm_equations,thumb_equations,origin,angle):
    n0=np.asarray(palm_equations)[:,:3];h0=-np.asarray(palm_equations)[:,3];n1=rotate_z(np.asarray(thumb_equations)[:,:3],angle);h1=-np.asarray(thumb_equations)[:,3]+n1@origin
    normals=np.r_[n0,n1];rhs=np.r_[h0,h1];A=np.c_[normals,np.ones(len(normals))];res=linprog([0.,0.,0.,-1.],A_ub=A,b_ub=rhs,bounds=[(None,None)]*4,method='highs')
    out=dict(success=bool(res.success),status=int(res.status),message=res.message,iterations=int(res.nit),returned_point_and_radius=None if res.x is None else res.x,maximum_constraint_residual=None if res.x is None else float((A@res.x-rhs).max()))
    return out


def ball_interval_lower_bound(palm_equations,thumb_equations,origin,point,angle_interval):
    n0=np.asarray(palm_equations)[:,:3];h0=-np.asarray(palm_equations)[:,3];n1=np.asarray(thumb_equations)[:,:3];h1=-np.asarray(thumb_equations)[:,3];v=np.asarray(point)-origin
    aa=n1[:,0]*v[0]+n1[:,1]*v[1];bb=-n1[:,1]*v[0]+n1[:,0]*v[1];cc=n1[:,2]*v[2];_,maximum=sinusoid_bounds(aa,bb,cc,angle_interval);radius=min(float((h0-n0@point).min()),float((h1-maximum).min()))
    return dict(shared_ball_center=point,guaranteed_common_radius_m=radius,penetration_translation_lower_bound_m=max(0.,2*radius),palm_minimum_margin_m=float((h0-n0@point).min()),thumb_plane_maxima=maximum)


def support_interval_gap(palm_vertices,thumb_vertices,origin,direction,angle_interval):
    n=np.asarray(direction);p=np.asarray(palm_vertices);v=np.asarray(thumb_vertices);a=n[0]*v[:,0]+n[1]*v[:,1];bb=-n[0]*v[:,1]+n[1]*v[:,0];w=n[2]*v[:,2]+n@origin;lower,_=sinusoid_bounds(a,bb,w,angle_interval);support=float((p@n).max());return dict(direction=n,thumb_minimum_support_m=float(lower.min()),palm_maximum_support_m=support,gap_lower_bound_m=float(lower.min()-support),all_vertex_minimum_projections_m=lower)

"""NEW sufficient palm-face clearance charts on compiled thumb surfaces.

For a fixed axial thumb angle, every thumb vertex must lie outside one palm
support plane to within the constructor overlap. The proximal-angle domains
are intersections of exact-form sinusoid inequalities. This sufficient subset
can miss valid separating axes and does not certify native active contacts.
"""
import math
import numpy as np

def sinusoid_nonnegative_intervals(coefficients,bounds):
    coef=np.asarray(coefficients,float);lo,hi=map(float,bounds)
    if coef.ndim!=2 or coef.shape[1]!=3 or not np.isfinite(coef).all() or not -math.pi<lo<=hi<math.pi:raise ValueError('finite sinusoid coefficients on a principal interval required')
    cuts=[lo,hi];roots=[]
    for i,(a,b,c) in enumerate(coef):
        amplitude=math.hypot(a,b)
        if amplitude>0 and abs(c)<=amplitude:
            phase=math.atan2(b,a);width=math.acos(float(np.clip(-c/amplitude,-1,1)))
            for root0 in [phase-width,phase+width]:
                for k in [-2,-1,0,1,2]:
                    v=root0+2*k*math.pi
                    if lo<v<hi:cuts.append(v);roots.append(dict(row=i,angle_rad=v))
    cuts=sorted(set(cuts));records=[]
    if lo==hi:cuts=[lo,hi]
    for left,right in zip(cuts,cuts[1:]):
        center=(left+right)/2;values=coef[:,0]*np.cos(center)+coef[:,1]*np.sin(center)+coef[:,2];records.append(dict(interval_rad=[left,right],midpoint_minimum=float(values.min()),inside_sufficient_plane=bool(values.min()>=0)))
    return dict(roots=roots,intervals=records,coefficients=coef)

def face_chart(palm_vertices,thumb_vertices,palm_normals,thumb1_origin,axial_angle,proximal_bounds,constructor_overlap):
    palm=np.asarray(palm_vertices,float);thumb=np.asarray(thumb_vertices,float);normals=np.asarray(palm_normals,float);origin=np.asarray(thumb1_origin,float)
    if palm.ndim!=2 or palm.shape[1]!=3 or thumb.ndim!=2 or thumb.shape[1]!=3 or normals.ndim!=2 or normals.shape[1]!=3 or origin.shape!=(3,) or constructor_overlap<0:raise ValueError('compiled finite surface vertices/normals and explicit nonnegative overlap required')
    if not all(np.isfinite(v).all() for v in [palm,thumb,normals,origin]) or not math.isfinite(axial_angle):raise ValueError('finite geometry required')
    Ry=np.array([[np.cos(axial_angle),0,np.sin(axial_angle)],[0,1,0],[-np.sin(axial_angle),0,np.cos(axial_angle)]]);records=[]
    for i,n in enumerate(normals):
        n=n/np.linalg.norm(n);local=Ry.T@n;support=float(np.max(palm@n));alpha=local[0]*thumb[:,0]+local[1]*thumb[:,1];beta=-local[0]*thumb[:,1]+local[1]*thumb[:,0];gamma=local[2]*thumb[:,2]+local@origin-support+constructor_overlap;domain=sinusoid_nonnegative_intervals(np.c_[alpha,beta,gamma],proximal_bounds)
        records.append(dict(palm_face_index=i,palm_normal=n,normal_in_thumb0_frame=local,palm_support=support,domain=domain))
    return records

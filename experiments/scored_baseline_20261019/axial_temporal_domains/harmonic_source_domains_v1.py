"""Numerical angle domains of a camera-point harmonic in a source disk.

Intervals preserve positive-depth exclusions and numerical boundary ambiguity.
They are not formal directed-rounding/global kinematic proofs.
"""
import math
import numpy as np


def harmonic_image_domain(C0,Cc,Cs,K,pixel,radius,angle_bounds):
    C0,Cc,Cs=[np.asarray(v,dtype=float) for v in [C0,Cc,Cs]];K=np.asarray(K,dtype=float);pixel=np.asarray(pixel,dtype=float);lo,hi=map(float,angle_bounds)
    if any(v.shape!=(3,) for v in [C0,Cc,Cs]) or K.shape!=(3,3) or pixel.shape!=(2,) or not -math.pi<lo<=hi<math.pi or radius<=0 or not np.isfinite(np.r_[C0,Cc,Cs,K.ravel(),pixel,radius,lo,hi]).all():
        raise ValueError('Finite harmonics, positive disk and one principal chart required')
    B=np.array([[K[0,0],0.,K[0,2]-pixel[0]],[0.,K[1,1],K[1,2]-pixel[1]]]);M=B.T@B-radius**2*np.diag([0.,0.,1.]);f0=float(C0@M@C0+.5*(Cc@M@Cc+Cs@M@Cs));fc=float(2*C0@M@Cc);fs=float(2*C0@M@Cs);fc2=float(.5*(Cc@M@Cc-Cs@M@Cs));fs2=float(Cc@M@Cs);polynomial=np.array([f0+fc+fc2,2*fs+4*fs2,2*f0-6*fc2,2*fs-4*fs2,f0-fc+fc2]);scale=max(float(abs(polynomial).max()),1e-20);trim=polynomial.copy()
    while len(trim)>1 and abs(trim[-1])<1e-14*scale:trim=trim[:-1]
    roots=np.polynomial.polynomial.polyroots(trim) if len(trim)>1 else np.array([],complex);breaks=[lo,hi];boundary_records=[]
    for root in roots:
        if abs(root.imag)<1e-8:
            angle=float(2*np.arctan(root.real))
            if lo<angle<hi:breaks.append(angle);boundary_records.append(dict(kind='source_disk_boundary',angle_rad=angle))
    amplitude=math.hypot(Cc[2],Cs[2])
    if amplitude>0 and abs((.01-C0[2])/amplitude)<=1:
        phase=math.atan2(Cs[2],Cc[2]);width=math.acos(float(np.clip((.01-C0[2])/amplitude,-1,1)))
        for base in [phase-width,phase+width]:
            for turn in [-2*math.pi,0.,2*math.pi]:
                angle=base+turn
                if lo<angle<hi:breaks.append(angle);boundary_records.append(dict(kind='positive_depth_boundary',angle_rad=angle))
    breaks=sorted(breaks);unique=[]
    for angle in breaks:
        if not unique or angle-unique[-1]>1e-11:unique.append(float(angle))
    if lo==hi:unique=[lo,hi]
    records=[]
    for left,right in zip(unique,unique[1:]):
        angle=(left+right)/2;C=C0+Cc*np.cos(angle)+Cs*np.sin(angle);value=float(C@M@C);tol=1e-12*scale
        classification='positive_depth_failure' if C[2]<=.01 else ('inside_source_disk' if value < -tol else ('outside_source_disk' if value>tol else 'unresolved_numerical_boundary'))
        records.append(dict(angle_interval_rad=[left,right],midpoint_camera_point=C,midpoint_source_inequality=value,classification=classification))
    return dict(angle_bounds=[lo,hi],camera_harmonics=[C0,Cc,Cs],source_disk_radius_px=radius,source_disk_center_px=pixel,trigonometric_coefficients=[f0,fc,fs,fc2,fs2],tan_half_angle_polynomial=polynomial,all_complex_roots=[[float(x.real),float(x.imag)] for x in roots],boundary_records=boundary_records,intervals=records,numerical_boundary_not_formal_global_proof=True)


def closest_interior_angle(domain,preferred):
    choices=[]
    for record in domain['intervals']:
        if record['classification']!='inside_source_disk':continue
        lo,hi=record['angle_interval_rad'];pad=min(1e-6,.1*(hi-lo));value=float(np.clip(preferred,lo+pad,hi-pad));choices.append((abs(value-preferred),value,record))
    return min(choices,key=lambda v:(v[0],v[1])) if choices else None

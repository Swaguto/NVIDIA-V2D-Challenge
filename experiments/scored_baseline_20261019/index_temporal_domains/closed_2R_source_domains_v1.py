"""Constructive closed-2R source-cone charts, including straight distal joints.

Camera harmonics on all four workspace boundaries avoid assuming that the
pullback conic is elliptic. Absence of sampled boundary/interior witnesses
remains unresolved. This module never claims global continuous exclusion.
"""
import math
import numpy as np
from harmonic_source_domains_v1 import harmonic_image_domain
from thumb_conic_geometry_v1 import conic_from_camera_plane

def closed_2R_source_chart(H,K,pixel,radius,proximal_bounds,distal_bounds,*,lengths=(.0458,.035),image_tolerance=1e-6,position_tolerance=1e-8):
    H=np.asarray(H,float);K=np.asarray(K,float);pixel=np.asarray(pixel,float);a0,a1=map(float,proximal_bounds);b0,b1=map(float,distal_bounds);l1,l2=map(float,lengths)
    if H.shape!=(3,3) or K.shape!=(3,3) or pixel.shape!=(2,) or not np.isfinite(np.r_[H.ravel(),K.ravel(),pixel,radius,a0,a1,b0,b1,l1,l2,image_tolerance,position_tolerance]).all():raise ValueError('finite plane/camera/disk/joint data required')
    if not -math.pi<a0<=a1<math.pi or a1-a0>=math.pi or not 0<=b0<=b1<math.pi or radius<=0 or min(l1,l2)<=0 or min(image_tolerance,position_tolerance)<0:raise ValueError('closed positive-distal chart and positive disk/lengths required')
    Q=conic_from_camera_plane(H,K,pixel,radius);witnesses=[];proposals=[];boundaries=[]
    def offer(a,b,kind,extra=None):
        xy=np.array([l1*np.cos(a)+l2*np.cos(a+b),l1*np.sin(a)+l2*np.sin(a+b)]);C=H@np.r_[xy,1.];depth=bool(C[2]>.01);error=None
        if depth:
            uv=np.array([K[0,0]*C[0]/C[2]+K[0,2],K[1,1]*C[1]/C[2]+K[1,2]]);error=float(np.linalg.norm(uv-pixel))
        valid=bool(a0-position_tolerance<=a<=a1+position_tolerance and b0-position_tolerance<=b<=b1+position_tolerance and depth and error<=radius+image_tolerance)
        row=dict(kind=kind,angles_rad=[a,b],planar_point=xy,camera_point=C,source_error_px=error,positive_depth=depth,passed_original_position_source_gates=valid,extra=extra);proposals.append(row)
        if valid and not any(np.max(abs(np.array(v['angles_rad'])-[a,b]))<1e-10 for v in witnesses):witnesses.append(row)
    for a in [a0,a1]:
        for b in [b0,b1]:offer(a,b,'closed_workspace_corner')
    # Any finite conic stationary point can be a constructive interior point;
    # it is not assumed to be an ellipse center or a global minimum.
    if abs(np.linalg.det(Q[:2,:2]))>1e-15:
        center=-np.linalg.solve(Q[:2,:2],Q[:2,2]);co=float((center@center-l1*l1-l2*l2)/(2*l1*l2))
        if abs(co)<=1+1e-12:
            b=float(np.arccos(np.clip(co,-1,1)));a=float(np.arctan2(center[1],center[0])-np.arctan2(l2*np.sin(b),l1+l2*np.cos(b)))
            for turn in [-2*math.pi,0.,2*math.pi]:offer(a+turn,b,'finite_conic_stationary_point')
    if a0==a1 and b0==b1:
        return dict(classification='constructive_closed_2R_source_witness' if witnesses else 'empty_single_configuration',witnesses=witnesses,proposals=proposals,boundary_charts=[],closed_distal_zero_handled=True,no_continuous_exclusion_claim=True)
    for b in sorted(set([b0,b1])):
        vx=l1+l2*np.cos(b);vy=l2*np.sin(b);boundaries.append(dict(kind='fixed_distal_closed_workspace_boundary',fixed_angle=b,parameter='proximal',domain=harmonic_image_domain(H[:,2],H[:,:2]@np.array([vx,vy]),H[:,:2]@np.array([-vy,vx]),K,pixel,radius,[a0,a1])))
    for a in sorted(set([a0,a1])):
        boundaries.append(dict(kind='fixed_proximal_closed_workspace_boundary',fixed_angle=a,parameter='distal',domain=harmonic_image_domain(H[:,2]+H[:,:2]@np.array([l1*np.cos(a),l1*np.sin(a)]),H[:,:2]@np.array([l2*np.cos(a),l2*np.sin(a)]),H[:,:2]@np.array([-l2*np.sin(a),l2*np.cos(a)]),K,pixel,radius,[b0,b1])))
    for boundary in boundaries:
        for interval in boundary['domain']['intervals']:
            if interval['classification']!='inside_source_disk':continue
            parameter=float(np.mean(interval['angle_interval_rad']));a,b=(parameter,boundary['fixed_angle']) if boundary['parameter']=='proximal' else (boundary['fixed_angle'],parameter);offer(a,b,'continuous_boundary_interval_midpoint',dict(boundary_kind=boundary['kind'],parameter_interval_rad=interval['angle_interval_rad']))
        for record in boundary['domain']['boundary_records']:
            if record['kind']!='source_disk_boundary':continue
            parameter=record['angle_rad'];a,b=(parameter,boundary['fixed_angle']) if boundary['parameter']=='proximal' else (boundary['fixed_angle'],parameter);offer(a,b,'source_boundary_root',dict(boundary_kind=boundary['kind']))
    return dict(classification='constructive_closed_2R_source_witness' if witnesses else 'unresolved_no_verified_boundary_or_stationary_witness',witnesses=witnesses,proposals=proposals,boundary_charts=boundaries,closed_distal_zero_handled=True,conic_eigenvalues=np.linalg.eigvalsh(Q[:2,:2]),no_continuous_exclusion_claim=True,numerical_roots_not_directed_rounding_proof=True)

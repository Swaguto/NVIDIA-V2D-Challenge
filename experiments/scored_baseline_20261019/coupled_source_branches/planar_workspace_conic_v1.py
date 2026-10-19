"""Numerical conic/closed positive-distal 2R workspace intersection.

For a nonsingular positive distal branch and a proximal chart shorter than
pi, the joint rectangle maps injectively to a simply connected curved region.
Corners, ellipse center and boundary intersections detect overlap. Numerical
quartics do not provide a formal exact-arithmetic exclusion certificate.
"""
import math
import numpy as np
from thumb_conic_geometry_v1 import ellipse_circle_roots, thumb_planar_inverse


def positive_2R_conic_intersection(ellipse,proximal_bounds,distal_bounds,lengths=(.0458,.035)):
    a0,a1=map(float,proximal_bounds);b0,b1=map(float,distal_bounds);l1,l2=lengths
    if not np.isfinite([a0,a1,b0,b1,l1,l2]).all() or a1<a0 or b1<b0 or min(l1,l2)<=0:
        raise ValueError('Ordered finite joint domains and positive lengths required')
    if ellipse['kind']!='ellipse' or not 0<b0<=b1<math.pi or a1-a0>=math.pi:
        return dict(classification='unresolved_nonelliptic_or_singular_chart',witnesses=[],root_records=[])
    center=np.asarray(ellipse['center']);factor=np.asarray(ellipse['factor']);inverse=np.linalg.inv(factor);witnesses=[];records=[];uncertain=False
    def point(a,b):return np.array([l1*np.cos(a)+l2*np.cos(a+b),l1*np.sin(a)+l2*np.sin(a+b)])
    def inverse_branches(p):
        return [v+[math.pi/2,0.] for v in thumb_planar_inverse(p,[0.,0.],lengths,[a0-math.pi/2,a1-math.pi/2],[b0,b1])]
    for a in [a0,a1]:
        for b in [b0,b1]:
            p=point(a,b);res=float(np.sum((inverse@(p-center))**2)-1)
            if res<=1e-12:witnesses.append(dict(kind='workspace_corner_inside_ellipse',point=p,angles=[a,b],ellipse_inequality=res))
    for angles in inverse_branches(center):witnesses.append(dict(kind='ellipse_center_in_workspace',point=center,angles=angles))
    circles=[]
    for b in [b0,b1]:circles.append(dict(kind='distal_workspace_boundary',joint_angle=b,center=[0.,0.],radius=float(np.sqrt(l1*l1+l2*l2+2*l1*l2*np.cos(b)))))
    for a in [a0,a1]:circles.append(dict(kind='proximal_workspace_boundary',joint_angle=a,center=[l1*np.cos(a),l1*np.sin(a)],radius=l2))
    for circle in circles:
        roots=ellipse_circle_roots(center,factor,circle['center'],circle['radius']);record=dict(circle=circle,roots=roots,root_point_records=[])
        if roots['degenerate_constant']:uncertain=True
        for real,imag in roots['complex_roots']:
            if 1e-8<=abs(imag)<1e-5:uncertain=True
        for t in roots['angles']:
            p=center+factor@np.array([np.cos(t),np.sin(t)]);branches=inverse_branches(p);res=float(np.sum((p-np.array(circle['center']))**2)-circle['radius']**2);row=dict(ellipse_angle_rad=t,point=p,circle_residual_m2=res,actual_bounded_branches=branches);record['root_point_records'].append(row)
            if abs(res)>1e-10:uncertain=True
            for angles in branches:witnesses.append(dict(kind='conic_workspace_boundary_intersection',point=p,angles=angles,circle_kind=circle['kind'],circle_residual_m2=res))
        records.append(record)
    classification='nonempty_numerical_positive_branch' if witnesses else ('unresolved_numerical_roots' if uncertain else 'empty_numerical_positive_branch')
    return dict(classification=classification,witnesses=witnesses,root_records=records,positive_distal_injective_chart=True,whole_rectangle_not_just_distal_samples=True,not_formal_directed_rounding_or_global_robot_proof=True)

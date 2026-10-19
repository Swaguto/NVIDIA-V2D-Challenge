"""Research-only camera-disk dual and rigid-rotation necessary bounds.

For positive camera depth, ||B C|| <= radius*Cz implies n.C <= 0,
where n = B.T u - radius*ez and ||u|| <= 1.  This 3-D halfspace
remains valid when a robot wrist rotates; a 2-D conic normal alone does not.
All floating-point certificates are numerical, not directed-rounding proofs.
"""
import math
import numpy as np

def disk_matrix(K, pixel):
    K=np.asarray(K,dtype=float); pixel=np.asarray(pixel,dtype=float)
    if K.shape!=(3,3) or pixel.shape!=(2,) or not np.isfinite(K).all() or not np.isfinite(pixel).all():
        raise ValueError('finite calibrated K and two-vector pixel required')
    if not np.array_equal(K[2],[0.,0.,1.]) or K[0,1]!=0 or K[1,0]!=0 or min(K[0,0],K[1,1])<=0:
        raise ValueError('positive focal lengths and zero-skew pinhole calibration required')
    return np.array([[K[0,0],0.,K[0,2]-pixel[0]],[0.,K[1,1],K[1,2]-pixel[1]]])

def tangent_dual(K,pixel,radius,camera_boundary):
    radius=float(radius); C=np.asarray(camera_boundary,dtype=float)
    if not math.isfinite(radius) or radius<=0 or C.shape!=(3,) or not np.isfinite(C).all() or C[2]<=0:
        raise ValueError('positive finite radius and positive-depth boundary required')
    B=disk_matrix(K,pixel); d=B@C; dn=float(np.linalg.norm(d))
    if dn==0 or abs(dn-radius*C[2])>1e-7*max(1.,dn):
        raise ValueError('point does not bind the stated source cone')
    u=d/dn; n=B.T@u-radius*np.array([0.,0.,1.]); length=float(np.linalg.norm(n))
    return dict(B=B,unit_disk_dual=u,camera_normal=n/length,normal_scale=length,
                camera_boundary=C,boundary_residual=float(n@C)/length,
                positive_depth_required=True,halfspace_sense='normal_dot_camera_point_le_zero')

def lift_ellipse_support(K,pixel,radius,H,center,factor,direction):
    H=np.asarray(H,dtype=float); center=np.asarray(center,dtype=float); L=np.asarray(factor,dtype=float); n2=np.asarray(direction,dtype=float)
    if H.shape!=(3,3) or center.shape!=(2,) or L.shape!=(2,2) or n2.shape!=(2,) or not all(np.isfinite(v).all() for v in [H,center,L,n2]):
        raise ValueError('finite nonsingular plane and ellipse support data required')
    norm=float(np.linalg.norm(L.T@n2))
    if norm<=0 or abs(np.linalg.det(H))<1e-15:
        raise ValueError('degenerate plane or support direction')
    x=center+(L@(L.T@n2))/norm; dual=tangent_dual(K,pixel,radius,H@np.r_[x,1.])
    pull=H[:,:2].T@dual['camera_normal']; scale=float(pull@n2/(n2@n2))
    if scale<=0: raise ValueError('source support has reversed sign or depth')
    es=float(n2@center+norm)
    return dict(**dual,planar_support_point=x,planar_direction=n2,
                planar_to_camera_halfspace_scale=scale,planar_ellipse_support=es,
                pullback_direction_residual= float(np.linalg.norm(pull-scale*n2)),
                pullback_constant_residual=float(dual['camera_normal']@H[:,2]+scale*es))

def necessary_rotation_angle(gap,world_normal_length,local_radius,binding_guard):
    """Any wrist rotation smaller than this cannot remove a positive plane gap.

    Translation fixed. For every local point |v| <= local_radius, rotating
    by theta moves Rv by <=2*local_radius*sin(theta/2). This is a necessary
    bound for one halfspace, not a source-cone intersection certificate.
    """
    vals=np.asarray([gap,world_normal_length,local_radius,binding_guard],float)
    if not np.isfinite(vals).all() or world_normal_length<=0 or local_radius<=0 or binding_guard<0:
        raise ValueError('finite positive geometry and nonnegative binding guard required')
    residual=max(0.,float(gap-binding_guard)); denom=2*world_normal_length*local_radius
    if residual>denom:return dict(angle_rad=math.pi,all_rotations_excluded_by_this_plane=True)
    return dict(angle_rad=2*math.asin(residual/denom),all_rotations_excluded_by_this_plane=False)

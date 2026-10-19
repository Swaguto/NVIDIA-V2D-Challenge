"""Exact algebraic form, evaluated numerically, of a two-joint source cone.

For u=tan(a/2), v=tan(b/2), camera numerators have bidegree (2,2).
The disk preimage has bidegree at most (4,4). The common denominator is
strictly positive for finite u,v; positive camera depth remains a separate
inequality. Bernstein ranges are numerical outer enclosures, not a formal
directed-rounding proof. No polynomial point is a measured robot pose/contact.
"""
import math
import numpy as np


def multiply(p,q):
    p=np.asarray(p,float);q=np.asarray(q,float)
    out=np.zeros(np.array(p.shape)+np.array(q.shape)-1)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):out[i:i+q.shape[0],j:j+q.shape[1]]+=p[i,j]*q
    return out


def source_polynomial(H,K,pixel,radius,*,lengths=(.0458,.035),minimum_depth=.01):
    H=np.asarray(H,float);K=np.asarray(K,float);pixel=np.asarray(pixel,float)
    if (H.shape!=(3,3) or K.shape!=(3,3) or pixel.shape!=(2,) or
        not np.isfinite(np.r_[H.ravel(),K.ravel(),pixel,radius,lengths,minimum_depth]).all() or
        len(lengths)!=2 or min(lengths)<=0 or radius<=0 or minimum_depth<=0 or
        K[0,0]<=0 or K[1,1]<=0 or K[0,1]!=0 or K[1,0]!=0 or not np.array_equal(K[2],[0,0,1])):
        raise ValueError('Finite pinhole camera, positive disk/lengths/depth required')
    den=np.array([1.,0.,1.]);co=np.array([1.,0.,-1.]);si=np.array([0.,2.,0.]);l1,l2=lengths
    D=np.outer(den,den)
    X=l1*np.outer(co,den)+l2*(np.outer(co,co)-np.outer(si,si))
    Y=l1*np.outer(si,den)+l2*(np.outer(si,co)+np.outer(co,si))
    C=np.array([H[i,0]*X+H[i,1]*Y+H[i,2]*D for i in range(3)])
    p=K[0,0]*C[0]+(K[0,2]-pixel[0])*C[2]
    q=K[1,1]*C[1]+(K[1,2]-pixel[1])*C[2]
    F=multiply(p,p)+multiply(q,q)-radius*radius*multiply(C[2],C[2])
    depth=C[2]-minimum_depth*D
    scale=max(float(abs(F).max()),1e-300);depthscale=max(float(abs(depth).max()),1e-300)
    return dict(camera_numerators=C,denominator=D,source_polynomial=F,
                normalized_source_polynomial=F/scale,source_scale=scale,
                depth_polynomial=depth,normalized_depth_polynomial=depth/depthscale,depth_scale=depthscale,
                source_bidegree=list(np.array(F.shape)-1),positive_denominator_for_finite_chart=True,
                positive_depth_inequality_separate=True,minimum_depth=minimum_depth,
                b_zero_is_kinematic_singularity_not_polynomial_failure=True,
                direct_original_image_gate_required_for_any_constructive_candidate=True)


def bernstein_range(coefficients,bounds):
    coefficients=np.asarray(coefficients,float);bounds=np.asarray(bounds,float)
    if (coefficients.ndim!=2 or bounds.shape!=(2,2) or not np.isfinite(np.r_[coefficients.ravel(),bounds.ravel()]).all()
        or (bounds[:,0]>bounds[:,1]).any()):raise ValueError('Finite polynomial and ordered rectangular bounds required')
    n,m=np.array(coefficients.shape)-1;lo=bounds[:,0];width=np.diff(bounds,axis=1)[:,0]
    local=np.zeros_like(coefficients)
    for k in range(n+1):
        for l in range(m+1):
            for i in range(k+1):
                for j in range(l+1):
                    local[i,j]+=coefficients[k,l]*math.comb(k,i)*lo[0]**(k-i)*width[0]**i*math.comb(l,j)*lo[1]**(l-j)*width[1]**j
    bernstein=np.zeros_like(coefficients)
    for i in range(n+1):
        for j in range(m+1):
            for k in range(i+1):
                for l in range(j+1):bernstein[i,j]+=local[k,l]*math.comb(i,k)/math.comb(n,k)*math.comb(j,l)/math.comb(m,l)
    pad=2e-12*(1+float(abs(local).sum())+float(abs(bernstein).sum()))
    return dict(lower=float(bernstein.min()-pad),upper=float(bernstein.max()+pad),bernstein_coefficients=bernstein,
                local_power_coefficients=local,numerical_outward_padding=pad,
                input_half_angle_bounds=bounds,formal_directed_rounding_proof=False)


def source_box_chart(polynomial,angle_bounds):
    bounds=np.asarray(angle_bounds,float)
    if bounds.shape!=(2,2) or not np.isfinite(bounds).all() or (bounds[:,0]>bounds[:,1]).any():
        raise ValueError('Finite ordered proximal/distal angle rectangle required')
    if (bounds<=-np.pi).any() or (bounds>=np.pi).any():
        return dict(classification='unresolved_half_angle_seam_or_pole',angle_bounds=bounds)
    half=np.tan(bounds/2);padding=1e-12*(1+abs(half));half[:,0]-=padding[:,0];half[:,1]+=padding[:,1]
    source=bernstein_range(polynomial['normalized_source_polynomial'],half)
    depth=bernstein_range(polynomial['normalized_depth_polynomial'],half)
    classification='unresolved_correlated_source_boundary'
    if source['lower']>0:classification='numerically_empty_original_source_disk_preimage'
    elif depth['upper']<=0:classification='numerically_empty_positive_depth_preimage'
    elif source['upper']<=0 and depth['lower']>0:classification='numerically_inside_original_source_cone'
    return dict(classification=classification,angle_bounds=bounds,half_angle_bounds=half,source_range=source,depth_range=depth,
                distal_zero_boundary_present=bool(bounds[1,0]<=0<=bounds[1,1]),
                no_new_candidate_or_native_contact_inferred=True,numerical_tan_endpoint_enclosure_not_formal=True)

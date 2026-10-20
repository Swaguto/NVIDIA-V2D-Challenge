"""NEW sufficient order-statistic support for a future source construction.

Fixed proxy vertices are algorithmic witnesses, never observed correspondences.
The original source gate and the saved hard-training oracle remain separate.
"""
import numpy as np

def selected_proxy_prefix(distances):
    d=np.asarray(distances,float)
    if d.ndim!=1 or not len(d) or not np.isfinite(d).all() or np.any(d<0):
        raise ValueError('finite nonnegative saved distance vector required')
    lo=(len(d)-1)//2;hi=len(d)//2;order=np.argsort(d,kind='stable')
    ids=order[:hi+1]
    caps=np.full(hi+1,d[order[lo]])
    if hi!=lo:caps[-1]=d[order[hi]]
    return dict(indices=ids,radius_caps=caps,lower_rank=lo,upper_rank=hi,
                lower_radius=d[order[lo]],upper_radius=d[order[hi]],
                lower_count=lo+1,upper_count=hi+1,
                nominal_median=(d[order[lo]]+d[order[hi]])/2,
                proof='At least lo+1 nearest distances <= lower_radius and hi+1 <= upper_radius.',
                proxy_correspondences_are_construction_ONLY=True)

def proposed_support_rows(offsets,B,radii):
    """s=b-A*delta=(radius,offset+B*delta) in SOC4, no solver."""
    c=np.asarray(offsets,float);M=np.asarray(B,float);r=np.asarray(radii,float)
    if c.ndim!=2 or c.shape[1]!=3 or M.shape!=(3,3) or r.shape!=(len(c),):
        raise ValueError('explicit point/same-world-translation map required')
    if not np.isfinite(np.r_[c.ravel(),M.ravel(),r]).all()or np.any(r<0):
        raise ValueError('finite data and nonnegative radius required')
    A=np.zeros((len(c),4,3));A[:,1:,:]=-M
    b=np.column_stack([r,c])
    return A,b

def source_filter_decision(returns,expected_frames):
    """Only actual per-frame hard-oracle returns, never an average loss."""
    indexed={x['frame']:x for x in returns}
    if len(indexed)!=len(returns):return dict(status='UNRESOLVED_duplicate_frame',accepted=False)
    missing=sorted(set(expected_frames)-set(indexed))
    extra=sorted(set(indexed)-set(expected_frames))
    failed=sorted(f for f,x in indexed.items()if not x.get('passes_TRAINING_construction',False))
    accepted=not missing and not extra and not failed
    return dict(status='all_declared_training_frames_pass'if accepted else'REJECTED_or_UNRESOLVED_training_filter',
                accepted=accepted,missing_frames=missing,extra_frames=extra,failed_frames=failed,
                no_global_source_native_or_trajectory_qualification=True)

"""A finite union of CONTINUOUS joint boxes with original temporal limits.

This mixed linear program is not the previous finite-node graph. Source
qualification of every box is the caller's responsibility. The returned
optimizer status is separate from independently verified candidate validity.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def continuous_box_path(banks, original, timestamps, lower, upper, velocity,
                        *, component=.25, adjacent=.05, second=.05,
                        position_tolerance=1e-8, time_limit=60., node_limit=5000):
    original=np.asarray(original,float);ts=np.asarray(timestamps,float)
    if original.ndim!=2 or len(original)<3 or len(banks)!=len(original):
        raise ValueError('One box bank per reference clock is required')
    T,D=original.shape
    lower,upper,velocity=[np.asarray(v,float) for v in [lower,upper,velocity]]
    if (ts.shape!=(T,) or (np.diff(ts)<=0).any() or
        any(v.shape!=(D,) for v in [lower,upper,velocity]) or (lower>upper).any() or (velocity<=0).any() or
        not np.isfinite(np.r_[original.ravel(),ts,lower,upper,velocity,component,adjacent,second,position_tolerance,time_limit,node_limit]).all() or
        min(component,adjacent,second,position_tolerance)<0 or min(time_limit,node_limit)<=0):
        raise ValueError('Explicit finite original reference and actual limits required')
    banks=[np.asarray(v,float) for v in banks]
    if any(v.ndim!=3 or v.shape[1:]!=(D,2) or not np.isfinite(v).all() or (v[:,:,0]>v[:,:,1]).any() for v in banks):
        raise ValueError('Each bank must be finite (K,D,2) closed boxes')
    if any(len(v)==0 for v in banks):
        return dict(status=None,success=False,message='Empty source-qualified inner-box bank',x=None,
                    empty_rows=[i for i,b in enumerate(banks) if not len(b)])
    Q=T*D; offsets=np.r_[2*Q,2*Q+np.cumsum([len(v) for v in banks])];N=int(offsets[-1])
    lb=np.r_[np.maximum(lower-position_tolerance,original-component).ravel(),np.zeros(Q),np.zeros(N-2*Q)]
    ub=np.r_[np.minimum(upper+position_tolerance,original+component).ravel(),np.full(Q,np.inf),np.ones(N-2*Q)]
    objective=np.r_[np.zeros(Q),np.ones(Q),np.zeros(N-2*Q)]
    integrality=np.r_[np.zeros(2*Q,dtype=int),np.ones(N-2*Q,dtype=int)]
    ii=[];jj=[];vv=[];rowlo=[];rowhi=[]
    def row(entries,lo=-np.inf,hi=np.inf):
        k=len(rowlo)
        for j,v in entries:ii.append(k);jj.append(int(j));vv.append(float(v))
        rowlo.append(lo);rowhi.append(hi)
    for t,bank in enumerate(banks):
        z=range(int(offsets[t]),int(offsets[t+1]))
        row([(j,1) for j in z],1.,1.)
        for d in range(D):
            q=t*D+d
            row([(q,1)]+[(j,-v) for j,v in zip(z,bank[:,d,0],strict=True)],0.,np.inf)
            row([(q,1)]+[(j,-v) for j,v in zip(z,bank[:,d,1],strict=True)],-np.inf,0.)
            row([(q,1),(Q+q,-1)],hi=original[t,d])
            row([(q,-1),(Q+q,-1)],hi=-original[t,d])
    for t in range(1,T):
        for d in range(D):
            entries=[(t*D+d,1),((t-1)*D+d,-1)]
            delta=original[t,d]-original[t-1,d]
            row(entries,delta-adjacent,delta+adjacent)
            row(entries,-velocity[d]*(ts[t]-ts[t-1]),velocity[d]*(ts[t]-ts[t-1]))
    for t in range(2,T):
        for d in range(D):
            delta=original[t,d]-2*original[t-1,d]+original[t-2,d]
            row([(t*D+d,1),((t-1)*D+d,-2),((t-2)*D+d,1)],delta-second,delta+second)
    matrix=coo_matrix((vv,(ii,jj)),shape=(len(rowlo),N)).tocsc()
    result=milp(objective,integrality=integrality,bounds=Bounds(lb,ub),
                constraints=LinearConstraint(matrix,np.array(rowlo),np.array(rowhi)),
                options=dict(time_limit=time_limit,node_limit=node_limit,mip_rel_gap=0.))
    return dict(status=int(result.status),success=bool(result.success),message=str(result.message),
                x=None if result.x is None else result.x,objective=None if result.fun is None else float(result.fun),
                mip_gap=getattr(result,'mip_gap',None),mip_node_count=getattr(result,'mip_node_count',None),
                variable_count=N,constraint_count=len(rowlo),box_offsets=offsets,
                proposed_angles=None if result.x is None else result.x[:Q].reshape(T,D),
                finite_inner_family_not_global_exclusion=True)

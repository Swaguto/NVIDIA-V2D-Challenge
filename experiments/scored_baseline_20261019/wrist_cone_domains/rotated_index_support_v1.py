"""NEW batched exact-form numerical linear supports of positive-2R rectangles.

Four circular boundaries suffice: there is no interior stationary point
when the distal angle lies strictly between zero and pi. Witnesses are
numerical; no collision, source truth or floating-point formal proof follows.
"""
import numpy as np

def minimum_support(normals,rotation,base,constants,boxes):
    W=np.asarray(normals,float);R=np.asarray(rotation,float);v0=np.asarray(base,float);offset=np.asarray(constants,float);boxes=np.asarray(boxes,float)
    count=len(W)
    if W.shape!=(count,3) or R.shape!=(3,3) or v0.shape!=(3,) or offset.shape!=(count,) or boxes.shape!=(count,2,2):raise ValueError('bound normals, poses and two-joint rectangles required')
    if not all(np.isfinite(v).all() for v in [W,R,v0,offset,boxes]) or not np.all(boxes[:,:,0]<=boxes[:,:,1]) or not np.all((boxes[:,1,0]>0)&(boxes[:,1,1]<np.pi)) or not np.all(abs(boxes[:,0])<=np.pi):raise ValueError('finite positive-distal joint chart required')
    if not np.allclose(R.T@R,np.eye(3),atol=1e-12,rtol=0) or np.linalg.det(R)<0:raise ValueError('proper common rotation required')
    V=np.einsum('ni,ij->nj',W,R,optimize=False);u=V[:,:2];phi=np.arctan2(u[:,1],u[:,0]);a0,a1=boxes[:,0,0],boxes[:,0,1];b0,b1=boxes[:,1,0],boxes[:,1,1];candidates=[]
    def offer(a,b,ok):
        x=.0458*np.cos(a)+.035*np.cos(a+b);y=.0458*np.sin(a)+.035*np.sin(a+b);value=u[:,0]*x+u[:,1]*y;candidates.append((np.where(ok,value,np.inf),a,b))
    for a in [a0,a1]:
        for b in [b0,b1]:offer(a,b,np.ones(count,bool))
    for b in [b0,b1]:
        shift=np.arctan2(.035*np.sin(b),.0458+.035*np.cos(b))
        for k in [-2,-1,0,1,2]:
            a=phi-shift+np.pi+2*k*np.pi;offer(a,b,(a>=a0)&(a<=a1))
    for a in [a0,a1]:
        for k in [-2,-1,0,1,2]:
            b=phi-a+np.pi+2*k*np.pi;offer(a,b,(b>=b0)&(b<=b1))
    values=np.stack([v[0] for v in candidates]);chosen=np.argmin(values,axis=0);columns=np.arange(count);aval=np.stack([v[1] for v in candidates])[chosen,columns];bval=np.stack([v[2] for v in candidates])[chosen,columns];total=values[chosen,columns]+np.einsum('ni,i->n',V,v0,optimize=False)+offset
    return dict(minimum=total,proximal_witness=aval,distal_witness=bval,selected_boundary_candidate=chosen,root_local_normals=V)

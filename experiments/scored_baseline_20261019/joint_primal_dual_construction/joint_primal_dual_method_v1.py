"""NEW bounded nonlinear whole-joint primal-dual construction primitives.

Fixed two updates, no continuation/line search/best-state selection. Step metrics
are algorithm operators; original matrices/objective/conditioning are unchanged.
All convergence/physical/source-accuracy claims remain outside this finite method.
"""
from joint_constraint_derivatives_v1 import *
from joint_lossless_returns_v1 import LosslessRecorder,seal_return
import time,hashlib

NPAIR=151*1653
NVAR=1523+2*NPAIR
NOBJ=4551571
NROB=8114
NSOCSRC=194*3
NDEPTH=194
NDOM=76
NDYN=NSOCSRC+NDEPTH+NDOM+NPAIR
NROW=NOBJ+NROB+NDYN


def sphere_chart(ab):
    """One continuous stereographic chart; absent pole is NOT excluded physically.

    Body-A local witness coordinates only, never a wrist axis/profile or a bank.
    No normalization/clipping of a supplied direction or source/command occurs.
    """
    a,b=np.asarray(ab);den=1+a*a+b*b
    numerator=np.array([2*a,2*b,1-a*a-b*b])
    n=numerator/den
    da=(np.array([2.,0.,-2*a])*den-numerator*(2*a))/(den*den)
    db=(np.array([0.,2.,-2*b])*den-numerator*(2*b))/(den*den)
    return n,np.column_stack([da,db])


def positive_sum_outer(matrix):
    """Conditional finite-RN upper L1 envelopes, not candidate tolerances.

    Positive additions are exact in the subnormal lattice until normal. For at
    most 2^32 finite nonnegative terms, gamma_n<1/2 in binary64. Twice the saved
    positive sum therefore encloses the exact saved-coefficient sum absent
    overflow/non-RN behavior. Unknown/nonfinite results stop the method.
    """
    if matrix.nnz>=2**32:raise ValueError('declared sum-count guard')
    absolute=matrix.copy();absolute.data=np.abs(absolute.data)
    rows=2*np.asarray(absolute.sum(axis=1)).ravel()
    cols=2*np.asarray(absolute.sum(axis=0)).ravel()
    return rows,cols


def steps(row_bounds,col_bounds,cones):
    rb=np.asarray(row_bounds).copy();cb=np.asarray(col_bounds)
    for start,width in cones:rb[start:start+width]=rb[start:start+width].max(initial=0.)
    sigma=np.ones_like(rb);tau=np.ones_like(cb)
    np.divide(.5,rb,out=sigma,where=rb>0)
    np.divide(.5,cb,out=tau,where=cb>0)
    return sigma,tau


def product_projection(value,cones):
    """Algorithm projection ONLY; preserve its raw input/output, never repair."""
    value=np.asarray(value);out=np.maximum(value,0.).copy();branches=[]
    for start,width in cones:
        z=value[start:start+width];t=z[0];v=z[1:];norm=float(np.linalg.norm(v))
        if norm<=t:out[start:start+width]=z;branch='inside'
        elif norm<=-t:out[start:start+width]=0.;branch='polar'
        else:
            head=.5*(norm+t);out[start]=head;out[start+1:start+width]=(head/norm)*v;branch='middle'
        branches.append(dict(start=start,width=width,norm=norm,branch=branch))
    return out,branches


def primal_prox(raw,tau,Pdiag):
    returned=np.asarray(raw)/(1+np.asarray(tau)*np.asarray(Pdiag))
    # Exact prox of the ALREADY ORIGINAL normalized phase interval [0,1].
    # The two original phase rows also remain present. This is never a repair
    # of a returned terminal candidate, and no robot/object command is clipped.
    phase_before=float(returned[1522]);returned[1522]=min(1.,max(0.,phase_before))
    return returned,dict(original_phase_prox_input=phase_before,original_phase_prox_output=float(returned[1522]))


def layout_cones(nonnegative):
    cones=[];start=nonnegative
    for width in [4]*900+[5]*151:
        cones.append((start,width));start+=width
    if start!=NOBJ:raise ValueError('all object cones required')
    for j in range(194):cones.append((NOBJ+NROB+3*j,3))
    return cones


def dynamic_maps(api,x,chart,record,tag,deadline):
    """ONE complete new oracle, per-frame durable returns, no selected row bank."""
    sources=api.source_and_domains(x);record(tag+'-ALL-source-RETURN',sources)
    f=np.zeros(NDYN);ri=[];ci=[];data=[];envelope=[]
    def offer(row,col,value,bound=None):
        ri.append(int(row));ci.append(int(col));data.append(float(value));envelope.append(float(abs(value) if bound is None else bound))
    for item in sources['ALL194source']:
        rec=api.source[item['factor']];out=item['returned'];Cpoint=out['camera_point'];CJ=out['camera_jacobian']
        K=np.asarray(rec['K']);pixel=rec['source_pixel'];radius=out['constructed_radius_px']
        B=np.array([[K[0,0],0.,K[0,2]-pixel[0]],[0.,K[1,1],K[1,2]-pixel[1]]])
        L=np.vstack([np.array([0.,0.,radius]),B])
        rows=3*item['factor'];f[rows:rows+3]=-L@Cpoint
        J=-L@CJ
        for k,col in enumerate(item['columns']):
            for component in range(3):offer(rows+component,col,J[component,k])
        row=NSOCSRC+item['factor'];f[row]=-out['strict_depth_value']
        for k,col in enumerate(item['columns']):offer(row,col,-out['strict_depth_jacobian'][k])
    for j,item in enumerate(sources['ALL38half_domains']):
        if item.get('status','').startswith('UNKNOWN'):raise ValueError('unresolved chart pole')
        row=NSOCSRC+NDEPTH+2*j;f[row]=-item['lower_residual'];f[row+1]=-item['upper_residual']
        offer(row,item['column'],-item['derivative']);offer(row+1,item['column'],item['derivative'])
    gstart=NSOCSRC+NDEPTH+NDOM
    for frame in range(151):
        if time.monotonic()>deadline:raise TimeoutError('declared method150sec cap')
        body=api.geometry_frame(x,frame);record(f'{tag}-frame-{frame:03d}-body-RETURN',body)
        pair_returns=[]
        for pair,(a,b) in enumerate(api.incidence['geom_pair']):
            a,b=int(a),int(b);global_pair=frame*1653+pair;row=gstart+global_pair
            n,DN=sphere_chart(chart[frame,pair]);ba=api.raw['geom_bodyid'][a]
            out=full_support_return(body['geometry'][a],body['geometry'][b],n,body['rotations'][ba],body['rotation_J'][a],x[1522],api.r)
            pair_returns.append(dict(pair=pair,chart=chart[frame,pair],n=n,DN=DN,
               conditional_filter=api.flags[pair]['conditional_builtin_filter_reasons'],ALL_support_RETURN=out))
            if out['status']!='RETURNED':raise ValueError('unknown support direction')
            # Conditional native-filter cases remain typed constant rows, with
            # their complete diagnostic support returns retained above.
            if api.flags[pair]['conditional_builtin_filter_reasons']:continue
            f[row]=-out['residual']
            for key,slopes in out['ALL_tie_factor_derivatives'].items():
                c=int(key);value=-out['SEARCH_command_residual_J'][key]
                bound=float(np.max(np.abs(slopes['A']),initial=0)+np.max(np.abs(slopes['B']),initial=0))
                offer(row,c,value,bound)
            dirs=out['ALL_tie_direction_derivatives']
            ad=dirs['A']@DN;bd=dirs['B']@DN
            for k in range(2):
                offer(row,1523+2*global_pair+k,float(ad[:,k].mean()+bd[:,k].mean()),
                    float(np.max(np.abs(ad[:,k]),initial=0)+np.max(np.abs(bd[:,k]),initial=0)))
            offer(row,1522,-api.r,api.r)
        record(f'{tag}-frame-{frame:03d}-ALL-pair-RETURN',pair_returns)
    J=sparse.coo_matrix((data,(ri,ci)),shape=(NDYN,NVAR)).tocsc()
    E=sparse.coo_matrix((envelope,(ri,ci)),shape=(NDYN,NVAR)).tocsc()
    record(tag+'-ALL-dynamic-F-J-envelope-RETURN',dict(F=f,J=J,absolute_tie_envelope=E))
    return f,J,E


def method_oracle(api,w,record,tag,deadline):
    record(tag+'-NEW-input-BEFORE',dict(w=w,coordinate_count=NVAR,NEW_method=True))
    x=w[:1523];chart=w[1523:].reshape(151,1653,2)
    obj=api.A@x[465:]-api.constants['object_b'];rob=api.robot@x[:465]-api.constants['robot_b']
    record(tag+'-ALL-object-robot-F-RETURN',dict(object_F=obj,robot_F=rob))
    dyn,J,E=dynamic_maps(api,x,chart,record,tag,deadline)
    Fvalue=np.r_[obj,rob,dyn]
    record(tag+'-ALL-F-RETURN',dict(F=Fvalue))
    return Fvalue,J,E


def adjoint(api,J,y):
    gradient=np.zeros(NVAR)
    gradient[465:1523]=api.A.T@y[:NOBJ]
    gradient[:465]=api.robot.T@y[NOBJ:NOBJ+NROB]
    gradient+=J.T@y[NOBJ+NROB:]
    return gradient


def finite_or_unknown(*arrays):
    return all(np.isfinite(a.data if sparse.issparse(a) else a).all() for a in arrays)

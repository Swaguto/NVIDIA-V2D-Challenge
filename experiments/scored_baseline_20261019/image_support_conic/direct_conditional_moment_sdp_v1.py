"""Research-only direct conditional moment SDP packing and finite-box certificates.

Ax+s=b; zero/equality, nonnegative/inequality, and PSD triangle cones.
Stored float packing has explicit outward matrix guards on 0<=weights<=1.
No solver vector, Gram factor or local moment table is a robot trajectory.
"""
from fractions import Fraction as F
from math import isqrt,sqrt
import math
import conditional_moment_cones_v1 as c

def layout(n):return [(i,j)for j in range(n)for i in range(j+1)]
def root_two_interval(bits=100):
    k=isqrt(2*(1<<(2*bits)));return F(k,1<<bits),F(k+1,1<<bits)
def up(x):
    z=float(x)
    return math.nextafter(z,math.inf)if F(z)<x else z
def svec(A):return [float(A[i][j])*(1 if i==j else sqrt(2))for i,j in layout(len(A))]
def smat(v,n):
    if len(v)!=n*(n+1)//2:raise ValueError('invalid packed PSD size')
    A=[[F(0)for _ in range(n)]for _ in range(n)]
    for z,(i,j)in zip(v,layout(n),strict=True):A[i][j]=A[j][i]=F(float(z)/(1 if i==j else sqrt(2)))
    return A
def exact_maps(group,kind):
    n=3 if kind=='Hankel'else 2
    maps=[[[F(0)for _ in range(5)]for _ in range(n)]for _ in range(n)]
    for k in range(5):
        w=[F(int(t==k))for t in range(5)];A=c.matrices(c.moments(w))[kind]
        for i in range(n):
            for j in range(n):maps[i][j][k]=A[i][j]
    return maps
def packed_outer(group,kind):
    maps=exact_maps(group,kind);n=len(maps);lo,hi=root_two_interval();packed=[];errors=[[F(0)for _ in range(n)]for _ in range(n)]
    for i,j in layout(n):
        values=[];err=F(0)
        for value in maps[i][j]:
            z=float(value)*(1 if i==j else sqrt(2));values.append(z)
            if i==j:e=abs(F(z)-value)
            else:e=max(abs(F(z)/lo-value),abs(F(z)/hi-value))
            err+=e
        packed.append(values);errors[i][j]=errors[j][i]=err
    radius=max(map(sum,errors));guard=up(radius)
    return dict(group=group,cone=kind,dimension=n,layout=layout(n),exact_entry_maps=[[[c.rec(v)for v in row]for row in A]for A in maps],packed_float_maps=packed,unpacked_entry_error_bounds=[[c.rec(v)for v in row]for row in errors],spectral_error_upper=c.rec(radius),outward_diagonal_guard=guard,sqrt_two_interval=[c.rec(lo),c.rec(hi)])
def outer_linear(row):
    values=[(v['column'],float(c.dec(v['value'])))for v in row['coefficients']]
    error=sum(abs(F(z)-c.dec(v['value']))for (_,z),v in zip(values,row['coefficients'],strict=True));b=c.dec(row['bound']);rhs=up(b+error)
    return values,rhs,dict(name=row['name'],coefficient_absolute_error=c.rec(error),original_bound=c.rec(b),packed_bound=rhs,necessary_outward_rhs=c.rec(F(rhs)-b))
def assemble(model,emit):
    import numpy as np
    from scipy import sparse
    n=len(model['columns']);assert model['variable_bounds']==[0,1]
    rr=[];cc=[];vv=[];b=[];blocks=[];linear=[];row=0
    for r in model['equalities']:
        for v in r['coefficients']:
            x=c.dec(v['value']);z=float(x)
            if F(z)!=x:raise ValueError('nonbinary equality requires a separately derived outer representation')
            rr.append(row);cc.append(v['column']);vv.append(z)
        x=c.dec(r['bound']);z=float(x)
        if F(z)!=x:raise ValueError('nonbinary equality RHS not silently rounded')
        b.append(z);row+=1
    eq=row
    for r in model['inequalities']:
        values,rhs,record=outer_linear(r)
        for col,z in values:rr.append(row);cc.append(col);vv.append(z)
        b.append(rhs);linear.append(record);row+=1
    for sign,rhs in [(-1.,0.),(1.,1.)]:
        for col in range(n):rr.append(row);cc.append(col);vv.append(sign);b.append(rhs);row+=1
    nonnegative=row-eq
    emit(dict(kind='AFTER_NEW_linear_box_equality_packing_RETURN',equalities=eq,inequalities=len(linear),nonnegative_rows=nonnegative,linear_outward_guards=linear))
    for group in c.groups(model['columns']):
        for kind in ['Hankel','localizer']:
            block=packed_outer(group,kind);block['row_start']=row
            emit(dict(kind='BEFORE_NEW_PSD_block_binding',group=group,cone=kind))
            for (i,j),values in zip(block['layout'],block['packed_float_maps'],strict=True):
                for col,z in zip(group['columns'],values,strict=True):
                    if z:rr.append(row);cc.append(col);vv.append(-z)
                b.append(block['outward_diagonal_guard']if i==j else 0.);row+=1
            block['row_end']=row;blocks.append(block)
            emit(dict(kind='AFTER_NEW_PSD_block_binding_RETURN',block=block))
    A=sparse.csc_matrix((vv,(rr,cc)),shape=(row,n));A.sort_indices()
    return A,np.array(b),dict(equalities=eq,original_inequalities=len(linear),nonnegative_rows=nonnegative,linear_rounding=linear,PSD_blocks=blocks,columns=n,rows=row,unit_box_exact=True,scope='Necessary direct conditional moment model only, no common measure or deterministic path claim')
def solver_return(solution):
    out={}
    for name in dir(solution):
        if not name.startswith('_'):
            value=getattr(solution,name)
            if not callable(value):out[name]=str(value)if name=='status'else value
    return out
def rational_gram(A,emit):
    """A NEW certificate matrix, preserving the raw returned dual separately.

    Rational Gram products are PSD by construction. This is certificate
    construction, never an alteration/qualification of primal table weights.
    """
    import numpy as np
    emit(dict(kind='BEFORE_NEW_dual_Gram_factor',raw_rational_matrix=[[c.rec(v)for v in row]for row in A]))
    vals,vecs=np.linalg.eigh(np.array(A,dtype=float))
    emit(dict(kind='AFTER_NEW_dual_eigendecomposition_RETURN',values=vals,vectors=vecs))
    B=[[F(float(vecs[i,k]*sqrt(float(vals[k]))))if vals[k]>0 else F(0)for k in range(len(A))]for i in range(len(A))]
    G=[[sum(B[i][k]*B[j][k]for k in range(len(A)))for j in range(len(A))]for i in range(len(A))]
    return G,B
def certificate(model,packing,z,emit):
    """Residual-corrected exact finite-box separation for the original model.

    alpha>=0, beta free, Z=B B^T: r=A^T alpha+E^T beta-sum M*Z.
    Any feasible table in [0,1] satisfies r.x<=alpha.b+beta.d.
    Hence sum min(0,r_i)>RHS excludes this necessary frozen model.
    No solver feasibility flag alone constitutes an exclusion certificate.
    """
    eq=packing['equalities'];ni=packing['original_inequalities'];n=len(model['columns']);r=[F(0)]*n;rhs=F(0)
    beta=list(map(F,z[:eq]));alpha=list(map(F,z[eq:eq+ni]))
    bad=[i for i,v in enumerate(alpha)if v<0]
    emit(dict(kind='AFTER_NEW_exact_dual_scalar_sign_RETURN',negative_original_inequality_duals=bad,raw_duals_not_changed=True))
    if bad:return dict(classification='UNRESOLVED_negative_dual_multipliers',indices=bad)
    for rows,weights in [(model['equalities'],beta),(model['inequalities'],alpha)]:
        for row,w in zip(rows,weights,strict=True):
            rhs+=w*c.dec(row['bound'])
            for v in row['coefficients']:r[v['column']]+=w*c.dec(v['value'])
    matrices=[]
    for block in packing['PSD_blocks']:
        start,end=block['row_start'],block['row_end'];raw=smat(z[start:end],block['dimension']);G,B=rational_gram(raw,emit);maps=exact_maps(block['group'],block['cone']);columns=block['group']['columns'];d=len(G)
        for k,col in enumerate(columns):r[col]-=sum(G[i][j]*maps[i][j][k]for i in range(d)for j in range(d))
        record=dict(group_id=block['group']['id'],cone=block['cone'],raw_dual_matrix=[[c.rec(v)for v in row]for row in raw],Gram_factor=[[c.rec(v)for v in row]for row in B],certificate_matrix=[[c.rec(v)for v in row]for row in G])
        emit(dict(kind='AFTER_NEW_exact_dual_Gram_map_RETURN',record=record));matrices.append(record)
    minimum=sum(min(F(0),v)for v in r);gap=minimum-rhs
    return dict(classification='EXACT_conditional_frozen_moment_model_exclusion'if gap>0 else'UNRESOLVED_no_strict_residual_corrected_conic_gap',alpha=[c.rec(v)for v in alpha],beta=[c.rec(v)for v in beta],matrices=matrices,residual=[c.rec(v)for v in r],finite_box_minimum=c.rec(minimum),RHS=c.rec(rhs),strict_gap=c.rec(gap),box_duals_not_required_finite_box_residual_bounded_exactly=True,certificate_applies_to_original_saved_rational_necessary_model_only=True)

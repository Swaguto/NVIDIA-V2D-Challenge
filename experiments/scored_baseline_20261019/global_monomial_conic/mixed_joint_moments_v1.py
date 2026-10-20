"""Exact mixed-variable square necessities in EXISTING degree-four tensor tables.

No degree increase, source construction, solver, or primal-weight alteration.
Moment matrices use monomials of total degree <=2; interval localizers use
t_i*(1-t_i) and monomials of total degree <=1. All needed powers already fit
the saved per-coordinate degree-four tensor representation.
"""
from fractions import Fraction as F
from itertools import product
from math import comb
import conditional_moment_cones_v1 as c

def basis(dim,degree):
    if type(dim)is not int or dim not in [2,3]or degree not in [1,2,4]:raise ValueError('declared mixed scope is dimension2/3, total degrees1/2/4 only')
    return sorted((a for a in product(range(degree+1),repeat=dim)if sum(a)<=degree),key=lambda a:(sum(a),a))
def tables(columns):
    scopes={}
    for col,v in enumerate(columns):
        scope=tuple(v['scope']);index=tuple(v['index'])
        if len(scope)!=len(index)or len(set(scope))!=len(scope)or any(type(k)is not int or not 0<=k<=4 for k in index):raise ValueError('malformed degree4 table')
        if index in scopes.setdefault(scope,{}):raise ValueError('duplicate tensor entry')
        scopes[scope][index]=col
    for scope,entries in scopes.items():
        if set(entries)!=set(product(range(5),repeat=len(scope))):raise ValueError('incomplete tensor table')
    return scopes
def monomial_coefficients(entries,power):
    if not entries or len(power)!=len(next(iter(entries)))or any(type(r)is not int or not 0<=r<=4 for r in power):raise ValueError('power outside existing degree4 tensor authority')
    result={}
    for index,col in entries.items():
        value=F(1)
        for k,r in zip(index,power,strict=True):value*=F(comb(k,r),comb(4,r))if k>=r else F(0)
        if value:result[col]=value
    return result
def polynomial_coefficients(entries,poly):
    out={}
    for power,value in poly.items():
        for col,a in monomial_coefficients(entries,power).items():out[col]=out.get(col,F(0))+F(value)*a
    return {col:a for col,a in out.items()if a}
def squared_polynomial(exponents,vector,axis=None):
    if len(exponents)!=len(vector)or not exponents:raise ValueError('basis/vector shape')
    dim=len(exponents[0]);p={}
    for a,v in zip(exponents,vector,strict=True):
        for b,w in zip(exponents,vector,strict=True):
            power=tuple(i+k for i,k in zip(a,b,strict=True));p[power]=p.get(power,F(0))+F(v)*F(w)
    if axis is not None:
        if type(axis)is not int or not 0<=axis<dim:raise ValueError('localizer axis')
        q={}
        for a,v in p.items():
            for k,sign in [(1,1),(2,-1)]:
                b=list(a);b[axis]+=k;b=tuple(b);q[b]=q.get(b,F(0))+sign*v
        p=q
    if any(any(v>4 for v in a)for a in p):raise ValueError('square requires unavailable tensor powers')
    return {a:v for a,v in p.items()if v}
def matrix_from_moments(exponents,moments,axis=None):
    M=[]
    for a in exponents:
        row=[]
        for b in exponents:
            p=tuple(i+k for i,k in zip(a,b,strict=True))
            if axis is None:value=moments[p]
            else:
                q=list(p);q[axis]+=1;r=list(p);r[axis]+=2;value=moments[tuple(q)]-moments[tuple(r)]
            row.append(value)
        M.append(row)
    return M
def quadratic(M,v):return sum(F(a)*M[i][k]*F(b)for i,a in enumerate(v)for k,b in enumerate(v))
def congruence_certificate(M,emit):
    """Exact elimination with a rational square witness or full congruence.

    Handles zero pivots with a nonzero row explicitly. No floating eigenvalues
    or numerical positivity tolerances select a result.
    """
    n=len(M)
    if not n or any(len(row)!=n for row in M)or any(M[i][k]!=M[k][i]for i in range(n)for k in range(n)):raise ValueError('symmetric finite square matrix required')
    S=[list(map(F,row))for row in M];columns=[[F(int(i==k))for i in range(n)]for k in range(n)];done=[];diagonal=[];steps=[]
    for stage in range(n):
        pivot=S[0][0];emit(dict(kind='BEFORE_NEW_exact_mixed_congruence_step',stage=stage,pivot=c.rec(pivot),Schur=[[c.rec(v)for v in row]for row in S],remaining_basis=[[c.rec(v)for v in col]for col in columns]))
        if pivot<0:
            v=columns[0];q=quadratic(M,v);out=dict(classification='exact_negative_mixed_square',reason='negative_Schur_pivot',vector=list(map(c.rec,v)),expectation=c.rec(q),stage=stage);emit(dict(kind='AFTER_NEW_exact_mixed_congruence_RETURN',result=out));return out
        if pivot==0:
            k=next((k for k in range(1,len(S))if S[0][k]),None)
            if k is not None:
                t=-(S[k][k]+1)/(2*S[0][k]);v=[t*a+b for a,b in zip(columns[0],columns[k],strict=True)];q=quadratic(M,v);out=dict(classification='exact_negative_mixed_square',reason='zero_pivot_nonzero_cross_term',vector=list(map(c.rec,v)),expectation=c.rec(q),stage=stage);emit(dict(kind='AFTER_NEW_exact_mixed_congruence_RETURN',result=out));return out
            done.append(columns[0]);diagonal.append(F(0));columns=columns[1:];S=[row[1:]for row in S[1:]]
        else:
            old=columns;done.append(old[0]);diagonal.append(pivot);factors=[S[0][k]/pivot for k in range(1,len(S))];columns=[[a-f*b for a,b in zip(col,old[0],strict=True)]for col,f in zip(old[1:],factors,strict=True)];S=[[S[i][k]-S[i][0]*S[0][k]/pivot for k in range(1,len(S))]for i in range(1,len(S))]
        step=dict(stage=stage,pivot=c.rec(pivot),remaining_dimension=len(S));steps.append(step);emit(dict(kind='AFTER_NEW_exact_mixed_Schur_RETURN',step=step,Schur=[[c.rec(v)for v in row]for row in S]))
    out=dict(classification='exact_mixed_matrix_PSD',basis_columns=[[c.rec(v)for v in col]for col in done],diagonal=list(map(c.rec,diagonal)),rank=sum(v>0 for v in diagonal),all_basis_changes_unit_triangular=True)
    emit(dict(kind='AFTER_NEW_exact_mixed_congruence_RETURN',result=out));return out
def square_witness(entries,exponents,axis,vector,weights):
    scale=max(map(abs,vector),default=F(0))
    if not scale:raise ValueError('zero witness')
    v=[F(a)/scale for a in vector];p=squared_polynomial(exponents,v,axis);co=polynomial_coefficients(entries,p);value=sum(a*F(weights[col])for col,a in co.items());norm=max(map(abs,co.values()),default=F(0));distance=-value/norm if norm and value<0 else F(0)
    return dict(basis=[list(a)for a in exponents],axis=axis,normalized_polynomial_vector=list(map(c.rec,v)),normalization_applies_to_certificate_NOT_actual_weights=True,power_coefficients=[dict(power=list(a),value=c.rec(b))for a,b in sorted(p.items())],Bernstein_expectation_coefficients=[dict(column=k,value=c.rec(a))for k,a in sorted(co.items())],actual_exact_expectation=c.rec(value),coefficient_linf=c.rec(norm),necessary_l1_weight_change_to_remove_THIS_witness=c.rec(distance),classification='exact_negative_mixed_polynomial_square'if value<0 else'UNRESOLVED_nonnegative_vector_expectation',scope='Rejects this actual local table only; no source/time family exclusion, command, physical force or trajectory claim')

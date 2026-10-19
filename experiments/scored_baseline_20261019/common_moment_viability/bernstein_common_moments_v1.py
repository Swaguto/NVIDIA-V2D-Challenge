"""Research necessary local-marginal model. Feasible tables are NOT poses.

At normalized deterministic t_j in[0,1], each table is the tensor product of
degree-D Bernstein weights. Shared subset marginals coincide. Nonnegative
polynomials times the remaining nonnegative weights have nonnegative means.
These are necessary restrictions on a common measure, not sufficient global
realizability, deterministic source fit or physical/contact feasibility.
"""
from fractions import Fraction as F
from itertools import product,combinations
from math import comb,prod

def rec(x):
    x=F(x);return dict(numerator_hex=hex(x.numerator),denominator_hex=hex(x.denominator),approx=float(x))
def dec(x):return F(int(x['numerator_hex'],16),int(x['denominator_hex'],16))
def indices(n,D):return product(range(D+1),repeat=n)
def deterministic_weights(t,D=4):
    if not all(0<=x<=1 for x in t):raise ValueError('normalized points must be in[0,1]')
    return {k:prod(F(comb(D,i))*v**i*(1-v)**(D-i)for v,i in zip(t,k,strict=True))for k in indices(len(t),D)}
def elevate(grid,degree,D=4):
    if any(type(n)is not int or n<0 or n>D for n in degree):raise ValueError('invalid elevation degree')
    return {j:sum(v*prod(F(comb(n,i)*comb(D-n,k-i),comb(D,k))for n,i,k in zip(degree,e,j,strict=True))for e,v in grid.items()if all(0<=k-i<=D-n for n,i,k in zip(degree,e,j,strict=True)))for j in indices(len(degree),D)}
def multiply(p,q):
    r={}
    for i,a in p.items():
        for j,b in q.items():r[i+j]=r.get(i+j,F(0))+a*b
    return r
def positive_tests(D=4):
    anchors=[F(i,4)for i in range(5)];out=[]
    for a in anchors:out.append((f'square_{a}',multiply({0:-a,1:F(1)},{0:-a,1:F(1)})))
    for a,b in zip(anchors[:-1],anchors[1:],strict=True):
        p=multiply({0:-a,1:F(1)},{0:-b,1:F(1)});out.append((f'double_square_{a}_{b}',multiply(p,p)))
    return [dict(name=n,power=p,Bernstein={k:sum(v*F(comb(k,i),comb(D,i))for i,v in p.items()if i<=k)for k in range(D+1)})for n,p in out]
def build(variables,factors,D=4):
    if D!=4 or len(set(variables))!=len(variables)or not variables:raise ValueError('unique variables and declared degree4 required')
    order={v:k for k,v in enumerate(variables)};cliques=sorted({tuple(sorted(f['variables'],key=order.__getitem__))for f in factors},key=lambda s:(len(s),tuple(order[v]for v in s)))
    if any(not 1<=len(s)<=3 or len(set(s))!=len(s)for s in cliques):raise ValueError('one/two/three-variable cliques required')
    columns=[];maps={}
    for scope in[(v,)for v in variables]+[s for s in cliques if len(s)>1]:
        mapping={}
        for k in indices(len(scope),D):mapping[k]=len(columns);columns.append(dict(scope=list(scope),index=list(k)))
        maps[scope]=mapping
    eq=[];ineq=[]
    def row(target,name,co,bound=F(0),meta=None):
        co={k:F(v)for k,v in co.items()if v};scale=max(map(abs,co.values()),default=F(1));target.append(dict(name=name,coefficients=[dict(column=k,value=rec(v/scale))for k,v in sorted(co.items())],bound=rec(F(bound)/scale),meta=meta))
    for scope,mapping in maps.items():row(eq,'normalize:'+','.join(scope),dict.fromkeys(mapping.values(),F(1)),F(1))
    for scope,mapping in maps.items():
        if len(scope)==1:continue
        # Every existing proper subset is shared, including pair marginals in triples.
        for size in range(1,len(scope)):
            for subset in combinations(scope,size):
                if subset not in maps:continue
                axes=[scope.index(v)for v in subset]
                for small,col in maps[subset].items():
                    co={c:F(1)for k,c in mapping.items()if tuple(k[a]for a in axes)==small};co[col]=co.get(col,F(0))-1
                    row(eq,'marginal:'+','.join(scope)+'->'+','.join(subset)+':'+str(small),co)
    for factor in factors:
        names=factor['variables'];scope=tuple(sorted(names,key=order.__getitem__));permutation=[names.index(v)for v in scope];mapping=maps[scope];co={mapping[tuple(k[a]for a in permutation)]:v for k,v in factor['grid'].items()}
        if set(co)!=set(mapping.values()):raise ValueError('factor must have a complete degree4 grid')
        row(ineq,'factor:'+factor['id'],co,meta=dict(kind='necessary_source_or_time_lower_grid',factor=factor['id']))
    positives=positive_tests(D)
    for scope,mapping in maps.items():
        for axis,name in enumerate(scope):
            others=[a for a in range(len(scope))if a!=axis]
            for other in indices(len(others),D):
                for p in positives:
                    # -E[q(t_axis)*prod(B_other)]<=0 for q>=0.
                    co={c:-p['Bernstein'][k[axis]]for k,c in mapping.items()if tuple(k[a]for a in others)==other}
                    row(ineq,'positive:'+','.join(scope)+':'+str(axis)+':'+str(other)+':'+p['name'],co,meta=dict(kind='conditioned_square_nonnegative',axis=axis,scope=list(scope),other=list(other),test=p['name']))
    return dict(schema='v2d.bernstein-common-marginal-relaxation.v1',degree=D,variables=variables,columns=columns,equalities=eq,inequalities=ineq,
                variable_bounds=[0,1],deterministic_embedding_necessary_only=True,relaxed_feasible_tables_NOT_poses_or_source_truth=True)
def solve_phase_one(model,time_limit=120):
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix,hstack
    n=len(model['columns'])
    def matrix(rows):
        ri=[];ci=[];values=[]
        for i,row in enumerate(rows):
            for v in row['coefficients']:ri.append(i);ci.append(v['column']);values.append(float(dec(v['value'])))
        return coo_matrix((values,(ri,ci)),shape=(len(rows),n)).tocsr()
    A=matrix(model['inequalities']);E=matrix(model['equalities']);B=np.array([float(dec(v['bound']))for v in model['inequalities']]);d=np.array([float(dec(v['bound']))for v in model['equalities']]);objective=np.zeros(n+1);objective[-1]=1
    result=linprog(objective,A_ub=hstack([A,-np.ones((A.shape[0],1))]).tocsr(),b_ub=B,A_eq=hstack([E,np.zeros((E.shape[0],1))]).tocsr(),b_eq=d,bounds=[(0,1)]*n+[(0,None)],method='highs',options=dict(time_limit=time_limit))
    def values(key):
        value=result.get(key)
        return None if value is None else {k:(None if v is None else np.asarray(v).tolist())for k,v in value.items()}
    return dict(status=int(result.status),success=bool(result.success),message=str(result.message),objective=None if result.fun is None else float(result.fun),
                primal=None if result.x is None else result.x.tolist(),ineqlin=values('ineqlin'),eqlin=values('eqlin'),lower=values('lower'),upper=values('upper'),iterations=int(result.nit))
def finite_box_dual(model,result):
    if not result['success']or result.get('ineqlin')is None:return dict(classification='UNRESOLVED_no_solver_dual')
    alpha=[max(F(0),-F(v))for v in result['ineqlin']['marginals']];beta=[-F(v)for v in result['eqlin']['marginals']];r=[F(0)]*len(model['columns']);rhs=F(0)
    for rows,multipliers in[(model['inequalities'],alpha),(model['equalities'],beta)]:
        for row,m in zip(rows,multipliers,strict=True):
            rhs+=m*dec(row['bound'])
            for v in row['coefficients']:r[v['column']]+=m*dec(v['value'])
    bound=sum(min(F(0),v)for v in r);gap=bound-rhs
    return dict(classification='exact_saved_LP_model_INFEASIBILITY_certificate'if gap>0 else'UNRESOLVED_no_positive_dual_margin',
                alpha=[rec(v)for v in alpha],beta=[rec(v)for v in beta],residual=[rec(v)for v in r],finite_box_lower_bound=rec(bound),combined_rhs=rec(rhs),strict_gap=rec(gap),
                scope='Exact rational check of saved binary/coefficient model on0<=moments<=1; no physical/global source-truth conclusion')

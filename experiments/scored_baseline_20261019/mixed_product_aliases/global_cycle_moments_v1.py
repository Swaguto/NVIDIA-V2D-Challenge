"""NEW partial-global moment/cycle necessities in unchanged tensor coordinates.

All intervals are diagnostic envelopes of SAVED copy/residual values. They are
not candidate tolerances or bounds on all possible corrections/trajectories.
Absent mixed moments remain free unknowns. No distribution/pose is fabricated.
"""
from fractions import Fraction as F
from itertools import product,combinations,permutations
from math import comb,isqrt
import conditional_moment_cones_v1 as c

def recI(v):return [c.rec(z)for z in v]
def addI(a,b):return [a[0]+b[0],a[1]+b[1]]
def negI(a):return [-a[1],-a[0]]
def subI(a,b):return addI(a,negI(b))
def mulI(a,b):
 v=[x*y for x in a for y in b];return [min(v),max(v)]
def squareI(a):return [F(0)if a[0]<=0<=a[1]else min(z*z for z in a),max(z*z for z in a)]
def divposI(a,b):
 if b[0]<=0:raise ValueError('UNRESOLVED_nonpositive_or_zero_pivot_domain')
 return mulI(a,[1/b[1],1/b[0]])
def sqrt96(value):
 v=F(value)
 if v<0:raise ValueError('negative square root domain')
 scale=1<<96;k=isqrt((v.numerator*scale*scale)//v.denominator);lo=F(k,scale);hi=lo if lo*lo==v else F(k+1,scale)
 return [lo,hi]
def triangle_interval(A,B,C,P,Q):
 """Schur necessary interval for missing covariance R between endpoints.

 Uses [[A,P,R],[P,B,Q],[R,Q,C]] PSD. Strict B>0 implies
 (R-PQ/B)^2 <= (A-P^2/B)(C-Q^2/B). Interval outer bounds remain necessary.
 """
 rec=dict(A=recI(A),B=recI(B),C=recI(C),P=recI(P),Q=recI(Q))
 if min(A[1],B[1],C[1])<0:return dict(rec,classification='EXCLUDED_negative_variance_upper')
 if B==[F(0),F(0)]:
  if any(v[0]>0 or v[1]<0 for v in [P,Q]):return dict(rec,classification='EXCLUDED_zero_pivot_nonzero_cross')
  if P!=[F(0),F(0)]or Q!=[F(0),F(0)]:return dict(rec,classification='UNRESOLVED_zero_pivot_cross_intervals')
  rad=mulI([F(0),max(F(0),A[1])],[F(0),max(F(0),C[1])])[1];root=sqrt96(rad);return dict(rec,classification='NECESSARY_missing_covariance_interval',zero_pivot_case=True,radicand=c.rec(rad),sqrt_bounds=recI(root),interval=recI([-root[1],root[1]]))
 if B[0]<=0:return dict(rec,classification='UNRESOLVED_pivot_interval_meets_zero')
 center=divposI(mulI(P,Q),B);left=subI(A,divposI(squareI(P),B));right=subI(C,divposI(squareI(Q),B))
 if min(left[1],right[1])<0:return dict(rec,classification='EXCLUDED_negative_Schur_residual_upper',center=recI(center),left=recI(left),right=recI(right))
 rad=max(F(0),left[1])*max(F(0),right[1]);root=sqrt96(rad);return dict(rec,classification='NECESSARY_missing_covariance_interval',zero_pivot_case=False,center=recI(center),left=recI(left),right=recI(right),radicand=c.rec(rad),sqrt_bounds=recI(root),interval=recI([center[0]-root[1],center[1]+root[1]]),max_zero_on_Schur_upper_is_PSD_necessity_NOT_weight_clipping=True)
def cycle4(variances,edges):
 first=triangle_interval(variances[0],variances[1],variances[2],edges[0],edges[1]);second=triangle_interval(variances[0],variances[3],variances[2],edges[3],edges[2]);out=dict(first_triangle=first,second_triangle=second,missing_diagonal=[0,2],other_missing_diagonal_FREE_UNKNOWN=[1,3],four_variable_completion_is_NOT_global_supported_measure=True)
 if any(z['classification'].startswith('EXCLUDED')for z in [first,second]):return dict(out,classification='EXCLUDED_cycle_covariance_envelope')
 if any(z['classification'].startswith('UNRESOLVED')for z in [first,second]):return dict(out,classification='UNRESOLVED_singular_cycle_domain')
 a=list(map(c.dec,first['interval']));b=list(map(c.dec,second['interval']));intersection=[max(a[0],b[0]),min(a[1],b[1])];gap=intersection[0]-intersection[1]
 return dict(out,classification='EXCLUDED_cycle_covariance_envelope'if gap>0 else'NECESSARY_cycle_bounds_NOT_EXCLUDED',intersection=recI(intersection),strict_separation=c.rec(max(F(0),gap)),passing_NOT_a_completed_covariance_or_pose=True)
def covariance_envelope(mass,first,second,cross):
 if mass[1]<0:return dict(classification='UNRESOLVED_signed_mass_NOT_positive_measure')
 if mass==[F(0),F(0)]:
  zero=all(v==[F(0),F(0)]for v in [*first,*second,*cross]);return dict(classification='ZERO_positive_measure_ONLY'if zero else'EXCLUDED_zero_positive_mass_nonzero_moments')
 if mass[0]<=0:return dict(classification='UNRESOLVED_mass_interval_meets_zero')
 V=[subI(mulI(mass,s),squareI(f))for f,s in zip(first,second,strict=True)];E=[subI(mulI(mass,cross[i]),mulI(first[i],first[(i+1)%4]))for i in range(4)]
 return dict(classification='positive_unnormalized_covariance_necessities',variance_intervals=[recI(v)for v in V],edge_intervals=[recI(v)for v in E],mass_NOT_divided_out=True)
def tables(columns):
 scopes={}
 for col,v in enumerate(columns):scopes.setdefault(tuple(v['scope']),{})[tuple(v['index'])]=col
 for scope,t in scopes.items():
  if set(t)!=set(product(range(5),repeat=len(scope))):raise ValueError('existing degree4 tensor incomplete')
 return scopes
def mono_map(scope,entries,requested):
 out={}
 for idx,col in entries.items():
  v=F(1)
  for name,k in zip(scope,idx,strict=True):
   power=requested.get(name,0);v*=F(comb(k,power),comb(4,power))if k>=power else F(0)
  if v:out[col]=v
 return out
def sparse_add(a,b,z=F(1)):
 out=dict(a)
 for k,v in b.items():out[k]=out.get(k,F(0))+z*v
 return {k:v for k,v in out.items()if v}
def sparse_key(co):return tuple(sorted((k,F(v))for k,v in co.items()if v))
def marginal_links(model,scopes):
 lookup={sparse_key({z['column']:c.dec(z['value'])for z in row['coefficients']}):i for i,row in enumerate(model['equalities'])if c.dec(row['bound'])==0};links={}
 for big,small in permutations(sorted(scopes),2):
  if not set(small)<set(big):continue
  records=[]
  for idx,col in scopes[small].items():
   co={col:F(-1)}
   for bigidx,bcol in scopes[big].items():
    if tuple(bigidx[big.index(v)]for v in small)==idx:co[bcol]=F(1)
   key=sparse_key(co)
   if key not in lookup:break
   records.append(dict(index=idx,equality_row=lookup[key]))
  else:links[big,small]=records
 return links
def projection_path(links,start,target):
 if start==target:return []
 queue=[(start,[])];seen={start}
 for scope,path in queue:
  for big,small in sorted(links):
   if big!=scope or small in seen:continue
   route=path+[(big,small)]
   if small==target:return route
   seen.add(small);queue.append((small,route))
 raise ValueError('UNRESOLVED_no_saved_marginal_path')
def path_combination(links,path,requested):
 out={}
 for big,small in path:
  for row in links[big,small]:
   value=F(1)
   for name,k in zip(small,row['index'],strict=True):
    power=requested.get(name,0);value*=F(comb(k,power),comb(4,power))if k>=power else F(0)
   if value:out[row['equality_row']]=out.get(row['equality_row'],F(0))+value
 return out
def scope_structure(scopes,variables):
 allscopes=sorted(scopes);maximal=[s for s in allscopes if not any(set(s)<set(t)for t in allscopes)];edges=[]
 for i,j in combinations(range(len(maximal)),2):edges.append(dict(i=i,j=j,intersection=sorted(set(maximal[i])&set(maximal[j])),weight=len(set(maximal[i])&set(maximal[j]))))
 parent=list(range(len(maximal)))
 def root(i):
  while i!=parent[i]:i=parent[i]
  return i
 tree=[]
 for edge in sorted(edges,key=lambda e:(-e['weight'],e['i'],e['j'])):
  a,b=root(edge['i']),root(edge['j'])
  if a!=b:parent[b]=a;tree.append(edge)
 required=sum(sum(v in s for s in maximal)-1 for v in variables);actual=sum(e['weight']for e in tree);known={tuple(sorted(pair))for s in allscopes for pair in combinations(s,2)};cycles=set()
 for nodes in combinations(sorted(variables),4):
  for cycle in permutations(nodes):
   if cycle[0]!=min(cycle)or cycle[1]>cycle[-1]:continue
   sides=[tuple(sorted((cycle[i],cycle[(i+1)%4])))for i in range(4)];diags=[tuple(sorted((cycle[0],cycle[2]))),tuple(sorted((cycle[1],cycle[3])))]
   if all(x in known for x in sides)and all(x not in known for x in diags):cycles.add(cycle)
 return dict(maximal_scopes=[list(s)for s in maximal],all_scope_intersections=edges,maximum_weight_tree=tree,maximum_intersection_weight=actual,running_intersection_required_weight=required,running_intersection_deficit=required-actual,standard_junction_tree_exists=actual==required,not_having_a_junction_tree_does_NOT_exclude_common_measure=True,known_variable_pairs=[list(k)for k in sorted(known)],unknown_variable_pairs=[list(k)for k in combinations(sorted(variables),2)if k not in known],induced_four_variable_cycles=[list(k)for k in sorted(cycles)],longer_cycles_and_global_completion_UNASSESSED=True)

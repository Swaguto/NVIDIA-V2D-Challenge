"""NEW matrix-valued camera-cone support within EXISTING degree4 tensors.

SAVED scalar and camera coefficients remain unchanged. An explicit uniform
coefficient discrepancy and positive-depth bound imply a necessary diagonal
arrow guard; this is model arithmetic, never a candidate acceptance tolerance.
"""
from fractions import Fraction as F
from itertools import product
import conditional_moment_cones_v1 as c
import mixed_joint_moments_v1 as m
import pointwise_support_moments_v1 as p

def array_poly(array):
 return {(i,j):F(v)for i,row in enumerate(array)for j,v in enumerate(row)if F(v)}
def scaled(poly,z):return {k:F(z)*v for k,v in poly.items()if z*v}
def uniform_bound(poly,bounds):
 return sum(abs(v)*product_bound(k,bounds)for k,v in poly.items())
def product_bound(power,bounds):
 v=F(1)
 for r,(lo,hi)in zip(power,bounds,strict=True):v*=max(abs(lo),abs(hi))**r
 return v
def degrees(poly,dim=2):return [max((k[i]for k,v in poly.items()if v),default=0)for i in range(dim)]
def arrow(polys):
 d,x,y=polys;zero={};return [[d,x,y],[x,d,zero],[y,zero,d]]
def point_matrix(polys,point):return [[p.evaluate(a,point)for a in row]for row in arrow(polys)]
def construct(source,bounds,emit):
 if len(bounds)!=2 or any(len(v)!=2 or v[0]>v[1]for v in bounds):raise ValueError('ordered finite half-angle bounds')
 sp=source['polynomial'];N=list(map(array_poly,sp['camera_numerators']));D=array_poly(sp['denominator']);expected={(0,0):F(1),(0,2):F(1),(2,0):F(1),(2,2):F(1)}
 if D!=expected or not sp['positive_denominator_for_finite_chart']or not sp['positive_depth_inequality_separate']:raise ValueError('UNRESOLVED_finite_positive_denominator_binding')
 radius=F(source['ORIGINAL_cap_px'])+F(source['ORIGINAL_image_tolerance_px']);depth=F(sp['minimum_depth']);ss=F(sp['source_scale']);ds=F(sp['depth_scale']);K=source['K'];pixel=source['pixel']
 if min(radius,depth,ss,ds)<=0:raise ValueError('UNRESOLVED_nonpositive_radius_depth_or_scale')
 t=scaled(N[2],radius);x=p.add(scaled(N[0],F(K[0][0])),scaled(N[2],F(K[0][2])-F(pixel[0])));y=p.add(scaled(N[1],F(K[1][1])),scaled(N[2],F(K[1][2])-F(pixel[1])))
 emit(dict(kind='AFTER_NEW_unrounded_arrow_numerators_RETURN',t=p.polyrec(t),x=p.polyrec(x),y=p.polyrec(y),radius=c.rec(radius),saved_scalar_models_NOT_regenerated=True))
 # This is a NEW determinant/discrepancy identity, not a replacement source polynomial.
 determinant_sign=p.add(p.add(p.multiply(x,x),p.multiply(y,y)),p.multiply(t,t),F(-1));saved_source=scaled(array_poly(sp['normalized_source_polynomial']),ss);error_poly=p.add(determinant_sign,saved_source,F(-1));epsilon=uniform_bound(error_poly,bounds)
 emit(dict(kind='AFTER_NEW_arrow_to_SAVED_scalar_discrepancy_RETURN',difference=p.polyrec(error_poly),uniform_absolute_error=c.rec(epsilon),source_scale=c.rec(ss),implication='saved_normalized_source<=0 implies norm(x,y)^2-t^2<=epsilon'))
 depth_identity=p.add(N[2],scaled(D,depth),F(-1));saved_depth=scaled(array_poly(sp['normalized_depth_polynomial']),ds);depth_error_poly=p.add(depth_identity,saved_depth,F(-1));eta=uniform_bound(depth_error_poly,bounds);nz_lower=depth-eta
 emit(dict(kind='AFTER_NEW_arrow_to_SAVED_depth_discrepancy_RETURN',difference=p.polyrec(depth_error_poly),uniform_absolute_error=c.rec(eta),positive_denominator_global_lower=1,Nz_necessary_lower=c.rec(nz_lower),depth_scale=c.rec(ds)))
 if nz_lower<=0:raise ValueError('UNRESOLVED_no_positive_depth_lower_bound')
 tmin=radius*nz_lower;delta=epsilon/(2*tmin);diagonal=p.add(t,{(0,0):delta});slack=2*tmin*delta+delta*delta-epsilon
 if slack<0:raise ArithmeticError('necessary arrow guard sign')
 rec=dict(uniform_source_discrepancy=c.rec(epsilon),uniform_depth_discrepancy=c.rec(eta),Nz_lower=c.rec(nz_lower),t_lower=c.rec(tmin),diagonal_guard=c.rec(delta),guard_squared_slack=c.rec(slack),original_radius=c.rec(radius),minimum_depth_COPIED=c.rec(depth),denominator_exact_product=True,denominator_positive_lower=1,source_scalar_sense='<=0',depth_scalar_sense='>=0',necessary_implication='saved_source<=0 and saved_depth>=0 imply guarded arrow PSD on declared box',source_accuracy_or_upstream_trig_proof=False,guard_NOT_original_image_depth_or_motion_tolerance=True,source_discrepancy_power=p.polyrec(error_poly),depth_discrepancy_power=p.polyrec(depth_error_poly),structural_degrees=[degrees(z)for z in [diagonal,x,y]])
 emit(dict(kind='AFTER_NEW_necessary_arrow_guard_RETURN',record=rec));return [diagonal,x,y],rec

def localizer(entries,polys,basis,weights,emit):
 if any(any(r>2 for r in k)for poly in polys for k,v in poly.items()if v):raise ValueError('UNRESOLVED_matrix_entry_degree_exceeds_multilinear_capacity')
 dim=len(basis);cache={}
 for family,poly in enumerate(polys):
  for shift in product(range(3),repeat=2):
   moved={tuple(a+b for a,b in zip(k,shift)):v for k,v in poly.items()};co=m.polynomial_coefficients(entries,moved);value=sum(v*F(weights[col])for col,v in co.items());rec=dict(family=family,shift=list(shift),polynomial=p.polyrec(moved),coefficients=[dict(column=col,value=c.rec(v))for col,v in sorted(co.items())],actual_value=c.rec(value));emit(dict(kind='AFTER_NEW_matrix_support_shift_map_RETURN',record=rec));cache[(family,shift)]=rec
 M=[];entry_records=[]
 for a,i in product(range(3),range(dim)):
  row=[]
  for b,j in product(range(3),range(dim)):
   shift=tuple(x+y for x,y in zip(basis[i],basis[j]));family=0 if a==b else 1 if {a,b}=={0,1}else 2 if {a,b}=={0,2}else None
   value=c.dec(cache[family,shift]['actual_value'])if family is not None else F(0);row.append(value);entry_records.append(dict(row=a*dim+i,column=b*dim+j,family=family,shift=list(shift),actual_entry=c.rec(value)))
  M.append(row)
 emit(dict(kind='AFTER_NEW_matrix_valued_support_RETURN',matrix=[[c.rec(v)for v in row]for row in M],dimension=3*dim,component_major_basis=True));return M,list(cache.values()),entry_records

def weighted_square(polys,basis,vector):
 if len(vector)!=3*len(basis):raise ValueError('vector dimension')
 q=[{tuple(power):F(v)for power,v in zip(basis,vector[i*len(basis):(i+1)*len(basis)],strict=True)if v}for i in range(3)];A=arrow(polys);out={}
 for i,j in product(range(3),repeat=2):out=p.add(out,p.multiply(A[i][j],p.multiply(q[i],q[j])))
 if any(any(r>4 for r in k)for k,v in out.items()if v):raise ValueError('UNRESOLVED_missing_tensor_power')
 return out

def square_witness(entries,polys,basis,vector,weights):
 norm=max(map(abs,vector));v=[F(z)/norm for z in vector];weighted=weighted_square(polys,basis,v);co=m.polynomial_coefficients(entries,weighted);value=sum(z*F(weights[col])for col,z in co.items())
 return dict(component_major_vector=list(map(c.rec,v)),multilinear_basis=[list(k)for k in basis],weighted_polynomial=p.polyrec(weighted),linear_coefficients=[dict(column=col,value=c.rec(z))for col,z in sorted(co.items())],actual_expectation=c.rec(value),normalization_is_certificate_ONLY_actual_weights_unchanged=True,no_family_or_task_exclusion=True)

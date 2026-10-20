"""NEW quotient/leaf-kernel/cache-alias/API-scope controls; no episode calls."""
from pathlib import Path
from fractions import Fraction as Q
import ctypes,datetime,hashlib,json,math,os
import numpy as np
from scipy.spatial import cKDTree
from source_numerical_enclosures_v1 import (interval,projected_operator_enclosure,
 norm_leaf_enclosure,common_dyadic,cached_camera_alias,
 cache_error_affine_coefficients,typed_query_return,U,ETA)
from exact_training_source_logic_v1 import quantized_interval  # unchanged evidence BY HASH

S=Path(__file__).resolve().parents[1];O=S/'e037-source-numerical-contracts-v1'
def pack(v):
 if isinstance(v,Q):return dict(numerator=str(v.numerator),denominator=str(v.denominator))
 if isinstance(v,np.ndarray):return pack(v.tolist())
 if isinstance(v,np.generic):return pack(v.item())
 if isinstance(v,dict):return {str(k):pack(x)for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [pack(x)for x in v]
 if isinstance(v,float)and not math.isfinite(v):return dict(nonfinite_float=repr(v))
 return v
def save(n,v):
 p=O/n;p.write_text(json.dumps(pack(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):log.write(json.dumps(pack(v),allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
 controls=[]
 def control(name,before,fn,test):
  ref=save(name+'-BEFORE.json',before);emit(dict(kind='BEFORE_NEW_contract',name=name,constructor=ref))
  r=fn();ref=save(name+'-RETURN.json',r);emit(dict(kind='AFTER_NEW_actual_RETURN',name=name,actual_RETURN=ref))
  assert test(r),name;controls.append(name)
 save('ONE-NEW-numerical-contract-scope-BEFORE.json',dict(
  conditional_RN_finite_operator_bounds_NOT_actual_source_or_global_search_proof=True,
  synthetic_KD_points_only_NOT_episode_NN_or_old_BVH_controls=True,
  original_fullpixel_query_k1_AND_training_query_k2_scope_distinct=True,
  cached_camera_aliases_are_exact_numeric_defects_NOT_data_repairs=True,
  all_prior_radii_arrays_gate_tolerances_and_failed_returns_UNALTERED=True,
  no_solver_or_nonzero_episode_point_selected=True))
 lib=ctypes.CDLL(None);lib.fegetround.restype=ctypes.c_int;roundmode=int(lib.fegetround())
 save('NEW-read-only-rounding-observation.json',dict(fegetround=roundmode,mode_NOT_mutated=True));assert roundmode==0
 # NEW multiply/divide/add graph, with signed projected coordinates and supplied
 # component uncertainty. This is not the old source/FK/dot-order audit.
 x=np.array([-.125,.25,1.25]);err=[Q(1,2**45),Q(1,2**46),Q(1,2**44)]
 def quotient():
  enclosure=projected_operator_enclosure([(Q(float(v))-e,Q(float(v))+e)for v,e in zip(x,err,strict=True)],701.25,699.75,640.5,400.25)
  actual=[float(701.25*x[0]/x[2]+640.5),float(699.75*x[1]/x[2]+400.25)]
  return dict(enclosure=enclosure,actual_numeric_projection=actual)
 control('NEW-signed-projection-quotient-operator-enclosure',dict(camera=x,error=err,intrinsics=[701.25,699.75,640.5,400.25]),quotient,
  lambda r:all(lo<=Q(a)<=hi for (lo,hi),a in zip(r['enclosure']['uv_intervals'],r['actual_numeric_projection'],strict=True)))
 def boundary():
  r=projected_operator_enclosure([interval(1),interval(0),interval(1)],1.,1.,0.,0.)
  q=quantized_interval(*r['uv_intervals'][0])
  return dict(enclosure=r,quantizer=q,actual_numeric_projection=1.)
 control('NEW-rounded-projection-keeps-crossed-quantizer-unknown',dict(ideal_uv=1,original_quantizer='clip then truncate'),boundary,
  lambda r:r['quantizer']['status']=='UNRESOLVED_integer_boundary')
 def near():
  cases=[(Q(1,100)-Q(1,2**70),Q(1,100)+Q(1,2**70)),(Q(1,100),Q(1,100)),(Q(3,200),Q(3,200))]
  return [projected_operator_enclosure([interval(1),interval(1),c],100.,100.,0.,0.)for c in cases]
 control('NEW-denominator-conditioning-and-STRICT-depth-unknown',dict(near=Q(1,100),no_changed_gate=True),near,
  lambda r:r[0]['status'].startswith('UNRESOLVED')and r[1]['status'].startswith('UNRESOLVED')and r[2]['status'].startswith('CONDITIONAL'))
 tiny=float(np.nextafter(0.,1.))
 def zero_norms():
  out=[]
  for x in [[0.,0,0],[tiny,0,0]]:
   tree=cKDTree(np.array([[0.,0,0]]));d,i=tree.query(np.array(x),workers=1)
   out.append(dict(x=x,raw_distance=float(d),raw_index=int(i),enclosure=norm_leaf_enclosure(x,[0.,0,0])))
  return out
 control('NEW-leaf-norm-zero-and-underflow-retain-real-numeric-distinction',dict(tiny=tiny,synthetic_only=True),zero_norms,
  lambda r:all(x['enclosure']['distance'][0]<=Q(x['raw_distance'])<=x['enclosure']['distance'][1]for x in r)and r[1]['enclosure']['exact_squared_distance']>0 and r[1]['raw_distance']==0.)
 V=np.array([[0.,0,0],[1.,0,0],[.5,.25,0],[.5,-.25,0]])
 queries=np.array([[.125,.03125,0],[.5,0,0]])
 def query_scopes():
  tree=cKDTree(V);out=[]
  for k,scope in [(1,'ORIGINAL_fullpixel_k1'),(2,'EVEN_training_k2')]:
   d,ids=tree.query(queries,k=k,workers=1)
   for p,x in enumerate(queries):
    d0=float(d[p]if k==1 else d[p,0]);v=int(ids[p]if k==1 else ids[p,0]);bounds=norm_leaf_enclosure(x,V[v])
    out.append(dict(scope=scope,k=k,point=p,raw_all_distances=np.atleast_1d(d[p]),raw_all_indices=np.atleast_1d(ids[p]),
     selected_norm_enclosure=bounds,first_distance=d0,first_index=v,
     typed=typed_query_return(d0,v,len(V),k,scope),
     synthetic_ideal_min_squared=min(sum((Q(float(a))-Q(float(b)))**2 for a,b in zip(x,vv,strict=True))for vv in V)))
  return out
 control('NEW-k1-k2-raw-query-scopes-and-leaf-only-certificate',dict(vertices=V,queries=queries,k=[1,2],no_episode=True),query_scopes,
  lambda r:all(x['selected_norm_enclosure']['distance'][0]<=Q(x['first_distance'])<=x['selected_norm_enclosure']['distance'][1]and not x['selected_norm_enclosure']['global_nearest_search_certified']for x in r))
 def missing():
  empty=cKDTree(np.empty((0,3)));d,i=empty.query([0.,0,0],workers=1)
  return dict(raw_empty_distance=float(d),raw_empty_index=int(i),
   typed=typed_query_return(float(d),int(i),0,1,'ORIGINAL_fullpixel_k1'),
   overflow=norm_leaf_enclosure([1e200,0,0],[0,0,0]),
   wrong_k=typed_query_return(.1,0,4,2,'ORIGINAL_fullpixel_k1'))
 control('NEW-missing-overflow-and-query-scope-not-repaired',dict(synthetic_empty_tree=True,synthetic_overflow=1e200),missing,
  lambda r:all(r[k]['status'].startswith('UNRESOLVED')for k in ['typed','overflow','wrong_k'])and math.isinf(r['raw_empty_distance']))
 VV=np.array([[.125,-.25,.5],[-.0625,.375,.25],[.5,.25,-.125]])
 RR=np.array([[1.,.1,0],[-.2,1.,.125],[.0625,0,1.]]);tt=np.array([.3,-.1,1.25]);C0=VV@RR.T+tt
 def dyadic():
  ints,e=common_dyadic(VV);ret=cached_camera_alias(ints,e,RR,tt,C0)
  exact=[[Q(float(C0[v,k]))-sum(Q(float(VV[v,j]))*Q(float(RR[k,j]))for j in range(3))-Q(float(tt[k]))for k in range(3)]for v in range(len(VV))]
  got=[[Q(int(n),1<<ret['denominator_exponent'])for n in row]for row in ret['defect_numerators']]
  return dict(exact_fraction_defects=exact,dyadic_defects=got,
   shared_denominator_exponent=ret['denominator_exponent'],component_maxima=ret['maximum_absolute_defects'],
   raw_C0_unchanged=np.array_equal(C0,VV@RR.T+tt))
 control('NEW-exact-cache-constant-alias-dyadic-versus-Fraction',dict(V=VV,R=RR,t=tt,C0=C0),dyadic,
  lambda r:r['exact_fraction_defects']==r['dyadic_defects']and r['raw_C0_unchanged']and max(r['component_maxima'])>0)
 W=np.array([[1.,.125,0],[0,1.,-.0625],[.125,0,1.]])
 def cache_bound():
  vint,ev=common_dyadic(VV);alias=cached_camera_alias(vint,ev,RR,tt,C0)
  coeff=cache_error_affine_coefficients([Q(float(x))for x in np.max(abs(C0),axis=0)],alias['maximum_absolute_defects'],W)
  H=[Q(1,64),Q(1,128),Q(1,256)];delta=np.array([float(H[0]),-float(H[1]),float(H[2])]);actual=C0+np.einsum('ij,j->i',W,delta,optimize=False)
  errors=[]
  for p in range(len(VV)):
   errors.append([Q(float(actual[p,i]))-sum(Q(float(VV[p,j]))*Q(float(RR[i,j]))for j in range(3))-Q(float(tt[i]))-sum(Q(float(W[i,j]))*Q(float(delta[j]))for j in range(3))for i in range(3)])
  bound=[row[0]+sum(a*h for a,h in zip(row[1:],H,strict=True))for row in coeff]
  return dict(NEW_error_coefficients=coeff,H=H,actual_error=errors,component_bounds=bound,
   H_is_synthetic_NOT_new_episode_box_or_gate=True)
 control('NEW-conditional-cache-projection-bound-keeps-exact-defect',dict(V=VV,R=RR,t=tt,C0=C0,Rcw=W),cache_bound,
  lambda r:all(abs(e)<=r['component_bounds'][i]for row in r['actual_error']for i,e in enumerate(row)))
 def norm_signed():
  x=[-.25,.125,-.0625];v=[.03125,-.5,.25]
  interval_return=norm_leaf_enclosure(x,v)
  d=float(cKDTree(np.array([v])).query(x,workers=1)[0])
  return dict(x=x,v=v,raw_distance=d,enclosure=interval_return)
 control('NEW-leaf-kernel-signed-coordinate-no-normalization',dict(synthetic_fixed_inputs=True),norm_signed,
  lambda r:r['enclosure']['distance'][0]<=Q(r['raw_distance'])<=r['enclosure']['distance'][1])
 def not_global():
  ret=norm_leaf_enclosure([0.,0,0],[10.,0,0]);d=10.
  return dict(returned_vertex_norm_interval=ret,raw_selected_distance=d,
   another_vertex=[1.,0,0],another_exact_squared_distance=Q(1),
   selected_leaf_roundoff_check_passes=ret['distance'][0]<=Q(d)<=ret['distance'][1],
   false_global_nearest_claim_rejected=(not ret['global_nearest_search_certified']))
 control('NEW-leaf-roundoff-pass-cannot-certify-global-KD-search',dict(selected_vertex=[10.,0,0],closer_vertex=[1.,0,0]),not_global,
  lambda r:r['selected_leaf_roundoff_check_passes']and r['false_global_nearest_claim_rejected']and r['another_exact_squared_distance']<r['returned_vertex_norm_interval']['exact_squared_distance'])
 report=dict(schema='v2d.NEW-source-numerical-contracts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  NEW_contracts=len(controls),contracts=controls,all_pass=True,
  no_episode_source_NN_raster_or_pose_solver_native_physics_GPU_calls=True,
  original_fullpixel_k1_and_training_k2_are_separate_operator_scopes=True,
  RN_finite_leaf_and_projection_bounds_CONDITIONAL_only=True,
  no_global_KD_search_binary_build_or_source_accuracy_proof=True,
  no_old_controls_producers_audits_radii_arrays_gate_tolerances_or_raw_return_changes=True)
 save('report.json',report);emit(dict(kind='AFTER_NEW_contracts',actual_RETURN=report));print(json.dumps(report))

"""NEW analytic object/source-translation contracts, no episode evaluation."""
from pathlib import Path
import datetime,hashlib,json,os
from fractions import Fraction as Q
import numpy as np
from object_translation_maps_v1 import exact_translation_motion_maps,scalar_apply,affine_camera_translation,source_pixel_translation_derivative,translated_triangle_ray_map,translate_support_halfspace
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];O=S/'e037-object-translation-contract-v1';rows=[]
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
def rationals(v):
 if isinstance(v,Q):return dict(n=v.numerator,d=v.denominator)
 if isinstance(v,dict):return {str(k):rationals(x)for k,x in v.items()}
 if isinstance(v,list):return [rationals(x) for x in v]
 return v
with (O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def test(name,proposal,fn):
  journal(log,dict(kind='BEFORE_NEW_analytic_object_translation_contract',name=name,proposal=proposal));result=fn();save(name+'-RETURN.json',result);journal(log,dict(kind='AFTER_NEW_analytic_object_translation_contract',name=name,returned=result));assert result['passed'],name;rows.append(name)
 R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]);T=np.eye(4);T[:3,:3]=R;T[:3,3]=[.25,-.5,1.];p=np.array([1.,2.,3.]);d=np.array([.125,-.25,.5])
 test('world_body_translation_alias',dict(R=R,p=p,d=d),lambda:dict(passed=np.array_equal(R@p+d,R@(p+R.T@d)),world_delta=d,body_delta=R.T@d,wrong_alias_rejected=not np.array_equal(R@p+d,R@(p+d))))
 test('camera_translation_ignores_camera_origin',dict(T=T,d=d),lambda:dict(passed=np.array_equal(affine_camera_translation(T,d),R@d)and np.array_equal((R@(p+d)+T[:3,3])-(R@p+T[:3,3]),R@d),returned=affine_camera_translation(T,d)))
 ts=[0.,.125,.375,.875];x=np.array([[0.,0.,0.],[.125,.25,0.],[.5,.25,.125],[.25,.75,.25]]);maps=exact_translation_motion_maps(ts)
 def motion_identity(kind):
  direct=[]
  for i in range(len(ts)-1):direct.append([(Q.from_float(float(x[i+1,k]))-Q.from_float(float(x[i,k])))/(Q.from_float(ts[i+1])-Q.from_float(ts[i]))for k in range(3)])
  if kind=='acceleration':direct=[[(direct[i+1][k]-direct[i][k])/( (Q.from_float(ts[i+2])-Q.from_float(ts[i]))/2)for k in range(3)]for i in range(len(ts)-2)]
  returned=[scalar_apply(r,x)for r in maps[kind]]
  return dict(passed=returned==direct,exact_maps=rationals(maps[kind]),exact_return=rationals(returned),exact_direct=rationals(direct))
 test('nonuniform_object_velocity_maps',dict(timestamps=ts,positions=x),lambda:motion_identity('velocity'))
 test('nonuniform_midpoint_object_acceleration',dict(timestamps=ts,positions=x),lambda:motion_identity('acceleration'))
 test('translation_nullspace_and_clock_locality',dict(maps=rationals(maps)),lambda:dict(passed=all(sum(r.values())==0 for rs in maps.values()for r in rs)and [list(r)for r in maps['velocity']]==[[0,1],[1,2],[2,3]]and [set(r)for r in maps['acceleration']]==[{0,1,2},{1,2,3}],constant_global_delta_does_not_change_motion=True))
 test('fixed_stage_norm_invariance',dict(R=R,x=x),lambda:dict(passed=np.array_equal(np.sum(x*x,axis=1),np.sum((x@R.T)**2,axis=1)),world_and_native_motion_norms_equal_under_FIXED_orthogonal_stage=True,NOT_robot_root_cap_authority=True))
 def angularcheck():
  A=np.eye(4);B=np.eye(4);A[:3,3]=[.5,-.25,1.];B[:3,:3]=R;B[:3,3]=[-.5,.25,2.];raw=B@np.linalg.inv(A);A[:3,3]+=[.125,.5,-.25];B[:3,3]+=[-.25,.125,.5];moved=B@np.linalg.inv(A);return dict(passed=np.array_equal(raw[:3,:3],moved[:3,:3]) and not np.array_equal(raw[:3,3],moved[:3,3]),actual_original_relative_rotation=raw[:3,:3],actual_translated_relative_rotation=moved[:3,:3],relative_rotation_independent_of_TRANSLATION_ONLY=True,all_original_angular_maxima_and_p95_still_must_pass=True)
 test('fixed_rotations_no_angular_motion_change',dict(translation_only=True),angularcheck)
 speeds=np.r_[np.ones(19),2.]
 test('p95_is_separate_from_maximum_speed',dict(speeds=speeds,maxcap=2.,p95cap=1.),lambda:dict(passed=speeds.max()<=2 and np.quantile(speeds,.95)>1,actual_p95=float(np.quantile(speeds,.95)),original_p95_NOT_implied_by_only_maximum_SOC=True))
 C=np.array([.25,-.125,2.]);K=np.array([[400.,0.,320.],[0.,600.,240.],[0.,0.,1.]])
 def pixelcheck():
  got=source_pixel_translation_derivative(C,K,T);eps=1e-25;direct=[]
  for j in range(3):
   cp=C.astype(complex)+1j*eps*R[:,j];direct.append(np.imag(np.array([K[0,0]*cp[0]/cp[2]+K[0,2],K[1,1]*cp[1]/cp[2]+K[1,2]]))/eps)
  err=float(abs(got['J']-np.array(direct).T).max());return dict(passed=err<1e-12,returned=got,independent_complex_error=err,diagnostic_NOT_candidate_image_tolerance=True)
 test('NEW_world_object_translation_pixel_derivative',dict(C=C,K=K,camera=T),pixelcheck)
 test('nonpositive_source_depth_UNRESOLVED',dict(camera_point=[0.,0.,0.]),lambda:dict(passed=source_pixel_translation_derivative([0,0,0],K,T)['status'].startswith('UNRESOLVED'),no_denominator_sign_assumption=True))
 tri=np.array([[-1.,-1.,2.],[1.,-1.,2.],[0.,1.,2.]]);ray=np.array([.125,0.,1.]);delta=np.array([.125,-.25,.5])
 def rootcheck():
  got=translated_triangle_ray_map(tri,ray);root=got['depth_constant']+np.dot(got['depth_translation_coefficients'],delta);n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);err=float(abs(np.dot(n,root*ray-(tri[0]+delta))));return dict(passed=err==0,returned=got,translated_plane_residual=err,root=float(root),first_surface_or_face_hit_NOT_implied=True)
 test('translated_triangle_ray_affine_root',dict(triangle=tri,ray=ray,delta=delta),rootcheck)
 test('parallel_triangle_ray_remains_UNRESOLVED',dict(triangle=tri,ray=[1.,0.,0.]),lambda:dict(passed=translated_triangle_ray_map(tri,[1.,0.,0.])['status'].startswith('UNRESOLVED'),parallel_case_retained=True))
 def supportcheck():
  h=translate_support_halfspace(1.,.5,[1.,0.,0.]);move=np.array([.5,0.,0.]);newgap=.5+move[0]-1.;return dict(passed=float(np.dot(h['normal'],move))>=h['minimum_dot_translation']and newgap>=-.001 and .5-move[0]-1.<-.001,returned=h,positive_direction_gap=newgap,opposite_direction_fails=True,NOT_measured_contact_or_force=True)
 test('translation_support_plane_sign_and_1mm',dict(hand_max=1.,object_min=.5,normal=[1.,0.,0.]),supportcheck)
 def strictcheck():
  gate=lambda z:z>.05;sourcegate=lambda z:z>.01
  # ORIGINAL object-motion tolerance is 1e-9 in retained object producers;
  # this algebra does not substitute the robot motion tolerance 1e-7.
  return dict(passed=not gate(.05)and not sourcegate(.01)and (1.+5e-8)>1.+1e-9 and (1.+5e-8)<=1.+1e-7,object_tolerance=1e-9,robot_tolerance_separate=1e-7,strict_near_planes_retained=True)
 test('object_vs_robot_authority_and_strict_planes',dict(object_tol=1e-9,robot_tol=1e-7),strictcheck)
 result=dict(schema='v2d.NEW-object-translation-analytic-contracts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_contracts_passed=len(rows),contracts=rows,old_controls_NOT_repeated=True,episode_inputs_or_numerical_source_native_solver_calls=0,no_candidate_or_source_truth=True,diagnostic_bounds_NOT_original_candidate_tolerances=True)
 save('report.json',result);journal(log,dict(kind='AFTER_NEW_contract_suite',returned=result));print(json.dumps(result))

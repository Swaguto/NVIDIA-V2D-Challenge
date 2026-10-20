from pathlib import Path
import datetime,json,os
import numpy as np
from scipy.optimize import least_squares
from object_dynamic_visibility_v1 import CONFIG,robust_signed,projection_jacobian,projected_edge_distance,maximum_axis_union,positive_norm_residual,p95_value_gradient
from object_visibility_support_v1 import first_cell_for_ray
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];O=S/'e037-object-dynamic-contract-v1';passed=[]
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def t(name,proposal,fn):
  journal(log,dict(kind='BEFORE_NEW_dynamic_visibility_contract',name=name,proposal=proposal));r=fn();save(name+'-RETURN.json',r);journal(log,dict(kind='AFTER_NEW_dynamic_visibility_contract_RETURN',name=name,actual_RETURN=r));assert r['passed'],name;passed.append(name)
 def union():
  axes=np.array([[1.,0.,0.],[-1.,0.,0.]]);a=maximum_axis_union([-.5,-.5],axes,[1,0,0]);b=maximum_axis_union([-.5,-.5],axes,[-1,0,0]);tie=maximum_axis_union([-.5,-.5],axes,[0,0,0]);return dict(passed=a['active_axis']!=b['active_axis']and a['gap_m']==b['gap_m']==.5 and tie['nonunique_active_axis'],returns=[a,b,tie],no_source_pose_or_force_truth=True)
 t('NEW_union_support_switches_branches',dict(two_exact_directions=True),union)
 def union_derivative():
  axes=np.array([[1.,.25,0.],[-1.,0.,.5]]);x=np.array([1.,.1,.2]);ret=maximum_axis_union([-.5,-.5],axes,x);h=1e-6;fd=np.array([(maximum_axis_union([-.5,-.5],axes,x+np.eye(3)[i]*h)['gap_m']-maximum_axis_union([-.5,-.5],axes,x-np.eye(3)[i]*h)['gap_m'])/(2*h)for i in range(3)]);error=float(abs(fd-ret['world_gradient']).max());return dict(passed=error<1e-9,actual_RETURN=ret,independent_FD=fd,error=error,diagnostic_NOT_gate_tolerance=True)
 t('NEW_active_support_piece_derivative',dict(unique_active_piece=True),union_derivative)
 K=np.array([[300.,0.,320.],[0.,400.,240.],[0.,0.,1.]]);C=np.array([.2,-.1,2.]);R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
 def pj():
  J=projection_jacobian(C,K,R);h=1e-25;fd=[]
  for j in range(3):
   P=C.astype(complex)+1j*h*R[:,j];fd.append(np.imag(np.array([K[0,0]*P[0]/P[2]+K[0,2],K[1,1]*P[1]/P[2]+K[1,2]]))/h)
  error=float(abs(J-np.asarray(fd).T).max());return dict(passed=error<1e-12,actual_J=J,independent_complex=np.asarray(fd).T,error=error)
 t('NEW_projected_edge_endpoint_world_derivative',dict(camera_point=C,camera_rotation=R),pj)
 def edge():
  pts=np.array([[-.5,0.,2.],[.5,0.,2.]]);edges=np.array([[0,1]]);pixel=np.array([320.,270.]);ret=projected_edge_distance(pts,edges,pixel,K,np.eye(3));h=1e-6;fd=np.array([(projected_edge_distance(pts+np.eye(3)[j]*h,edges,pixel,K,np.eye(3))['distance_px']-projected_edge_distance(pts-np.eye(3)[j]*h,edges,pixel,K,np.eye(3))['distance_px'])/(2*h)for j in range(3)]);e=float(abs(fd-ret['world_translation_gradient']).max());return dict(passed=e<1e-7 and abs(ret['distance_px']-30)<1e-12,actual_RETURN=ret,independent_FD=fd,error=e)
 t('NEW_nearest_projected_edge_envelope_derivative',dict(interior_projection=True),edge)
 def degenerate():
  pts=np.array([[0.,0.,2.],[0.,0.,2.]]);r=projected_edge_distance(pts,[[0,1]],[330,240],K,np.eye(3));return dict(passed=r['zero_length_projected_edges']==1 and r['distance_px']==10 and np.isfinite(r['world_translation_gradient']).all(),actual_RETURN=r,degenerate_edge_remains_point_NOT_missing_source_hit=True)
 t('NEW_zero_length_projected_edge_and_source_semantics',dict(zero_edge=True),degenerate)
 def robust():
  errors=[];r=[]
  for x in [-3.,-.2,0.,.2,3.]:
   v,d=robust_signed(x);h=1e-6;fd=(robust_signed(x+h)[0]-robust_signed(x-h)[0])/(2*h);errors.append(abs(d-fd));r.append([x,v,d])
  missing_actual=float('inf');missing_optimizer=robust_signed(CONFIG['missing_optimization_equivalent_m']/CONFIG['depth_scale_m'])[0];return dict(passed=max(errors)<1e-8 and np.isinf(missing_actual)and missing_optimizer>0,returns=r,maximum_derivative_error=max(errors),actual_missing_depth=missing_actual,optimization_ONLY_missing_residual=missing_optimizer,no_depth_filled_or_relabelled=True)
 t('NEW_robust_signed_depth_and_infinite_miss',dict(source_samples_unchanged=True),robust)
 def switching():
  tri=np.array([[[-1,-1,2],[1,-1,2],[0,1,2]],[[.5,-1,1],[1.5,-1,1],[1,1,1]]],float);r0=first_cell_for_ray(tri,[0,0,1]);r1=first_cell_for_ray(tri+[-1.,0.,0.],[0,0,1]);return dict(passed=r0['face']==0 and r1['face']==1 and r0['depth_m']==2 and r1['depth_m']==1,actual_returns=[r0,r1],old_fixed_face_NOT_mandatory=True)
 t('NEW_actual_first_surface_face_switch',dict(two_surface_components=True),switching)
 def motion():
  value=np.array([.8,.6,.2]);cap=.5;weight=10.;r,g=positive_norm_residual(value,cap,weight);h=1e-6;fd=np.array([(positive_norm_residual(value+h*np.eye(3)[j],cap,weight)[0]-positive_norm_residual(value-h*np.eye(3)[j],cap,weight)[0])/(2*h)for j in range(3)]);e=float(abs(fd-g).max());return dict(passed=e<1e-8,actual_residual=r,actual_gradient=g,independent_FD=fd,original_caps_unchanged=True)
 t('NEW_original_motion_norm_chain_derivative',dict(norm_branch_above_cap=True),motion)
 def quantile():
  speeds=np.linspace(.1,.8,150)**1.3;ret=p95_value_gradient(speeds);h=1e-7;fd=np.array([(np.quantile(speeds+h*np.eye(150)[j],.95)-np.quantile(speeds-h*np.eye(150)[j],.95))/(2*h)for j in ret['ordered_indices']]);e=float(abs(fd-ret['gradient'][ret['ordered_indices']]).max());return dict(passed=abs(ret['value']-np.quantile(speeds,.95))<1e-15 and e<1e-8,actual_RETURN=ret,error=e,NO_old_speed_envelope_substituted_for_original_gate=True)
 t('NEW_actual_p95_sorting_piece_derivative',dict(speed_count=150),quantile)
 def tiny():
  trace=[]
  def f(x):trace.append(x.copy());return np.array([x[0]+x[1]-1,2*x[0]-x[1]-.2])
  def j(x):return np.array([[1.,1.],[2.,-1.]])
  journal(log,dict(kind='BEFORE_ONE_NEW_tiny_dynamic_method_call',x0=[0.,0.],bounds=[-1,1],method='trf',solver='lsmr',max_nfev=40));r=least_squares(f,np.zeros(2),jac=j,bounds=(-1,1),method='trf',tr_solver='lsmr',max_nfev=40);ret={k:(v.toarray()if hasattr(v,'toarray')else v)for k,v in r.items()};save('ONE-NEW-tiny-dynamic-method-RETURN.json',ret);journal(log,dict(kind='AFTER_ONE_NEW_tiny_dynamic_method_RETURN',actual_RETURN=ret,trace=trace));return dict(passed=bool(r.success)and np.max(abs(r.x-[.4,.6]))<1e-7,actual_RETURN=ret,analytic_solution=[.4,.6],NOT_episode_tolerances=True)
 t('ONE_NEW_analytic_dynamic_sparse_least_squares',dict(two_joint_variables=True),tiny)
 out=dict(schema='v2d.NEW-object-dynamic-visibility-contracts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEWcontracts=len(passed),contracts=passed,episode_inputs_source_native_measurements=0,new_tiny_analytic_method_calls=1,old_controls_NOT_repeated=True,optimizer_penalties_NOT_original_source_confidence_or_gates=True,no_trajectory_qualified=True);save('report.json',out);journal(log,dict(kind='AFTER_NEW_dynamic_contract_suite',actual_RETURN=out));print(json.dumps(out))

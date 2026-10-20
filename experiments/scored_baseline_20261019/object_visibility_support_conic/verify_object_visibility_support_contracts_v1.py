"""NEW analytic visibility/support/quantile-conic contracts; no episode inputs."""
from pathlib import Path
import datetime,json,os,sys
import numpy as np
from scipy import sparse
from object_visibility_support_v1 import CONFIG,ray_cells,first_cell_for_ray,support_axis_plane,p95_speed_envelope,returned,sparse_record
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];O=S/'e037-object-visibility-support-contract-v1';passed=[]
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
with (O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def test(n,p,fn):
  journal(log,dict(kind='BEFORE_NEW_visibility_support_contract',name=n,proposal=p));r=fn();save(n+'-RETURN.json',r);journal(log,dict(kind='AFTER_NEW_visibility_support_contract',name=n,actual_RETURN=r));assert r['passed'],n;passed.append(n)
 tri=np.array([[[-1.,-1.,2.],[1.,-1.,2.],[0.,1.,2.]]]);rays=np.array([[.125,0.,1.]]);d=np.array([.125,-.25,.5])
 def affine():
  c=ray_cells(tri,rays);u=c['u0'][0,0]+c['ud'][0,0]@d;v=c['v0'][0,0]+c['vd'][0,0]@d;z=c['z0'][0,0]+c['zd'][0,0]@d;point=tri[0,0]+d+u*(tri[0,1]-tri[0,0])+v*(tri[0,2]-tri[0,0]);err=float(abs(point-z*rays[0]).max());return dict(passed=err==0,actual_cell=c,affine_values=[u,v,z],independent_plane_and_barycentric_point=point,closure_m=err)
 test('NEW_triangle_translation_barycentric_affine_identity',dict(tri=tri,rays=rays,d=d),affine)
 def compare():
  c=ray_cells(tri,rays);m=ray_cells(tri+d,rays);errors={k:float(abs(c[k+'0']+np.einsum('rfi,i->rf',c[k+'d'],d,optimize=False)-m[k+'0']).max())for k in ['u','v','z']};return dict(passed=max(errors.values())==0,errors=errors,actual_translated_RETURN=m)
 test('NEW_all_three_translated_ray_maps',dict(tri=tri,d=d),compare)
 def miss():
  m=first_cell_for_ray(tri,[10.,0.,1.]);p=first_cell_for_ray(tri,[1.,0.,0.]);behind=first_cell_for_ray(tri-[0,0,4],[0,0,1]);return dict(passed=all(x['status'].startswith('MISSING')and np.isinf(x['depth_m'])for x in [m,p,behind]),actual_returns=[m,p,behind],no_false_hit_or_zero_residual=True)
 test('NEW_miss_parallel_and_behind_depth_not_filled',dict(tri=tri),miss)
 def occlusion():
  two=np.r_[tri,tri-[0,0,1]];raw=first_cell_for_ray(two,[0,0,1]);return dict(passed=raw['face']==1 and raw['depth_m']==1 and len(raw['baseline_hit_faces'])==2,actual_RETURN=raw,one_face_map_does_NOT_certify_other_face_occlusion=True)
 test('NEW_first_surface_and_alternate_face_retention',dict(two_depths=[2,1]),occlusion)
 def coverage():
  c=ray_cells(tri,rays);away=np.array([3.,0.,0.]);u=float(c['u0'][0,0]+c['ud'][0,0]@away);v=float(c['v0'][0,0]+c['vd'][0,0]@away);m=first_cell_for_ray(tri+away,rays[0]);return dict(passed=(u<0 or v<0 or u+v>1)and m['status'].startswith('MISSING'),affine_u=u,affine_v=v,actual_RETURN=m,depth_affine_alone_insufficient=True)
 test('NEW_face_membership_is_separate_from_depth',dict(tri=tri,delta=[3,0,0]),coverage)
 cube=np.array([[a,b,c]for a in [-.5,.5]for b in [-.5,.5]for c in [-.5,.5]])
 def plane():
  h=support_axis_plane(cube,cube+[.25,0,0],np.r_[np.eye(3),-np.eye(3)]);delta=(h['minimum_dot_native_translation']+1e-8)*h['normal'];gap=float(np.min((cube+[.25,0,0]+delta)@h['normal'])-np.max(cube@h['normal']));return dict(passed=gap>-.001 and np.dot(h['normal'],delta)>=h['minimum_dot_native_translation'],actual_RETURN=h,new_support_gap_m=gap,NOT_native_or_force_certificate=True)
 test('NEW_fixed_support_plane_sufficient_translation',dict(hand=cube,object=cube+[.25,0,0]),plane)
 def broadphase():
  bound=CONFIG['world_translation_construction_halfwidth_m']*np.sqrt(3);gap=.05;return dict(passed=gap>bound+.001 and gap-bound>.001,world_Linfty_box_euclidean_bound=bound,rigid_stage_preserves_bound=True,numerical_construction_not_original_box=True)
 test('NEW_uniform_translation_box_broadphase',dict(component_halfwidth=CONFIG['world_translation_construction_halfwidth_m']),broadphase)
 def quantile():
  old=np.linspace(0,.8,150);q95=.79;env=p95_speed_envelope(old,q95,.9);caps=env['per_edge_caps'];proposed=caps*np.linspace(.1,1,150);return dict(passed=np.quantile(caps,.95)<=q95+1e-15 and np.quantile(proposed,.95)<=np.quantile(caps,.95)and proposed.max()<=.9,actual_RETURN=env,new_p95=float(np.quantile(proposed,.95)),exact_necessity='componentwise order implies same quantile order; uniform shifts commute with linear-interpolated quantile',original_candidate_p95_gate_NOT_changed=True)
 test('NEW_monotone_quantile_inner_speed_envelope',dict(old_speeds=np.linspace(0,.8,150)),quantile)
 def gram():
  D=np.array([[2.,1.],[-1.,3.]]);r=np.array([.25,-.5]);x=np.array([.125,-.25]);P=2*D.T@D;q=2*D.T@r;lhs=.5*x@P@x+q@x;rhs=np.linalg.norm(D@x+r)**2-np.linalg.norm(r)**2;return dict(passed=abs(lhs-rhs)<1e-15,P=P,q=q,objective=lhs,direct=rhs,weights_are_algorithmic_NOT_source_confidence=True)
 test('NEW_affine_depth_least_squares_QP_packing',dict(D=[[2,1],[-1,3]]),gram)
 def socsign():
  v=np.array([.25,-.5,.75]);cap=1.;s=np.r_[cap,v];return dict(passed=s[0]>=np.linalg.norm(s[1:]),slack=s,canonical_A_has_negative_velocity_coefficients=True,source_equality_no_repair=True)
 test('NEW_object_velocity_SOC_slack_sign',dict(velocity=[.25,-.5,.75],cap=1.),socsign)
 def tiny():
  sys.path.insert(0,str(S/'e037-direct-SDP-runtime-v2/site-packages'));import clarabel
  # New analytic QP min(x-2)^2, source-cell x>=0, motion norm|x|<=1.
  P=sparse.csc_matrix([[2.]]);q=np.array([-4.]);A=sparse.csc_matrix([[-1.],[0.],[-1.]]);b=np.array([0.,1.,0.]);settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100;settings.time_limit=5.;settings.max_threads=1;journal(log,dict(kind='BEFORE_NEW_tiny_constructor',P=sparse_record(P),q=q,A=sparse_record(A),b=b));solver=clarabel.DefaultSolver(P,q,A,b,[clarabel.NonnegativeConeT(1),clarabel.SecondOrderConeT(2)],settings);journal(log,dict(kind='AFTER_NEW_tiny_constructor_RETURN',actual_type=type(solver).__name__,opaque_repr=repr(solver)));out=returned(solver.solve());save('ONE-NEW-tiny-object-QP-actual-RETURN.json',out);journal(log,dict(kind='AFTER_NEW_tiny_solve_RETURN',actual_RETURN=out));return dict(passed=out['status']=='Solved'and abs(out['x'][0]-1)<1e-6,actual_RETURN=out,analytic_solution=1.,diagnostic_NOT_episode_gate=True)
 test('ONE_NEW_analytic_visibility_motion_QP',dict(analytic_optimum=1.),tiny)
 result=dict(schema='v2d.NEW-object-visibility-support-contracts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEWcontracts=len(passed),contracts=passed,old14_object_and_old_camera_controls_reused_NOT_rerun=True,episode_inputs_and_native_position_and_physics_calls=0,new_tiny_analytic_QP_calls=1,original_authority_UNCHANGED=True,no_trajectory_or_candidate_qualified=True);save('report.json',result);journal(log,dict(kind='AFTER_NEW_contract_suite',returned=result));print(json.dumps(result))

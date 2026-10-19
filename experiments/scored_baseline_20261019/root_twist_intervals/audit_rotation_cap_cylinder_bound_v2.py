"""NEW cap-cylinder necessary bound; saved old ball calculations are not rerun."""
from pathlib import Path
import datetime,hashlib,json,math,time
S=Path(__file__).resolve().parents[1];O=S/'e037-source-rotation-cap-cylinder-bound-v2';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
P=S/'e037-source-rotation-image-ball-bound-v1/report.json';prior=json.loads(P.read_text());F=S/'e037-thumb-ray-independent-validation-v1/frozen';assert sha(F/'manifest.json')==prior['input_manifest_sha256']
with np.load(F/'targets.npz') as z:q=z['robot_qpos'].copy()
with np.load(F/'aligned-source-reference.npz') as z:cameras=z['camera_c2w'].copy()
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz') as z:K=z['intrinsics'].copy()
stage=np.array(json.loads((F/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
def support(C,axis,half_height,perp_radius,w):
 axial=float(w@axis);perp=w-axial*axis;norm=float(np.linalg.norm(perp));point=C-half_height*np.sign(axial)*axis
 if norm>1e-14:point-=perp_radius*perp/norm
 value=float(w@C-half_height*abs(axial)-perp_radius*norm)
 assert abs(float(w@point)-value)<1e-9
 return value,point
def bound(C,axis,h,r,intrinsics,source_pixel,n):
 assert abs(np.linalg.norm(axis)-1)<1e-10 and h>=0 and r>=0
 zmin=float(C[2]-h*abs(axis[2])-r*math.sqrt(max(0.,1-axis[2]**2)))
 if zmin<=0:return dict(resolved=False,reason='outer cylinder crosses camera plane')
 v=np.r_[n*[intrinsics[0,0],intrinsics[1,1]],float(n@([intrinsics[0,2],intrinsics[1,2]]-source_pixel))]
 def equation(t):return support(C,axis,h,r,v-t*np.array([0.,0.,1.]))[0]
 extent=float(np.linalg.norm(v)*(np.linalg.norm(C)+h+r)/zmin+1);low,high=-extent,extent;assert equation(low)>=-1e-8 and equation(high)<=1e-8;root,result=brentq(equation,low,high,xtol=1e-12,rtol=1e-14,full_output=True);assert result.converged;plane=v-root*np.array([0.,0.,1.]);value,point=support(C,axis,h,r,plane);pixel=np.array([intrinsics[0,0]*point[0]/point[2]+intrinsics[0,2],intrinsics[1,1]*point[1]/point[2]+intrinsics[1,2]]);assert abs(float(n@(pixel-source_pixel))-root)<1e-7 and abs(value)<1e-8
 return dict(resolved=True,normal_from_source=n.tolist(),directional_image_lower_bound_px=float(root),supporting_plane_camera=plane.tolist(),tangent_camera_point=point.tolist(),support_residual=float(value),minimum_cylinder_depth_m=zmin,root_converged=True)
controls=[];testK=np.diag([100.,100.,1.]);n=np.array([1.,0.]);c=bound(np.array([.1,.2,1.]),np.array([0.,0.,1.]),0,0,testK,np.zeros(2),n);assert abs(c['directional_image_lower_bound_px']-10)<1e-10;controls.append('point_support')
c=bound(np.array([0.,0.,1.]),np.array([0.,0.,1.]),.1,.2,testK,np.zeros(2),n);assert abs(c['directional_image_lower_bound_px']+100*.2/.9)<1e-9;controls.append('axis_cylinder_exact_support')
axis=np.array([.3,-.1,.8]);axis/=np.linalg.norm(axis);c=bound(np.array([.1,-.1,1.]),axis,.02,.1,testK,np.zeros(2),n);assert c['resolved'] and c['root_converged'];controls.append('oblique_support_tangent_closure')
# Independently sampled cap endpoints lie in the enclosing cylinder.
alpha=.2;rad=.1;root=np.array([.1,.2,1.]);mid=root+rad*(1+math.cos(alpha))/2*axis;h=rad*(1-math.cos(alpha))/2;perp=rad*math.sin(alpha);rng=np.random.default_rng(84)
for _ in range(80):
 direction=rng.normal(size=3);direction/=np.linalg.norm(direction);angle=float(rng.uniform(0,alpha));point=root+rad*Rotation.from_rotvec(direction*angle).apply(axis);offset=point-mid;axial=float(offset@axis);assert abs(axial)<=h+1e-12 and np.linalg.norm(offset-axial*axis)<=perp+1e-12
controls.append('sampled_orientation_cap_cylinder_containment')
rows=[]
for saved in prior['rows']:
 frame=saved['frame'];camera=cameras[frame];A=camera[:3,:3].T@stage.T;offset=camera[:3,:3].T@camera[:3,3];root_camera=A@q[frame,1,:3]-offset;model_tip=np.array(saved['outer_ball_center_camera']);axis=(model_tip-root_camera)/saved['local_thumb_radius_m'];assert abs(np.linalg.norm(axis)-1)<1e-10;alpha=saved['source_model_orientation_cap_rad'];assert 0<=alpha<=math.pi/2;rad=saved['local_thumb_radius_m'];center=root_camera+rad*(1+math.cos(alpha))/2*axis;h=rad*(1-math.cos(alpha))/2;perp=rad*math.sin(alpha);source=np.array(saved['source_thumb_pixel']);normals=[np.array([math.cos(t),math.sin(t)]) for t in np.linspace(0,2*math.pi,32,endpoint=False)]
 if 'normal_from_source_pixel' in saved['bound']:normals.append(np.array(saved['bound']['normal_from_source_pixel']))
 certs=[bound(center,axis,h,perp,K,source,n) for n in normals];resolved=[c for c in certs if c['resolved']];chosen=max(resolved,key=lambda c:c['directional_image_lower_bound_px']) if resolved else None;lower=max(0.,chosen['directional_image_lower_bound_px']) if chosen else 0.;margin=lower-saved['original_source_image_error_cap_px'];rows.append(dict(frame=frame,kind=saved['kind'],frozen_thumb_joints_rad=saved['preferred_thumb_joints_rad'],cylinder_center_camera=center.tolist(),axis_camera=axis.tolist(),half_height_m=float(h),perpendicular_radius_m=float(perp),source_model_orientation_cap_rad=alpha,original_source_image_error_cap_px=saved['original_source_image_error_cap_px'],all_directional_certificates=certs,finite_directional_image_lower_bound_px=lower,chosen_certificate=chosen,separation_margin_px=float(margin),frozen_thumb_configuration_excluded=bool(chosen is not None and margin>1e-6),finite_normals_cannot_establish_feasibility=True,other_thumb_angles_and_task_impossibility_not_assessed=True))
report=dict(schema='v2d.source-rotation-cap-cylinder-bound.v2',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),input_manifest_sha256=sha(F/'manifest.json'),saved_ball_report_sha256=sha(P),new_analytical_controls_passed=controls,old_ball_or_native_or_source_jobs_not_repeated=True,rows=rows,method='For alpha<=pi/2, the endpoint cap lies in an axial cylinder with parallel interval[r cos(alpha),r] and perpendicular radius r sin(alpha). Directional projective support uses a monotone scalar root and a saved tangent/plane certificate.',all_separating_directions_certificates_saved=True,root_component_box_ignored_for_optimistic_bound=True,numerical_certificate_not_global_task_or_source_truth=True,current_reference_and_all_original_gates_unchanged=True,no_native_model_compile_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,whole_task_or_policy_qualified=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(controls=controls,rows=[dict(frame=r['frame'],lower_bound=r['finite_directional_image_lower_bound_px'],cap=r['original_source_image_error_cap_px'],margin=r['separation_margin_px'],excluded=r['frozen_thumb_configuration_excluded']) for r in rows]),indent=2))

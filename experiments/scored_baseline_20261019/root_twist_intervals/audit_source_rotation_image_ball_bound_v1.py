"""NEW necessary image bound for a frozen thumb vector under an orientation cap."""
from pathlib import Path
import datetime,hashlib,json,math,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');O=S/'e037-source-rotation-image-ball-bound-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();sys.path.insert(0,str(R/'scripts/dex3_training'))
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.kinematics import URDFKinematics
F=S/'e037-thumb-ray-independent-validation-v1/frozen';mf=json.loads((F/'manifest.json').read_text());kin=URDFKinematics(F/'robot/urdfs/g1/dex3_right.urdf');B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037';sources={r['path']:r['sha256'] for r in json.loads((B/'manifest.json').read_text())['sources']}
with np.load(F/'targets.npz') as z:q=z['robot_qpos'].copy()
with np.load(F/'aligned-source-reference.npz') as z:cameras=z['camera_c2w'].copy();human=z['hand_joints_world'].copy()
with np.load(B/'episode.npz') as z:K=z['intrinsics'].copy()
stage=np.array(json.loads((F/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal']);caps_path=S/'e037-coupled-source-disk-pose-bank-v2/report.json';caps_report=json.loads(caps_path.read_text());assert sha(F/'manifest.json')==caps_report['input_manifest_sha256'];before={r['frame']:np.array(r['before_source_errors_px']) for r in caps_report['rows']};seeds_path=S/'e037-source-orientation-coupled-fingers-v1/report.json';seeds={r['frame']:np.array(r['command']) for r in json.loads(seeds_path.read_text())['candidate_commands']};legacy_path=S/'e037-right-thumb-bracketed-v2';legacy=json.loads((legacy_path/'report.json').read_text());assert sha(legacy_path/'right-thumb-reference-commands.npz')==legacy['commands_sha256'];assert not legacy['all_original_motion_caps_passed']
with np.load(legacy_path/'right-thumb-reference-commands.npz') as z:old=z['robot_qpos'].copy()
def uv(point,intrinsics):return np.array([intrinsics[0,0]*point[0]/point[2]+intrinsics[0,2],intrinsics[1,1]*point[1]/point[2]+intrinsics[1,2]])
def directional_support(C,radius,intrinsics,source_pixel,normal):
 assert C[2]>radius>=0 and abs(np.linalg.norm(normal)-1)<1e-10
 fx,fy=intrinsics[0,0],intrinsics[1,1];v=np.r_[normal*[fx,fy],float(np.dot(normal,[intrinsics[0,2],intrinsics[1,2]]-source_pixel))];a=C[2]**2-radius**2;dot=float(v@C);middle=(dot*C[2]-radius**2*v[2])/a;spread=radius*math.sqrt((v[0]*C[0]+v[1]*C[1])**2+a*(v[0]**2+v[1]**2))/a
 return float(middle-spread),float(middle+spread),v
def bound(C,radius,intrinsics,source_pixel):
 if C[2]<=radius:return dict(resolved=False,reason='outer ball crosses camera plane')
 fx,fy=intrinsics[0,0],intrinsics[1,1];a=C[2]**2-radius**2;center=np.array([fx*C[0]*C[2]/a+intrinsics[0,2],fy*C[1]*C[2]/a+intrinsics[1,2]])
 if radius==0:
  distance=float(np.linalg.norm(center-source_pixel));return dict(resolved=True,lower_bound_image_error_px=distance,ellipse_center_px=center.tolist(),ellipse_shape_matrix=[[0,0],[0,0]],closest_point_px=center.tolist(),scalar_root_converged=True)
 shape=(radius/a)**2*np.array([[fx*fx*(C[0]**2+a),fx*fy*C[0]*C[1]],[fx*fy*C[0]*C[1],fy*fy*(C[1]**2+a)]]);eigen,axes=np.linalg.eigh(shape);assert (eigen>0).all();z=axes.T@(source_pixel-center);inside=float(np.sum(z*z/eigen))<=1
 if inside:return dict(resolved=True,lower_bound_image_error_px=0.,ellipse_center_px=center.tolist(),ellipse_shape_matrix=shape.tolist(),source_inside_outer_ellipse=True,scalar_root_converged=True)
 def equation(t):return float(np.sum(eigen*z*z/(eigen+t)**2)-1)
 upper=max(1.,float(eigen.max()))
 for _ in range(64):
  if equation(upper)<=0:break
  upper*=2
 else:raise RuntimeError('Unique ellipse projection root not bracketed')
 value,result=brentq(equation,0,upper,xtol=1e-12,rtol=1e-14,full_output=True);assert result.converged;y=eigen*z/(eigen+value);point=center+axes@y;distance=float(np.linalg.norm(point-source_pixel));normal=(point-source_pixel)/distance;low,high,v=directional_support(C,radius,intrinsics,source_pixel,normal);assert abs(low-distance)<1e-7 and abs(np.sum(y*y/eigen)-1)<1e-10
 # A saved supporting-plane tangent is an independently checkable witness.
 plane=v-low*np.array([0.,0.,1.]);tangent=C-radius*plane/np.linalg.norm(plane);assert abs(np.linalg.norm(tangent-C)-radius)<1e-12 and abs(float(normal@(uv(tangent,intrinsics)-source_pixel))-low)<1e-7
 return dict(resolved=True,lower_bound_image_error_px=max(0.,low),ellipse_center_px=center.tolist(),ellipse_shape_matrix=shape.tolist(),source_inside_outer_ellipse=False,closest_point_px=point.tolist(),normal_from_source_pixel=normal.tolist(),directional_support_px=[low,high],supporting_plane_camera=plane.tolist(),outer_ball_tangent_camera=tangent.tolist(),projection_scalar_root=value,scalar_root_converged=True,ellipse_constraint_residual=float(np.sum(y*y/eigen)-1))
controls=[];testK=np.diag([100.,100.,1.])
point=bound(np.array([.1,.2,1.]),0,testK,np.zeros(2));assert abs(point['lower_bound_image_error_px']-math.sqrt(500))<1e-10;controls.append('zero_radius')
low,high,_=directional_support(np.array([0.,0.,1.]),.1,testK,np.zeros(2),np.array([1.,0.]));assert abs(high-100*.1/math.sqrt(.99))<1e-10 and abs(low+high)<1e-10;controls.append('axis_sphere_tangencies')
control=bound(np.array([.3,.1,1.]),.05,testK,np.zeros(2));assert control['resolved'] and control['lower_bound_image_error_px']>0 and control['scalar_root_converged'];controls.append('oblique_supporting_plane_closure')
inside=bound(np.array([0.,0.,1.]),.1,testK,np.zeros(2));assert inside['lower_bound_image_error_px']==0;controls.append('source_inside_ellipse_not_rejected')
rows=[]
for frame,kind in [(109,'current_reference'),(111,'retained_isolated'),(113,'retained_isolated'),(112,'rejected_path_thumb_component'),(114,'rejected_path_thumb_component')]:
 base=q[frame,1].copy();preferred=seeds[frame].copy() if kind=='retained_isolated' else base.copy()
 if kind=='rejected_path_thumb_component':preferred[11:]=old[frame,1,11:]
 camera=cameras[frame];A=camera[:3,:3].T@stage.T;offset=camera[:3,:3].T@camera[:3,3];p=B/f'source/hands/right/{frame:06d}.json';assert sha(p)==sources[str(p.relative_to(B))];record=json.loads(p.read_text());sourceR=stage@camera[:3,:3]@Rotation.from_rotvec(record['mano']['global_orient']).as_matrix()@np.diag([-1.,-1.,1.]);oldR=Rotation.from_quat(base[3:7]).as_matrix();orientation_cap=float(Rotation.from_matrix(sourceR@oldR.T).magnitude());poses=kin.poses(preferred[:7],dict(zip(mf['joint_names']['right'],preferred[7:],strict=True)));local=Rotation.from_quat(preferred[3:7]).as_matrix().T@(poses['right_thumb_tip'][:3,3]-preferred[:3]);model_tip=base[:3]+sourceR@local;center=A@model_tip-offset;radius=2*np.linalg.norm(local)*math.sin(orientation_cap/2);target=uv((human[frame,1,4]-camera[:3,3])@camera[:3,:3],K);result=bound(center,float(radius),K,target);image_cap=float(before[frame][1]);margin=result.get('lower_bound_image_error_px',0)-image_cap;rows.append(dict(frame=frame,kind=kind,source_record_sha256=sha(p),preferred_thumb_joints_rad=preferred[11:].tolist(),local_thumb_point=local.tolist(),local_thumb_radius_m=float(np.linalg.norm(local)),source_model_orientation_cap_rad=orientation_cap,outer_displacement_ball_radius_m=float(radius),outer_ball_center_camera=center.tolist(),source_thumb_pixel=target.tolist(),original_source_image_error_cap_px=image_cap,bound=result,separation_margin_px=float(margin),frozen_thumb_configuration_excluded_by_necessary_outer_bound=bool(result['resolved'] and margin>1e-6),bound_does_not_assess_other_allowed_thumb_configurations=True,source_model_and_inferred_landmark_not_measured_joint_or_contact_truth=True))
report=dict(schema='v2d.source-rotation-image-ball-bound.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),input_manifest_sha256=sha(F/'manifest.json'),cached_source_caps_sha256=sha(caps_path),isolated_control_report_sha256=sha(seeds_path),rejected_path_report_sha256=sha(legacy_path/'report.json'),analytical_controls_passed=controls,rows=rows,method='An orientation cap moves a frozen local point by at most2r sin(alpha/2). Project that enclosing CAMERA ball to an ellipse; unique convex scalar projection and independent supporting-plane/tangent closure give a necessary image-error lower bound.',root_component_box_ignored_for_optimistic_necessary_bound=True,ellipse_overlap_not_source_or_pose_feasibility=True,numerical_bound_not_formal_global_task_proof=True,no_source_fit_native_model_compile_physics_or_old_bound_jobs_repeated=True,current_verified_reference_and_source_flags_unchanged=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_task_or_policy_qualified=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

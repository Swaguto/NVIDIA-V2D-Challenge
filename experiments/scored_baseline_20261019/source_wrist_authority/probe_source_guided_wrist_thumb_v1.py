"""NEW half source-orientation step plus analytic whole-finger source rays."""
from pathlib import Path
import datetime,hashlib,json,math,sys,time,xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');O=S/'e037-source-guided-wrist-thumb-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
np=c.np;mujoco=c.mujoco
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.planar_thumb_ik import PlanarThumbChain,thumb_point_branches
from v2d_dex3_flashchord.kinematics import URDFKinematics
F=c.F;original=c.q.copy();authority_path=S/'e037-wrist-source-authority-v1/report.json';authority=json.loads(authority_path.read_text());assert authority['source_exporter_and_retarget_code_match_frozen_manifests'];kin=URDFKinematics(F/'robot/urdfs/g1/dex3_right.urdf');chain=PlanarThumbChain((.0255,0,0),(-.0025,.0193),.0458,.035)
def planar_branches(point,lo,hi):
 x,y=point;cosine=(x*x+y*y-.0458**2-.035**2)/(2*.0458*.035)
 if abs(cosine)>1+1e-12:return []
 out=[]
 for b in [math.acos(np.clip(cosine,-1,1)),-math.acos(np.clip(cosine,-1,1))]:
  a=math.atan2(y,x)-math.atan2(.035*math.sin(b),.0458+.035*math.cos(b))
  for aa in [a-2*math.pi,a,a+2*math.pi]:
   for bb in [b-2*math.pi,b,b+2*math.pi]:
    v=np.array([aa,bb])
    if (v>=lo).all() and (v<=hi).all() and not any(np.max(abs(v-z))<1e-10 for z in out):out.append(v)
 return out
controls=[]
for q in [[0.,0.],[.4,.8],[1.2,1.5]]:
 q=np.array(q);point=np.array([.0458*math.cos(q[0])+.035*math.cos(sum(q)),.0458*math.sin(q[0])+.035*math.sin(sum(q))]);bank=planar_branches(point,np.zeros(2),np.array([1.57,1.745]));assert any(np.max(abs(x-q))<1e-7 for x in bank);controls.append(dict(angles=q.tolist(),recovered=True))
assert not planar_branches([.5,.5],np.zeros(2),np.ones(2));controls.append(dict(unreachable_point_rejected=True))
# Bind the X/Z/Z planar index/middle positional chain to the actual immutable URDF.
joints={j.attrib['name']:j for j in ET.parse(F/'robot/urdfs/g1/dex3_right.urdf').getroot().findall('joint')}
for digit,z in [('index',.0285),('middle',-.0285)]:
 for num,xyz in [(0,[.0777,-.0016,z]),(1,[.0458,0,0])]:
  j=joints[f'right_hand_{digit}_{num}_joint'];np.testing.assert_allclose(np.fromstring(j.find('origin').attrib['xyz'],sep=' '),xyz,atol=1e-14);np.testing.assert_array_equal(np.fromstring(j.find('axis').attrib['xyz'],sep=' '),[0,0,1]);assert j.find('origin').attrib['rpy']=='0 0 0'
 np.testing.assert_allclose(np.fromstring(joints[f'right_{digit}_tip_joint'].find('origin').attrib['xyz'],sep=' '),[.035,0,0],atol=1e-14)
rows=[];commands=[]
for frame in [111,113]:
 c.q=original.copy();base=original[frame,1].copy();before=c.evaluate(frame,base[7:]);camera=c.cameras[frame];A=camera[:3,:3].T@c.stage.T;offset=camera[:3,:3].T@camera[:3,3];source=next(x for x in authority['observed_transition_rows'] if x['frame']==frame);targetR=np.array(source['source_model_target_rotation_native']);oldR=Rotation.from_quat(base[3:7]).as_matrix();descent=Rotation.from_matrix(targetR@oldR.T).as_rotvec();newR=Rotation.from_rotvec(.5*descent).as_matrix()@oldR;newroot=base[:7].copy();newroot[3:7]=Rotation.from_matrix(newR).as_quat();relative=Rotation.from_matrix(newR@Rotation.from_quat(c.aligned[frame,1,3:7]).as_matrix().T).as_rotvec();assert abs(relative).max()<=.08+1e-10;c.q[frame,1,:7]=newroot;poses=kin.poses(base[:7],dict(zip(c.manifest['joint_names']['right'],base[7:],strict=True)));world=np.stack([poses[f'right_{d}_tip'][:3,3] for d in ['thumb','index','middle']]);oldpixel=c.uv((A@world.T).T-offset);observed=c.uv((c.human[frame,1,[4,8,12]]-camera[:3,3])@camera[:3,:3]);distance=np.linalg.norm(oldpixel-observed,axis=1);margins=np.minimum(.001,distance*.01);pixels=oldpixel+(observed-oldpixel)*(margins/np.maximum(distance,1e-12))[:,None];lower,upper=c.bounds(frame);angles=base[7:].copy();digit_rows=[];failed=False
 for di,(digit,z,cols) in enumerate([('index',.0285,[0,1]),('middle',-.0285,[2,3])],start=1):
  ray=np.array([(pixels[di,0]-c.K[0,2])/c.K[0,0],(pixels[di,1]-c.K[1,2])/c.K[1,1],1.]);local_origin=newR.T@(A.T@offset-newroot[:3]);local_ray=newR.T@(A.T@ray)
  if abs(local_ray[2])<1e-10:digit_rows.append(dict(digit=digit,failed='image ray parallel to finger plane'));failed=True;continue
  depth=(z-local_origin[2])/local_ray[2];p=local_origin+depth*local_ray;bank=planar_branches(p[:2]-[.0777,-.0016],lower[cols],upper[cols]);ok=bool(depth>.01 and bank)
  digit_rows.append(dict(digit=digit,camera_depth_hypothesis_m=float(depth),bounded_branches=len(bank),valid=ok))
  if ok:angles[cols]=min(bank,key=lambda x:np.linalg.norm(x-base[7+np.array(cols)]))
  else:failed=True
 result=dict(frame=frame,source_direct=source['source_direct_frame_flag'],source_metric=source['source_metric_frame_flag'],model_orientation_hypothesis_is_inferred=True,source_orientation_fraction=.5,source_orientation_error_before_after_deg=[float(np.degrees(np.linalg.norm(descent))),float(np.degrees(Rotation.from_matrix(targetR@newR.T).magnitude()))],root_translation_exact_unchanged=True,root_orientation_original_residual_rad=relative.tolist(),other_digit_ray_results=digit_rows,thumb_bounded_branches=0,half_mm_source_eligible_thumb_candidates=0,one_mm_source_eligible_thumb_candidates=0,selected=None)
 if not failed:
  ray=np.array([(pixels[0,0]-c.K[0,2])/c.K[0,0],(pixels[0,1]-c.K[1,2])/c.K[1,1],1.]);sphere=A@(newroot[:3]+newR@np.array(chain.origin))-offset;radius=np.linalg.norm(chain.offset_xy)+chain.proximal_length+chain.distal_length;aa=ray@ray;bb=-2*ray@sphere;cc=sphere@sphere-radius**2;disc=bb*bb-4*aa*cc;eligible=[]
  if disc>=0:
   dlo=max(.0100001,(-bb-np.sqrt(disc))/(2*aa));dhi=(-bb+np.sqrt(disc))/(2*aa)
   for depth in np.linspace(dlo,dhi,1025):
    point=newR.T@(A.T@(depth*ray+offset)-newroot[:3]);branches=thumb_point_branches(chain,point,np.c_[lower[4:],upper[4:]])
    for t in branches.joints_rad:
     result['thumb_bounded_branches']+=1;a=angles.copy();a[4:]=t;v=c.evaluate(frame,a);pen=max(0.,float(-v['gaps'].min()));image=bool((v['errors']<=before['errors']+1e-6).all());row=dict(finger_angles_rad=a.tolist(),thumb_depth_m=float(depth),native_self_penetration_m=pen,source_image_errors_px=v['errors'].tolist(),source_image_nonregression=image,all_native_self_contacts=v['self_contacts'])
     if image and pen<=.001:result['one_mm_source_eligible_thumb_candidates']+=1
     if image and pen<=.0005:result['half_mm_source_eligible_thumb_candidates']+=1;eligible.append(row)
  if eligible:
   selected=min(eligible,key=lambda x:np.linalg.norm(np.array(x['finger_angles_rad'])-base[7:]));result['selected']=selected;command=base.copy();command[:7]=newroot;command[7:]=selected['finger_angles_rad'];commands.append(dict(frame=frame,command=command.tolist()));v=c.evaluate(frame,command[7:]);target=[]
   for ct in c.data.contact[:c.data.ncon]:
    ba,bb=int(c.model.geom_bodyid[ct.geom1]),int(c.model.geom_bodyid[ct.geom2])
    if ((c.bn[ba].startswith('dex3_right_') and not c.bn[bb].startswith('dex3_')) or (c.bn[bb].startswith('dex3_right_') and not c.bn[ba].startswith('dex3_'))):target.append(dict(bodies=[c.bn[ba],c.bn[bb]],signed_gap_m=float(ct.dist)))
   result['new_native_hand_target_contacts']=target
 rows.append(result);print(json.dumps(result),flush=True)
c.q=original;report=dict(schema='v2d.source-guided-wrist-whole-finger-probe.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),context_sha256=sha(S/'compute/source_orientation_context_v1.py'),input_manifest_sha256=sha(F/'manifest.json'),authority_report_sha256=sha(authority_path),planar_finger_controls=controls,actual_URDF_planar_axis_origin_contract_verified=True,rows=rows,candidate_commands=commands,one_knot_constructive_seeds_only=True,source_model_orientation_is_not_measured_robot_pose=True,original_source_confidence_and_all_three_components_unchanged_on_disk=True,no_old_source_or_native_jobs_repeated=True,no_native_compile_or_physics_steps=True,no_whole_trajectory_motion_motor_grasp_or_task_certificate=True,new_GPU_jobs=0,new_PPO_steps=0,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n')

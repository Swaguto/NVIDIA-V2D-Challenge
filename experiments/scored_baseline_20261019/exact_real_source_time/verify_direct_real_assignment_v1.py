"""Independent NEW rational assignment: direct source/URDF and whole-clock motion.
No solver return is edited or promoted; rejected sparse embeddings stay rejected.
"""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,math,os,sys,time,xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1];O=S/'e037-direct-real-assignment-independent-v1';sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as ctx
import direct_real_feasibility_v3 as q
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
from v2d_dex3_flashchord.kinematics import URDFKinematics
np=ctx.np;read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();J=(O/'all-independent-rational-angle-URDF-source-and-motion-checks.jsonl').open('w');emit=lambda v:g.journal(J,v)
def save(n,v):
 with(O/n).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
E=S/'e037-direct-real-evaluation-v2';raw=read(E/'SOLE-exact-real-SUBPROCESS-RETURN.json');assert raw['exit_code']==0 and raw['stdout'].startswith('sat\n')and '(error'not in raw['stdout'];values=read(E/'ALL-local-model-values-and-unresolved.json');assert not values['unresolved']and len(values['actual_values'])==10;u={}
for name,value in values['actual_values'].items():
 assert value['kind']=='exact_rational'and c.dec(value['lower'])==c.dec(value['upper']);u[name]=c.dec(value['lower'])
records=read(S/'e037-direct-real-multiplication-query-v1/exact-query-constraint-records.json');query_checks=[]
for record in records:
 value=F(0)
 for term in record['coefficients']:
  z=c.dec(term['value'])
  for name,k in zip(record['scope'],term['power'],strict=True):z*=u[name]**k
  value+=z
 passed=value>=0 if record['sense']=='>='else value<=0 if record['sense']=='<='else value>0
 result=dict(id=record['id'],exact_value=c.rec(value),sense=record['sense'],passed=passed);emit(dict(kind='AFTER_NEW_independent_actual_rational_query_value',RETURN=result));query_checks.append(result)
system=read(S/'e037-index-source-time-system-v1/coupled-source-time-polynomial-system.json');angles={name:2*math.atan(float(value))for name,value in u.items()};localtime=[]
for t in system['temporal_constraints']:
 r=t['request'];Q=sum(w*angles[f"u_{frame}_{r['component']}"]for frame,w in zip(r['clocks'],r['weights'],strict=True));lo=c.dec(r['ORIGINAL_lower_rad']);hi=c.dec(r['ORIGINAL_upper_rad']);passed=float(lo)<=Q<=float(hi);result=dict(id=r['id'],kind=r['kind_of_constraint'],actual_angle_combination_rad=Q,ORIGINAL_lower_rad=c.rec(lo),ORIGINAL_upper_rad=c.rec(hi),passed=passed,ORIGINAL_motion_tolerance_ALREADY_in_bounds=1e-7,added_tolerance=0);emit(dict(kind='AFTER_NEW_direct_actual_angle_timestamp_gate',RETURN=result));localtime.append(result)
rootpath=S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz';assert sha(rootpath)=='15a362bbf9eb6956ba487b175ce4d04e0ac42ecfa2e4d245c22799cca81c2932'
with np.load(rootpath)as z:root_hypothesis=z['robot_qpos'].copy()
proposal=ctx.q.copy();proposal[:,1,:7]=root_hypothesis[:,1,:7]
for name,value in angles.items():_,frame,joint=name.split('_');proposal[int(frame),1,7+int(joint)]=value
emit(dict(kind='BEFORE_NEW_REJECTABLE_fullclock_proposal',frames=151,local_rational_values={k:c.rec(v)for k,v in u.items()},F_preserved_as_readonly=True,root_mode_UNQUALIFIED_and_UNCHANGED=True,no_inherited_native_or_physical_certificate=True));np.savez_compressed(O/'UNQUALIFIED-sparse-five-clock-index-proposal.npz',robot_qpos=proposal,timestamps_s=ctx.ts);source_caps={v['frame']:v for v in read(S/'e037-thumb-ray-independent-validation-v1/report.json')['source_rows']if v['hand']=='right'}
urdf=ctx.F/'robot/urdfs/g1/dex3_right.urdf';kin=URDFKinematics(urdf);URDF=ET.parse(urdf).getroot();limit_by_name={a.attrib['name']:a.find('limit')for a in URDF.findall('joint')};right_names=ctx.manifest['joint_names']['right'];source=[];closures=[];local=[];missing=[]
for frame in range(len(ctx.ts)):
 cmd=proposal[frame,1];camera=ctx.cameras[frame];human=(ctx.human[frame,1,[8]]-camera[:3,3])@camera[:3,:3];available=bool(ctx.flags['hand_valid'][frame,1]and np.isfinite(human).all()and human[0,2]>.01)
 if not available:
  unchanged=bool(np.array_equal(proposal[frame,1,7:9],ctx.q[frame,1,7:9]));missing.append(dict(frame=frame,index_commands_unchanged=unchanged));emit(dict(kind='AFTER_NEW_missing_source_anchor_retained',RETURN=missing[-1]));continue
 emit(dict(kind='BEFORE_NEW_actual_URDF_source_check',frame=frame,command=cmd));tip=kin.poses(cmd[:7],dict(zip(right_names,cmd[7:],strict=True)))['right_index_tip'][:3,3];C=(tip@ctx.stage-camera[:3,3])@camera[:3,:3];cap=source_caps[frame]['source_before_after_px'][0][2];depth=bool(C[2]>.01);error=float(np.linalg.norm(ctx.uv(C[None])[0]-ctx.uv(human)[0]))if depth else None;passed=bool(depth and error<=cap+1e-6);result=dict(frame=frame,available_original_index_anchor=True,actual_URDF_camera_point_m=C,actual_depth_passed=depth,error_px=error,ORIGINAL_cap_px=cap,ORIGINAL_image_tolerance_px=1e-6,passed=passed);source.append(result);emit(dict(kind='AFTER_NEW_actual_URDF_source_RETURN',RETURN=result))
 if frame in system['frames']:
  saved=next(v for v in system['source_constraints']if v['frame']==frame)['source_record_COPIED_not_regenerated']['polynomial'];a,b=[float(u[f'u_{frame}_{k}'])for k in [0,1]];D=(1+a*a)*(1+b*b);pred=np.array([np.polynomial.polynomial.polyval2d(a,b,p)for p in saved['camera_numerators']])/D;closure=float(abs(C-pred).max());closures.append(closure);local.append(result);emit(dict(kind='AFTER_NEW_actual_URDF_saved_polynomial_closure',frame=frame,closure_m=closure,representation_check_ONLY_not_source_accuracy=True))
position=[];motions=[]
for hand,s in enumerate(['left','right']):
 names=ctx.manifest['joint_names'][s];tree=ET.parse(ctx.F/f'robot/urdfs/g1/dex3_{s}.urdf').getroot();limits={a.attrib['name']:a.find('limit')for a in tree.findall('joint')};lo=np.array([float(limits[n].attrib['lower'])for n in names]);hi=np.array([float(limits[n].attrib['upper'])for n in names]);speedcaps=np.array([float(limits[n].attrib['velocity'])for n in names]);v=proposal[:,hand,7:];original=ctx.aligned[:,hand,7:];delta=v-original;comp=abs(delta);adj=abs(np.diff(delta,axis=0));second=abs(np.diff(delta,n=2,axis=0));speed=abs(np.diff(v,axis=0))/np.diff(ctx.ts)[:,None];positionbad=np.argwhere((v<lo-1e-8)|(v>hi+1e-8));added_bad=np.argwhere(comp>.25+1e-7);adjbad=np.argwhere(adj>.05+1e-7);secondbad=np.argwhere(second>.05+1e-7);speedbad=np.argwhere(speed>speedcaps+1e-7);record=dict(hand=s,actual_URDF_names=names,actual_URDF_lower_rad=lo,actual_URDF_upper_rad=hi,actual_URDF_velocity_rad_s=speedcaps,independent_position_tolerance_rad=1e-8,ORIGINAL_motion_tolerance_rad=1e-7,component_max_rad=float(comp.max()),adjacent_added_max_rad=float(adj.max()),second_added_max_rad=float(second.max()),actual_speed_max_rad_s=float(speed.max()),position_failures=positionbad,component_failures=added_bad,adjacent_failures=adjbad,second_failures=secondbad,velocity_failures=speedbad,passed=not any(len(a)for a in[positionbad,added_bad,adjbad,secondbad,speedbad]));motions.append(record);emit(dict(kind='AFTER_NEW_WHOLEclock_actual_URDF_original_finger_motion',RETURN=record))
localpass=all(v['passed']for v in query_checks+localtime+local)and len(local)==5 and max(closures)<1e-8;fullsource=all(v['passed']for v in source)and all(v['index_commands_unchanged']for v in missing);fullmotion=all(v['passed']for v in motions);save('independent-assignment-and-fullclock-rejection-certificates.json',dict(query_checks=query_checks,local_actual_angle_timestamp_checks=localtime,local_URDF_source_checks=local,wholeclock_index_source_rows=source,missing_anchors=missing,wholeclock_finger_motion=motions,local_assignment_passed_original_stated_scope=localpass,sparse_fullclock_index_source_passed=fullsource,sparse_fullclock_finger_motion_passed=fullmotion,source_accuracy_and_whole_reference_native_motor_grasp_NOT_qualified=True));J.close();report=dict(schema='v2d.direct-real-assignment-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_new_rational_query_rows=74,all_exact_rational_query_rows_passed=all(v['passed']for v in query_checks),direct_original_angle_timestamp_gates=22,all22_direct_original_gates_passed=all(v['passed']for v in localtime),new_actual_URDF_source_rows=len(source),local_original_source_URDF_points=5,local_original_source_URDF_passed=localpass,max_URDF_saved_camera_polynomial_closure_m=max(closures),wholeclock_index_source_failures=[v['frame']for v in source if not v['passed']],missing_index_anchors_retained=len(missing),wholeclock_original_finger_motion_passed=fullmotion,sparse_embedding_REJECTED_unless_ALL_source_motion_pass=not(fullsource and fullmotion),NOT_full_reference_or_trajectory_qualified=True,original_F_middle_wrist_source_missing_heldout_model_parameters_gates_UNCHANGED=True,new_solver_calls=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',report);print(json.dumps({k:v for k,v in report.items()if k!='all_outputs_sha256'}))

"""Independent compiled certificates, URDF/native witnesses and temporal LPs."""
from pathlib import Path
import datetime,hashlib,json,math,sys,time,xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');O=S/'e037-thumb-conic-independent-v1';start=time.monotonic();sys.path[:0]=[str(S/'compute'),str(R/'scripts/track3')]
import thumb_conic_geometry_v1 as g
import source_orientation_context_v1 as c
from scipy.spatial.transform import Rotation
from scipy.optimize import linprog
from v2d_dex3_flashchord.kinematics import URDFKinematics
from temporal_pose_path import motion_summary
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();P=S/'e037-thumb-conic-clearance-v1';Q=S/'e037-thumb-conic-arcs-v1';pr=json.loads((P/'report.json').read_text());qr=json.loads((Q/'report.json').read_text())
for folder,report in [(P,pr),(Q,qr)]:
 for n,h in report['all_outputs_sha256'].items():assert sha(folder/n)==h
geometry=json.loads((P/'compiled-parent-geometry.json').read_text());certificates=json.loads((P/'all-compiled-parent-clearance-intervals.json').read_text());witnesses=json.loads((Q/'accepted-isolated-point-witnesses.json').read_text());assert len(witnesses)==6;journal=(O/'all-independent-records.jsonl').open('w');pe=np.array(geometry['palm_equations']);te=np.array(geometry['thumb1_equations']);pv=np.array(geometry['palm_vertices_in_thumb0_frame']);tv=np.array(geometry['thumb1_vertices_in_own_frame']);origin=np.array(geometry['thumb1_joint_origin']);certificate_rows=[];max_violation=0.
for cert in certificates:
 if 'ball_certificate' not in cert:continue
 left,right=cert['angle_interval'];ball=cert['ball_certificate'];x=np.array(ball['shared_ball_center']);radius=float(ball['guaranteed_common_radius_m']);plane=cert['support_certificate'];direction=np.array(plane['direction']);assert abs(np.linalg.norm(direction)-1)<1e-10;worst_ball=-1e9;worst_support=-1e9
 for angle in np.linspace(left,right,9):
  rot=Rotation.from_rotvec([0.,0.,angle]).as_matrix();pn=te[:,:3]@rot.T;ph=-te[:,3]+pn@origin;palm_margin=-pe[:,3]-pe[:,:3]@x;thumb_margin=ph-pn@x;ball_excess=max(float(radius-palm_margin.min()),float(radius-thumb_margin.min()));actual_gap=float(((tv@rot.T+origin)@direction).min()-(pv@direction).max());support_excess=float(plane['gap_lower_bound_m']-actual_gap);worst_ball=max(worst_ball,ball_excess);worst_support=max(worst_support,support_excess);assert max(ball_excess,support_excess)<1e-10
 max_violation=max(max_violation,worst_ball,worst_support);row=dict(kind='independent_compiled_certificate_samples',id=cert['id'],sampled_angles=9,maximum_ball_excess_m=worst_ball,maximum_support_excess_m=worst_support,classification=cert['classification']);certificate_rows.append(row);g.journal(journal,row)
frame=114;original=c.q.copy();kin=URDFKinematics(c.F/'robot/urdfs/g1/dex3_right.urdf');names=c.manifest['joint_names']['right'];limits={j.get('name'):j.find('limit').attrib for j in ET.parse(c.F/'robot/urdfs/g1/dex3_right.urdf').getroot().findall('joint') if j.get('type')!='fixed'};lo=np.array([float(limits[n]['lower']) for n in names]);hi=np.array([float(limits[n]['upper']) for n in names]);vel=np.array([float(limits[n]['velocity']) for n in names]);tipnames=['right_hand_palm_link','right_thumb_tip','right_index_tip','right_middle_tip'];rows=[];camera=c.cameras[frame];observed=(c.human[frame,1,[0,4,8,12]]-camera[:3,3])@camera[:3,:3];sourcefile=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/source/hands/right/000114.json';source=json.loads(sourcefile.read_text());sourceR=c.stage@camera[:3,:3]@Rotation.from_rotvec(source['mano']['global_orient']).as_matrix()@np.diag([-1.,-1.,1.])
def independent_images(cmd):
 poses=kin.poses(cmd[:7],dict(zip(names,cmd[7:],strict=True)));points=np.stack([poses[n][:3,3] for n in tipnames]);C=(points@c.stage-camera[:3,3])@camera[:3,:3];errors=np.linalg.norm(c.uv(C)-c.uv(observed),axis=1);return points,errors
_,before=independent_images(original[frame,1]);oldR=Rotation.from_quat(original[frame,1,3:7]).as_matrix();ori_cap=float(Rotation.from_matrix(sourceR@oldR.T).magnitude())
for candidate in witnesses:
 cmd=np.array(candidate['command']);g.journal(journal,dict(kind='BEFORE_independent_native_position',command=cmd,source_queue_index=candidate['queue_index']));points,errors=independent_images(cmd);c.data.qpos[:]=c.model.qpos0;c.data.qvel[:]=0.
 for si,s in enumerate(['left','right']):
  v=original[frame,si] if s=='left' else cmd;ids=c.wrists[s];c.data.qpos[c.model.jnt_qposadr[ids[:3]]]=v[:3];addr=int(c.model.jnt_qposadr[ids[3]]);c.data.qpos[addr:addr+4]=v[[6,3,4,5]];c.data.qpos[c.model.jnt_qposadr[c.fingers[s]]]=v[7:]
 for oi,obj in enumerate(c.manifest['objects']):
  bid=c.canonical_id(c.bn,obj['name']+'_object');addr=int(c.model.jnt_qposadr[c.model.body_jntadr[bid]]);v=c.objects[frame,oi];c.data.qpos[addr:addr+3]=v[:3];c.data.qpos[addr+3:addr+7]=v[[6,3,4,5]]
 c.mujoco.mj_fwdPosition(c.model,c.data);closure=float(abs(points-c.data.xpos[c.tipids]).max());contacts=[];depths=dict(right_self=0.,external=0.,interhand=0.)
 for ct in c.data.contact[:c.data.ncon]:
  a,b=int(c.model.geom_bodyid[ct.geom1]),int(c.model.geom_bodyid[ct.geom2]);ra=c.bn[a].startswith('dex3_right_');rb=c.bn[b].startswith('dex3_right_')
  if not(ra or rb):continue
  other=c.bn[b] if ra else c.bn[a];kind='right_self' if ra and rb else ('interhand' if other.startswith('dex3_left_') else 'external');depths[kind]=max(depths[kind],max(0.,-float(ct.dist)));contacts.append(dict(kind=kind,bodies=[c.bn[a],c.bn[b]],geom_ids=[int(ct.geom1),int(ct.geom2)],signed_gap_m=float(ct.dist)))
 newR=Rotation.from_quat(cmd[3:7]).as_matrix();rootres=Rotation.from_matrix(newR@Rotation.from_quat(c.aligned[frame,1,3:7]).as_matrix().T).as_rotvec();ori=float(Rotation.from_matrix(sourceR@newR.T).magnitude());source_ok=bool((errors<=before+1e-6).all() and ori<=ori_cap+1e-9);position_ok=bool((cmd[7:]>=lo-1e-8).all() and (cmd[7:]<=hi+1e-8).all() and abs(cmd[7:]-c.aligned[frame,1,7:]).max()<=.25+1e-7 and abs(rootres).max()<=.08+1e-9);passed=bool(closure<1e-8 and source_ok and position_ok and max(depths.values())<=.001);row=dict(kind='AFTER_independent_native_position',command=cmd,source_queue_index=candidate['queue_index'],URDF_native_tip_closure_max_m=closure,source_before_after_px=[before,errors],source_model_orientation_before_after_rad=[ori_cap,ori],source_original_nonregression_passed=source_ok,actual_URDF_original_components_passed=position_ok,native_penetrations_m=depths,native_contacts=contacts,isolated_source_native_position_passed=passed);rows.append(row);g.journal(journal,row);assert passed
# Select a robot control by smallest original-command joint change after all
# independent point checks, not an unqualified human/source-pose hypothesis.
selected=min(rows,key=lambda r:float(np.sum((np.array(r['command'])[7:]-original[frame,1,7:])**2)));command=np.array(selected['command']);n=len(c.ts);A=[];rhs=[]
for order in [1,2]:
 D=np.diff(np.eye(n),n=order,axis=0)
 for row in D:A.extend([row,-row])
A=np.array(A);D1=np.diff(np.eye(n),axis=0);Avel=np.r_[D1,-D1];lp_rows=[];motion_only=original[:,1,7:].copy()
for joint in range(7):
 aligned=c.aligned[:,1,7+joint];current=original[:,1,7+joint];bounds=[]
 for index in range(n):
  if index==frame:bounds.append((command[7+joint]-aligned[index],command[7+joint]-aligned[index]))
  elif 104<=index<119:bounds.append((max(lo[joint]-aligned[index],-.25),min(hi[joint]-aligned[index],.25)))
  else:bounds.append((current[index]-aligned[index],current[index]-aligned[index]))
 bound_rhs=np.r_[np.full(2*(n-1)+2*(n-2),.05),vel[joint]*np.diff(c.ts)-np.diff(aligned),vel[joint]*np.diff(c.ts)+np.diff(aligned)];fullA=np.r_[A,Avel];res=linprog(np.zeros(n),A_ub=fullA,b_ub=bound_rhs,bounds=bounds,method='highs');row=dict(kind='conditional_motion_only_LP_return',joint=names[joint],success=bool(res.success),status=int(res.status),message=res.message,iterations=int(res.nit),returned_original_correction=None if res.x is None else res.x,maximum_inequality_residual=None if res.x is None else float((fullA@res.x-bound_rhs).max()),scope=[104,119],candidate114_fixed=True,other_source_or_native_constraints_not_supplied=True);g.journal(journal,row);lp_rows.append(row)
 if res.success:motion_only[:,joint]=aligned+res.x
all_motion=all(r['success'] and r['maximum_inequality_residual']<=1e-7 for r in lp_rows);commands=original.copy();commands[frame,1]=command;delta=commands[:,1,7:]-c.aligned[:,1,7:];sparse=dict(maximum_added_adjacent_rad=float(abs(np.diff(delta,axis=0)).max()),maximum_added_second_rad=float(abs(np.diff(delta,n=2,axis=0)).max()),actual_velocity_max_rad_s=(abs(np.diff(commands[:,1,7:],axis=0))/np.diff(c.ts)[:,None]).max(axis=0),actual_velocity_caps_rad_s=vel)
def transforms(a):
 out=np.tile(np.eye(4),(len(a),1,1));out[:,:3,:3]=Rotation.from_quat(a[:,3:7]).as_matrix();out[:,:3,3]=a[:,:3];return out
root_summary=motion_summary(transforms(commands[:,1]),c.ts);rootcaps=json.loads((S/'e037-wrist-source-authority-v1/report.json').read_text())['original_root_motion_caps'];root_ok=all(root_summary[k]<=v+1e-7 for k,v in rootcaps.items());g.journal(journal,dict(kind='sparse_insertion_diagnostic_only',finger_motion=sparse,root_motion=root_summary,root_caps=rootcaps,root_motion_passed=root_ok));journal.close()
for name,h in c.manifest['files'].items():assert sha(c.F/name)==h
for name,value in [('independent-isolated-witnesses.json',rows),('selected-conditional-robot-control.json',selected),('all-motion-only-LP-returns.json',lp_rows),('motion-only-joint-path.json',motion_only if all_motion else None)]:
 (O/name).write_text(json.dumps(g.plain(value),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.thumb-conic-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),input_manifest_sha256=sha(c.F/'manifest.json'),native_model_sha256=sha(c.MP),compiled_certificate_records_sampled=len(certificate_rows),maximum_sampled_certificate_excess_m=max_violation,independent_isolated_witnesses=len(rows),all_isolated_actual_URDF_source_native_checks_passed=True,URDF_native_maximum_closure_m=max(r['URDF_native_tip_closure_max_m'] for r in rows),selected_queue_index=selected['source_queue_index'],selected_penetrations_m=selected['native_penetrations_m'],conditional_joint_motion_LPs_feasible=all_motion,motion_only_path_not_source_native_root_or_trajectory_qualified=True,sparse_insertion_finger_motion=sparse,sparse_insertion_root_motion_passed=root_ok,no_full_clock_native_swept_motor_grasp_or_whole_source_qualification=True,current_combined_reference_unchanged=True,no_old_family_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_task_or_policy_qualified=False,all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out),indent=2))

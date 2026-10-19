"""NEW independent URDF/source checks; reuse prior mathematical witnesses.

Neither the 28 old conic solves nor the ten new chart solves execute here.
Boundary points that fail actual source/position gates stay rejected.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-closed-2R-independent-witnesses-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
import thumb_conic_geometry_v1 as g
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.kinematics import URDFKinematics
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();oldpath=S/'e037-coupled-root-full-2R-diagnosis-v1/all-new-full-2R-source-domain-records.json';old=json.loads(oldpath.read_text());newpath=S/'e037-new-root-closed-source-domains-v1/all-ten-closed-source-domain-records.json';new=json.loads(newpath.read_text());oldrep=json.loads((oldpath.parent/'report.json').read_text());newrep=json.loads((newpath.parent/'report.json').read_text());assert sha(oldpath)==oldrep['all_outputs_sha256'][oldpath.name] and sha(newpath)==newrep['all_outputs_sha256'][newpath.name]
with np.load(S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz') as z:commands=z['robot_qpos'].copy();weights=z['weights'].copy()
source_rows={v['frame']:v for v in json.loads((S/'e037-thumb-ray-independent-validation-v1/report.json').read_text())['source_rows'] if v['hand']=='right'};proposals={}
for row in old:
    if row['result']['classification'].startswith('unresolved'):continue
    key=(row['frame'],row['digit']);proposals[key]=[dict(origin='reused_old_28domain_witness',angles=w['angles'],source_file_sha256=sha(oldpath)) for w in row['result']['witnesses']]
for row in new:
    key=(row['frame'],row['digit']);proposals[key]=[dict(origin='new_closed_domain_witness',angles=w['angles_rad'],source_file_sha256=sha(newpath)) for w in row['result']['witnesses']]
assert len(proposals)==38
# Reuse already constructed proximal source sections, never regenerate them.
prior=S/'e037-coupled-source-face-path-v1/all-chart-sections-source-proposals-and-path-records.jsonl'
with prior.open() as f:
    for line in f:
        row=json.loads(line)
        if row['kind']!='new_common_root_other_finger_source_section' or row['selected_angle'] is None:continue
        ji=0 if row['digit']=='index' else 2;angles=[row['selected_angle'],commands[row['frame'],1,8+ji]];proposals[(row['frame'],row['digit'])].append(dict(origin='reused_prior_harmonic_source_section',angles=angles,source_file_sha256=sha(prior)))
j=(O/'all-independent-source-URDF-proposals-outcomes.jsonl').open('w');kin=URDFKinematics(c.F/'robot/urdfs/g1/dex3_right.urdf');domains=[];checks=0;rejected=0;maximum=0.;new10checks=0;verified_new10=0
for (frame,digit),pool in sorted(proposals.items()):
    ji=0 if digit=='index' else 2;source_id=8 if digit=='index' else 12;sign=1 if digit=='index' else -1;camera=c.cameras[frame];observed=c.uv((c.human[frame,1,[source_id]]-camera[:3,3])@camera[:3,:3])[0];cap=source_rows[frame]['source_before_after_px'][0][2 if digit=='index' else 3];lo,hi=c.bounds(frame);accepted=[];seen=set()
    pool.append(dict(origin='new_current_joint_guide',angles=c.q[frame,1,7+ji:9+ji],source_file_sha256=None))
    for proposal in pool:
        angles=np.array(proposal['angles']);key=tuple(np.round(angles,12))
        if key in seen:continue
        seen.add(key);cmd=commands[frame,1].copy();cmd[7+ji:9+ji]=angles;g.journal(j,dict(kind='BEFORE_independent_2R_URDF_source_point',frame=frame,digit=digit,proposal=proposal,command=cmd,not_native_or_trajectory_qualification=True));poses=kin.poses(cmd[:7],dict(zip(c.manifest['joint_names']['right'],cmd[7:],strict=True)));point=poses['right_'+digit+'_tip'][:3,3];a,b=angles;local=np.array([.0777,-.0016,sign*.0285])+np.array([.0458*np.cos(a)+.035*np.cos(a+b),.0458*np.sin(a)+.035*np.sin(a+b),0]);predicted=cmd[:3]+Rotation.from_quat(cmd[3:7]).apply(local);closure=float(abs(point-predicted).max());C=(point@c.stage-camera[:3,3])@camera[:3,:3];depth=bool(C[2]>.01);error=float(np.linalg.norm(c.uv(C[None])[0]-observed)) if depth else None;position=bool((angles>=lo[ji:ji+2]-1e-8).all() and (angles<=hi[ji:ji+2]+1e-8).all());passed=bool(position and depth and error<=cap+1e-6 and closure<1e-8);record=dict(frame=frame,digit=digit,angles_rad=angles,origin=proposal['origin'],command=cmd,URDF_tip_world=point,camera_point=C,source_error_px=error,ORIGINAL_saved_native_source_cap_px=cap,positive_depth=depth,original_position_passed=position,URDF_analytic_closure_m=closure,source_position_binding_passed=passed);g.journal(j,dict(kind='AFTER_independent_2R_URDF_source_point',record=record));checks+=1;rejected+=not passed;maximum=max(maximum,closure)
        if proposal['origin']=='new_closed_domain_witness':new10checks+=1;verified_new10+=passed
        if passed:accepted.append(record)
    domains.append(dict(frame=frame,digit=digit,accepted_source_points=accepted,proposed_point_count=len(seen),all_available_sources_and_original_tolerances_preserved=True))
j.close();(O/'verified-2R-source-point-domains.json').write_text(json.dumps(g.plain(domains),allow_nan=False,indent=2)+'\n');out=dict(schema='v2d.independent-reused-closed-2R-source-witnesses.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),input_manifest_sha256=sha(c.F/'manifest.json'),prior28domains_reused_not_resolved=True,new10domains_reused_not_resolved=True,source_domains=len(domains),new_independent_URDF_source_states=checks,rejected_actual_source_or_position_points=rejected,new10_closed_domain_points_checked=new10checks,new10_closed_domain_points_passing_actual_gates=verified_new10,maximum_URDF_analytic_tip_closure_m=maximum,source_domains_without_verified_point=[dict(frame=v['frame'],digit=v['digit']) for v in domains if not v['accepted_source_points']],all_rejected_points_retained=True,source_model_not_measured_robot_depth_or_contact=True,current_verified_reference_unchanged=True,no_native_position_or_physics=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_task_qualified=False,all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

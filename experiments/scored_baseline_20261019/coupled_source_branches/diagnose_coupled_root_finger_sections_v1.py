"""NEW continuous 2R source domains for a NEW whole-clock wrist hypothesis.

This distinguishes a failed fixed-distal section from the actual two-joint
workspace. It does not vary the wrist amplitude, rerun old conics, or certify
temporal/native feasibility of any isolated mathematical witness.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-coupled-root-full-2R-diagnosis-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
import thumb_conic_geometry_v1 as g
import planar_workspace_conic_v1 as p
from scipy.spatial.transform import Rotation
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();P=S/'e037-coupled-source-face-path-v1';report=json.loads((P/'report.json').read_text());journalpath=P/'all-chart-sections-source-proposals-and-path-records.jsonl';assert sha(journalpath)==report['all_outputs_sha256'][journalpath.name];saved=[json.loads(v) for v in journalpath.read_text().splitlines()];source_sections=[v for v in saved if v['kind']=='new_common_root_other_finger_source_section'];status={}
for v in source_sections:status.setdefault(v['frame'],{})[v['digit']]=v['selected_angle'] is not None
rootpath=S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz'
with np.load(rootpath) as z:commands=z['robot_qpos'];weights=z['weights']
source_rows={v['frame']:v for v in json.loads((S/'e037-thumb-ray-independent-validation-v1/report.json').read_text())['source_rows'] if v['hand']=='right'};j=(O/'all-new-continuous-2R-source-domains.jsonl').open('w');records=[];counts={};E=np.eye(3)[:,:2]
for frame in np.flatnonzero(weights>1e-14):
    frame=int(frame);cmd=commands[frame,1];R=Rotation.from_quat(cmd[3:7]).as_matrix();camera=c.cameras[frame];A=camera[:3,:3].T@c.stage.T;offset=camera[:3,:3].T@camera[:3,3];lo,hi=c.bounds(frame);observed=c.uv((c.human[frame,1,[8,12]]-camera[:3,3])@camera[:3,:3]);caps=source_rows[frame]['source_before_after_px'][0]
    for digit,ji,zi,si in [('index',0,1,0),('middle',2,-1,1)]:
        H=np.c_[A@R@E,A@(cmd[:3]+R@np.array([.0777,-.0016,zi*.0285]))-offset];radius=caps[2+si];Q=g.conic_from_camera_plane(H,c.K,observed[si],radius);ellipse=g.conic_ellipse(Q);box=np.c_[lo[ji:ji+2],hi[ji:ji+2]];g.journal(j,dict(kind='BEFORE_NEW_full_2R_source_domain',frame=frame,digit=digit,root_command=cmd,actual_joint_box_rad=box,original_source_radius_px=radius,H=H,Q=Q,ellipse=ellipse))
        if ellipse['kind']=='ellipse' and box[1,0]>0 and box[1,1]<np.pi:result=p.positive_2R_conic_intersection(ellipse,box[0],box[1])
        else:result=dict(classification='unresolved_non_ellipse_or_positive_branch',witnesses=[],root_records=[])
        record=dict(frame=frame,digit=digit,fixed_original_distal_section_was_possible=status[frame][digit],actual_joint_box_rad=box,result=result,conditional_new_root_mode_only=True,temporal_and_native_not_assessed=True);g.journal(j,dict(kind='AFTER_NEW_full_2R_source_domain',record=record));records.append(record);counts[result['classification']]=counts.get(result['classification'],0)+1
j.close();(O/'all-new-full-2R-source-domain-records.json').write_text(json.dumps(g.plain(records),allow_nan=False,indent=2)+'\n');section_failures=[v for v in records if not v['fixed_original_distal_section_was_possible']];rescued=[v for v in section_failures if v['result']['classification'].startswith('nonempty_')];out=dict(schema='v2d.new-root-continuous-2R-section-diagnosis.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),input_manifest_sha256=sha(c.F/'manifest.json'),new_root_mode_sha256=sha(rootpath),new_full_2R_source_domains=len(records),classifications=counts,fixed_distal_failed_sections=len(section_failures),failed_sections_with_full_2R_constructive_source_witnesses=len(rescued),rescued_frame_digits=[dict(frame=v['frame'],digit=v['digit']) for v in rescued],old_root_old_conic_families_and_source_fits_not_repeated=True,no_root_amplitude_or_prior_sweep=True,source_witnesses_not_temporal_native_or_measured_truth=True,current_reference_and_original_gates_unchanged=True,no_native_position_or_physics=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_task_qualified=False,all_outputs_sha256={q.name:sha(q) for q in O.iterdir() if q.is_file() and q.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

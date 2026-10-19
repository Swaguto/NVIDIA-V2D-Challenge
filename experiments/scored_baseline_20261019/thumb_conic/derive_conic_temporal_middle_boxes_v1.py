"""NEW necessary temporal boxes for analytic114 controls and saved neighbors.

Only middle joint angles are free in this conditional test. Saved source/self
point nodes are reused without native or source producers being repeated.
Neither a nonempty box nor a selected neighbor pair qualifies a trajectory.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time,xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1];O=S/'e037-conic-temporal-middle-boxes-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import thumb_conic_geometry_v1 as g
import source_orientation_context_v1 as c
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();bankp=S/'e037-coupled-source-disk-pose-bank-v2/pose-bank.json';bankrep=json.loads((bankp.parent/'report.json').read_text());assert sha(bankp)==bankrep['pose_bank_sha256'];bank=json.loads(bankp.read_text());rootp=S/'e037-root-twist-interval-study-v1/accepted-finite-witnesses.json';rootrep=json.loads((rootp.parent/'report.json').read_text());assert sha(rootp)==rootrep['accepted_witnesses_sha256'];extra=json.loads(rootp.read_text());newp=S/'e037-thumb-conic-independent-v1/independent-isolated-witnesses.json';newrep=json.loads((newp.parent/'report.json').read_text());assert sha(newp)==newrep['all_outputs_sha256'][newp.name];new=json.loads(newp.read_text());neighbors={f:[] for f in [113,115]}
for p in bank+extra:
 if p['frame'] not in neighbors:continue
 cmd=np.array(p['command']);k=tuple(np.round(cmd,12))
 if not any(v['key']==k for v in neighbors[p['frame']]):neighbors[p['frame']].append(dict(key=k,command=cmd,record=p))
lo,hi=c.bounds(114);names=c.manifest['joint_names']['right'];lim={j.get('name'):j.find('limit').attrib for j in ET.parse(c.F/'robot/urdfs/g1/dex3_right.urdf').getroot().findall('joint') if j.get('type')!='fixed'};vel=np.array([float(lim[n]['velocity']) for n in names]);dt=[c.ts[114]-c.ts[113],c.ts[115]-c.ts[114]];aligned=c.aligned[:,1,7:];middle=np.array([2,3]);fixed=np.array([0,1,4,5,6]);journal=(O/'all-neighbor-pair-boxes.jsonl').open('w');eligible=[];all_count=0;failures=dict(empty_full_box=0,fixed_new_control_outside=0,eligible_middle_box=0);position_tolerance=1e-8;motion_tolerance=1e-7
for ni,n in enumerate(new):
 cmd=np.array(n['command']);delta=cmd[7:]-aligned[114]
 for ai,a in enumerate(neighbors[113]):
  for bi,b in enumerate(neighbors[115]):
   qa=a['command'][7:];qb=b['command'][7:];da=qa-aligned[113];db=qb-aligned[115];lower=np.maximum.reduce([lo-aligned[114],da-.05,db-.05,(da+db-.05)/2,qa-vel*dt[0]-aligned[114],qb-vel*dt[1]-aligned[114]]);upper=np.minimum.reduce([hi-aligned[114],da+.05,db+.05,(da+db+.05)/2,qa+vel*dt[0]-aligned[114],qb+vel*dt[1]-aligned[114]]);empty=np.flatnonzero(lower>upper+motion_tolerance);fixedbad=fixed[(delta[fixed]<lower[fixed]-motion_tolerance)|(delta[fixed]>upper[fixed]+motion_tolerance)];valid=bool(not len(empty) and not len(fixedbad));record=dict(new_control_index=ni,source_queue_index=n['source_queue_index'],neighbor113_index=ai,neighbor115_index=bi,lower_original_correction_rad=lower,upper_original_correction_rad=upper,empty_box_joint_indices=empty,fixed_control_outside_joint_indices=fixedbad,necessary_middle_box_eligible=valid)
   if valid:
    record['middle_joint_angle_box_rad']=np.c_[aligned[114,middle]+lower[middle],aligned[114,middle]+upper[middle]];record['original_middle_closest_box_point_rad']=np.clip(c.q[114,1,7+middle],record['middle_joint_angle_box_rad'][:,0],record['middle_joint_angle_box_rad'][:,1]);record['original_middle_distance_to_box_rad']=float(np.linalg.norm(record['original_middle_closest_box_point_rad']-c.q[114,1,7+middle]));eligible.append(record);failures['eligible_middle_box']+=1
   elif len(empty):failures['empty_full_box']+=1
   else:failures['fixed_new_control_outside']+=1
   g.journal(journal,record);all_count+=1
journal.close();selected=None
if eligible:
 best=min(eligible,key=lambda r:(r['original_middle_distance_to_box_rad'],r['new_control_index'],r['neighbor113_index'],r['neighbor115_index']));selected=dict(box_record=best,new114_control=new[best['new_control_index']],neighbor113=neighbors[113][best['neighbor113_index']]['record'],neighbor115=neighbors[115][best['neighbor115_index']]['record'],selection='minimum change of ORIGINAL middle command to necessary box, then deterministic IDs',conditional_robot_neighbors_not_measured_source_poses=True,only_two_step_finger_necessity_no_root_source_native_trajectory_certificate=True)
for name,value in [('all-eligible-middle-boxes.json',eligible),('selected-conditional-middle-box.json',selected)]:
 (O/name).write_text(json.dumps(g.plain(value),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.conic-temporal-middle-boxes.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),input_manifest_sha256=sha(c.F/'manifest.json'),saved_node_inputs_sha256={str(p):sha(p) for p in [bankp,rootp,newp]},new114_controls=len(new),saved_neighbor113_nodes=len(neighbors[113]),saved_neighbor115_nodes=len(neighbors[115]),all_pairs_examined=all_count,pair_outcomes=failures,selected_box_exists=selected is not None,original_added_adjacent_and_second_cap_rad=.05,original_actual_URDF_velocity=vel,original_numeric_motion_tolerance=motion_tolerance,position_tolerance=position_tolerance,all_joint_boxes_and_rejection_causes_durable=True,no_saved_source_or_native_nodes_rechecked=True,no_completed_graph_bank_or_motion_LP_repeated=True,conditional_necessary_not_global_or_physical_success=True,current_combined_freeze_unchanged=True,new_GPU_jobs=0,new_PPO_steps=0,no_native_or_physics_steps=True,whole_reference_or_task_qualified=False,all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out),indent=2))

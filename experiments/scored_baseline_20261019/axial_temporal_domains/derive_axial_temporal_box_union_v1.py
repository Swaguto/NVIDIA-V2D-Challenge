"""NEW correlated temporal-box union and local root necessity for113:115.

Saved full seven-joint boxes are not recomputed or convexified into one box.
Neighbors remain finite robot controls; all root rejections stay explicit.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');O=S/'e037-axial-temporal-box-union-v1';start=time.monotonic();sys.path[:0]=[str(S/'compute'),str(R/'scripts/track3')]
import thumb_conic_geometry_v1 as g
import source_orientation_context_v1 as c
from scipy.spatial.transform import Rotation
from temporal_pose_path import motion_summary
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();P=S/'e037-conic-temporal-middle-boxes-v1';rep=json.loads((P/'report.json').read_text());records_path=P/'all-neighbor-pair-boxes.jsonl';assert sha(records_path)==rep['all_outputs_sha256'][records_path.name];records=[json.loads(line) for line in records_path.read_text().splitlines()];records=[r for r in records if r['new_control_index']==0];assert len(records)==500
bankp=S/'e037-coupled-source-disk-pose-bank-v2/pose-bank.json';rootp=S/'e037-root-twist-interval-study-v1/accepted-finite-witnesses.json';bank=json.loads(bankp.read_text());extra=json.loads(rootp.read_text());neighbors={f:[] for f in [113,115]}
for p in bank+extra:
 if p['frame'] not in neighbors:continue
 cmd=np.array(p['command']);key=tuple(np.round(cmd,12))
 if not any(v['key']==key for v in neighbors[p['frame']]):neighbors[p['frame']].append(dict(key=key,command=cmd,record=p))
assert [len(neighbors[f]) for f in [113,115]]==[4,125];newpath=S/'e037-thumb-conic-independent-v1/selected-conditional-robot-control.json';control=json.loads(newpath.read_text());base=np.array(control['command']);aligned=c.aligned[114,1,7:];original=c.q[114,1,7:];caps=json.loads((S/'e037-wrist-source-authority-v1/report.json').read_text())['original_root_motion_caps'];max_keys=[k for k in caps if k.startswith('maximum_')];journal=(O/'all-correlated-box-and-root-records.jsonl').open('w');eligible=[];counts=dict(empty_saved_joint_box=0,unresolved_numerical_box=0,local_root_rejection=0,conditional_correlated_box=0)
for row in records:
 ai=row['neighbor113_index'];bi=row['neighbor115_index'];lower=np.array(row['lower_original_correction_rad']);upper=np.array(row['upper_original_correction_rad']);entry=dict(neighbor113_index=ai,neighbor115_index=bi,original_correction_box_rad=np.c_[lower,upper],joint_order=c.manifest['joint_names']['right'])
 if len(row['empty_box_joint_indices']):entry['classification']='empty_saved_joint_box';counts[entry['classification']]+=1;g.journal(journal,entry);continue
 if (lower>upper).any():entry['classification']='unresolved_numerical_box';counts[entry['classification']]+=1;g.journal(journal,entry);continue
 commands=np.array([neighbors[113][ai]['command'],base,neighbors[115][bi]['command']]);poses=np.tile(np.eye(4),(3,1,1));poses[:,:3,3]=commands[:,:3];poses[:,:3,:3]=Rotation.from_quat(commands[:,3:7]).as_matrix();root_motion=motion_summary(poses,c.ts[[113,114,115]]);bad=[k for k in max_keys if root_motion[k]>caps[k]+1e-7];entry.update(local_root_motion=root_motion,local_maximum_root_caps={k:caps[k] for k in max_keys},local_root_bad_keys=bad,global_p95_and_outside_boundaries_not_assessed=True)
 if bad:entry['classification']='local_root_rejection';counts[entry['classification']]+=1;g.journal(journal,entry);continue
 absolute=np.c_[aligned+lower,aligned+upper];closest=np.clip(original,absolute[:,0],absolute[:,1]);entry.update(classification='conditional_correlated_box',joint_angle_box_rad=absolute,closest_original_joint_point_rad=closest,original_joint_distance_to_box_rad=float(np.linalg.norm(closest-original)),not_a_source_native_pose_or_whole_trajectory=True);counts[entry['classification']]+=1;eligible.append(entry);g.journal(journal,entry)
journal.close();selected=None
if eligible:
 best=min(eligible,key=lambda e:(e['original_joint_distance_to_box_rad'],e['neighbor113_index'],e['neighbor115_index']));selected=dict(correlated_box=best,new114_root_control=control,neighbor113=neighbors[113][best['neighbor113_index']]['record'],neighbor115=neighbors[115][best['neighbor115_index']]['record'],selection='closest ORIGINAL seven-joint command to one retained correlated box, then deterministic neighbor IDs',source_pose_not_selected_as_truth_by_physical_outcome=True,conditional_robot_source_model_hypothesis=True,all_old_control_failures_retained=True)
for name,value in [('all-eligible-correlated-boxes.json',eligible),('selected-correlated-box.json',selected)]:
 (O/name).write_text(json.dumps(g.plain(value),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.axial-temporal-box-union.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),input_manifest_sha256=sha(c.F/'manifest.json'),saved_box_sha256=sha(records_path),source_node_inputs_sha256={str(p):sha(p) for p in [bankp,rootp,newpath]},unique_saved_neighbor_pairs=500,classifications=counts,correlated_boxes_not_replaced_by_coordinate_hull=True,selected_exists=selected is not None,selected_joint_angle_box_rad=None if selected is None else selected['correlated_box']['joint_angle_box_rad'],selected_closest_original_point_rad=None if selected is None else selected['correlated_box']['closest_original_joint_point_rad'],local_root_max_checks_necessary_not_global_motion_certificate=True,original_authority_bounds_and_reference_unchanged=True,no_old_box_LP_graph_bank_native_or_source_study_repeated=True,new_GPU_jobs=0,new_PPO_steps=0,no_native_or_physics_steps=True,whole_reference_or_task_qualified=False,all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out),indent=2))

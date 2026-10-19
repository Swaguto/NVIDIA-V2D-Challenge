"""NEW integration of TWO independently checked source point corpora.

At this fixed wrist hypothesis the two digits' image/position/temporal gates
separate. All saved points are retained per digit; no Cartesian top24 pruning,
new samples, source solve, graph-family rerun or native work occurs.
"""
from pathlib import Path
import dataclasses,datetime,hashlib,itertools,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-integrated-source-corpora-reachability-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
import thumb_conic_geometry_v1 as g
from v2d_dex3_flashchord.temporal_pose_graph import finite_temporal_reachability
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();P=S/'e037-2R-workspace-source-continuation-v1';path=P/'all-new-workspace-source-domains.json';assert sha(path)==json.loads((P/'report.json').read_text())['all_outputs_sha256'][path.name];domains=json.loads(path.read_text());lookup={(v['frame'],v['digit']):v['source_points'] for v in domains}
with np.load(S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz') as z:commands=z['robot_qpos'].copy();weights=z['weights'].copy()
j=(O/'all-factorized-saved-banks-and-recurrence-returns.jsonl').open('w');results={};banks_by_digit={};controls=[]
prior_controls_path=S/'e037-factorized-saved-source-reachability-v1/report.json';prior_controls=json.loads(prior_controls_path.read_text());controls=prior_controls['new_analytic_factorization_contracts'];assert len(controls)==2 and all(v['passed'] for v in controls);g.journal(j,dict(kind='reuse_factorization_controls_without_reexecution',controls_sha256=sha(prior_controls_path),controls=controls))
levelpath=S/'e037-continuous-2R-source-projection-v1/all-new-continuous-source-branches.json';levelreport=json.loads((levelpath.parent/'report.json').read_text());assert sha(levelpath)==levelreport['all_outputs_sha256'][levelpath.name];levelrows=json.loads(levelpath.read_text());added=0
for row in levelrows:
    if not row.get('source_position_passed'):continue
    frame=row['frame'];digit=row['digit'];cmd=np.array(row['command']);assert np.array_equal(cmd[:7],commands[frame,1,:7]);angles=np.array(row['result']['angles']);ji=0 if digit=='index' else 2;assert np.max(abs(angles-cmd[7+ji:9+ji]))<1e-12;pool=lookup[(frame,digit)]
    if any(np.max(abs(angles-np.array(v['angles_rad'])))<1e-12 for v in pool):continue
    pool.append(dict(frame=frame,digit=digit,angles_rad=angles,source_error_px=row['actual_URDF_source_error_px'],source_position_binding_passed=True,origin='independently_passed_scalar_source_root_NOT_failed_whole_path_promotion'));added+=1;g.journal(j,dict(kind='integrate_verified_point_from_distinct_source_constructor',frame=frame,digit=digit,angles_rad=angles,source_contract_sha256=sha(levelpath),whole_level_projection_path_remains_REJECTED=True))

for digit,ji in [('index',0),('middle',2)]:
    banks=[];costs=[]
    for frame in range(len(c.q)):
        pool=[commands[frame,1,7+ji:9+ji].copy()] if weights[frame]<=1e-14 else [np.array(v['angles_rad']) for v in lookup[(frame,digit)]];unique={tuple(np.round(v,12)):v for v in pool};bank=np.array(list(unique.values())).reshape(-1,2);banks.append(bank);costs.append(np.sum((bank-c.aligned[frame,1,7+ji:9+ji])**2,axis=1));g.journal(j,dict(kind='reused_all_source_points_without_Cartesian_pruning',frame=frame,digit=digit,bank=bank,point_count=len(bank),no_new_source_geometry_or_native_execution=True))
    result=finite_temporal_reachability(banks,c.aligned[:,1,7+ji:9+ji],c.ts,c.limits[ji:ji+2,0],c.limits[ji:ji+2,1],np.ones(2)*12,component_cap=.25,adjacent_added_cap=.05,second_added_cap=.05,node_costs=costs,tolerance=1e-7);record=dataclasses.asdict(result);g.journal(j,dict(kind='NEW_exact_finite_factor_reachability_return',digit=digit,record=record));results[digit]=record;banks_by_digit[digit]=banks
selected=None;motion=None
if all(v['path_indices'] is not None for v in results.values()):
    selected=commands.copy()
    for digit,ji in [('index',0),('middle',2)]:
        for frame,index in enumerate(results[digit]['path_indices']):selected[frame,1,7+ji:9+ji]=banks_by_digit[digit][frame][index]
    delta=selected[:,1,7:11]-c.aligned[:,1,7:11];speed=abs(np.diff(selected[:,1,7:11],axis=0))/np.diff(c.ts)[:,None];motion=dict(component_rad=float(abs(delta).max()),added_adjacent_rad=float(abs(np.diff(delta,axis=0)).max()),added_second_rad=float(abs(np.diff(delta,n=2,axis=0)).max()),maximum_URDF_velocity_rad_s=float(speed.max()));g.journal(j,dict(kind='direct_combined_four_joint_original_motion',record=motion));assert motion['component_rad']<=.25+1e-7 and motion['added_adjacent_rad']<=.05+1e-7 and motion['added_second_rad']<=.05+1e-7 and motion['maximum_URDF_velocity_rad_s']<=12+1e-7;np.savez_compressed(O/'UNQUALIFIED-factorized-four-joint-source-component.npz',robot_qpos=selected)
j.close();(O/'all-factorized-reachability-returns.json').write_text(json.dumps(g.plain(results),allow_nan=False,indent=2)+'\n');out=dict(schema='v2d.integrated-independent-source-corpora-reachability.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),input_manifest_sha256=sha(c.F/'manifest.json'),all_saved_source_points_sha256=sha(path),new_analytic_factorization_contracts=controls,index_path_exists=results['index']['path_indices'] is not None,middle_path_exists=results['middle']['path_indices'] is not None,new_combined_four_joint_source_component_exists=selected is not None,direct_combined_motion=motion,index_first_zero_forward_pair=next((i for i,v in enumerate(results['index']['forward_pair_counts']) if v==0),None),middle_first_zero_forward_pair=next((i for i,v in enumerate(results['middle']['forward_pair_counts']) if v==0),None),all_saved_nodes_retained_per_digit=True,new_distinct_verified_source_points_integrated=added,source_level_corpus_sha256=sha(levelpath),failed_whole_projection_path_not_promoted=True,no_new_samples_source_solves_URDF_or_native_positions=True,factorization_covers_only_kinematic_source_and_original_motion_constraints=True,interfinger_native_motor_grasp_and_whole_reference_unqualified=True,current_verified_reference_unchanged=True,no_physics=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_task_qualified=False,all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

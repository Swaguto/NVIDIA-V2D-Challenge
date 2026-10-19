"""NEW independent saved-partition cover, inner-box and URDF/Taylor audits.

No source chart, MILP, branch LP or earlier verification job is repeated.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-index-continuous-domain-independent-v1'
sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
import thumb_conic_geometry_v1 as g
from v2d_dex3_flashchord.kinematics import URDFKinematics
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
P=S/'e037-index-source-temporal-cone-branch-v1';path=P/'all-continuous-domain-proposals-LP-returns-and-source-outcomes.jsonl'
assert sha(path)==json.loads((P/'report.json').read_text())['all_outputs_sha256'][path.name]
before={};splits=[];saved_planes={};returns={};candidate_summary=[]
for line in path.open():
    row=json.loads(line);kind=row['kind']
    if kind=='BEFORE_NEW_continuous_source_temporal_domain':before[row['node']['id']]=row['node']
    elif kind=='NEW_domain_binary_split':splits.append(row)
    elif kind=='BEFORE_NEW_source_temporal_LP' and not row['inner']:saved_planes[row['node_id']]=row['all_source_Taylor_witnesses']
    elif kind=='AFTER_NEW_source_temporal_LP_RETURN' and not row['inner']:returns[row['node_id']]=row['returned']
    elif kind=='AFTER_NEW_source_temporal_analytic_candidate':
        a=row['assessment'];bad=[v for v in a['source'] if not v['passed']]
        candidate_summary.append(dict(node_id=row['node_id'],inner=row['inner'],failed_source_frames=[v['frame'] for v in bad],
            maximum_source_regression_px=max(v['error_px']-v['original_cap_px'] for v in a['source']),motion=a['motion']))
cover=json.loads((P/'processed-and-unresolved-domains.json').read_text());leaves={v['id']:v for v in cover['unresolved']}
for s in splits:
    for child in s['children']:
        if child['id'] not in before:assert child['id'] in leaves
        else:assert np.array_equal(child['box'],before[child['id']]['box'])
    parent=np.array(before[s['node_id']]['box']);left,right=[np.array(v['box']) for v in s['children']];f,d=s['frame'],s['dimension'];mid=s['midpoint']
    expectedleft=parent.copy();expectedright=parent.copy();expectedleft[f,d,1]=mid;expectedright[f,d,0]=mid
    assert np.array_equal(left,expectedleft) and np.array_equal(right,expectedright)
assert len(before)==len(cover['processed'])==255
terminal=[v for v in cover['processed'] if v['outcome']!='split_full_continuous_joint_domain']
assert 1+2*len(splits)==len(before)+len(leaves) and len(terminal)+len(leaves)==len(splits)+1
j=(O/'all-independent-cover-URDF-and-enclosure-records.jsonl').open('w')
g.journal(j,dict(kind='independent_binary_cover_counts',processed=len(before),splits=len(splits),terminal=len(terminal),unresolved=len(leaves),complete_binary_cover=True))
with np.load(S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz') as z:commands=z['robot_qpos'].copy()
kin=URDFKinematics(c.F/'robot/urdfs/g1/dex3_right.urdf');rng=np.random.default_rng(191092);checks=0;maxclosure=0.;maxexcess=-1e9
node_ids=np.linspace(0,len(before)-1,8,dtype=int)
for node_id in node_ids:
    node=before[int(node_id)];box=np.array(node['box']);planes=saved_planes[int(node_id)]
    for k in range(32):
        frame=103+k%19;angles=rng.uniform(box[frame,:,0],box[frame,:,1]);cmd=commands[frame,1].copy();cmd[7:9]=angles
        g.journal(j,dict(kind='BEFORE_independent_new_URDF_Taylor_sample',node_id=int(node_id),frame=frame,command=cmd))
        tip=kin.poses(cmd[:7],dict(zip(c.manifest['joint_names']['right'],cmd[7:],strict=True)))['right_index_tip'][:3,3]
        camera=c.cameras[frame];C=(tip@c.stage-camera[:3,3])@camera[:3,:3]
        rows=[v for v in planes if v['frame']==frame];residuals=[]
        for row in rows:
            linear=row['affine_constant']+np.array(row['derivative'])@angles
            residual=float(np.array(row['normal'])@C-linear);excess=abs(residual)-row['remainder'];maxexcess=max(maxexcess,excess)
            residuals.append(dict(kind=row['kind'],residual=residual,remainder=row['remainder'],within_saved_bound=bool(excess<=1e-8)))
        # The independent URDF sample checks the saved actual camera-plane
        # remainder directly, without recomputing the old bound or source fit.
        g.journal(j,dict(kind='AFTER_independent_new_URDF_Taylor_sample',node_id=int(node_id),frame=frame,angles=angles,actual_camera_point=C,residuals=residuals))
        assert all(v['within_saved_bound'] for v in residuals);checks+=1

B=S/'e037-index-continuous-source-box-path-v1';rpath=B/'all-continuous-source-box-records.json'
assert sha(rpath)==json.loads((B/'report.json').read_text())['all_outputs_sha256'][rpath.name]
innerchecks=0;maxerr=-1e9
for frame_row in json.loads(rpath.read_text()):
    frame=frame_row['frame']
    pool=[v for v in frame_row['proposals'] if v.get('certificate',{}).get('classification')=='continuous_source_certified_inner_box']
    if not pool:continue
    record=max(pool,key=lambda v:v['certificate']['construction_radius_rad']);box=np.array(record['box']);H=np.array(record['H'])
    for k in range(8):
        angles=rng.uniform(box[:,0],box[:,1]);cmd=commands[frame,1].copy();cmd[7:9]=angles
        g.journal(j,dict(kind='BEFORE_independent_new_URDF_source_inner_box_sample',frame=frame,box=box,command=cmd))
        tip=kin.poses(cmd[:7],dict(zip(c.manifest['joint_names']['right'],cmd[7:],strict=True)))['right_index_tip'][:3,3]
        camera=c.cameras[frame];C=(tip@c.stage-camera[:3,3])@camera[:3,:3];a,b=angles
        xy=np.array([.0458*np.cos(a)+.035*np.cos(a+b),.0458*np.sin(a)+.035*np.sin(a+b)])
        closure=float(abs(C-H@np.r_[xy,1.]).max());maxclosure=max(maxclosure,closure)
        error=float(np.linalg.norm(c.uv(C[None])[0]-record['pixel']));excess=error-record['ORIGINAL_radius_px'];maxerr=max(maxerr,excess)
        passed=bool(error<=record['ORIGINAL_radius_px']+1e-6 and closure<1e-8)
        g.journal(j,dict(kind='AFTER_independent_new_URDF_source_inner_box_sample',frame=frame,angles=angles,camera_point=C,source_error_px=error,original_radius_px=record['ORIGINAL_radius_px'],closure_m=closure,passed=passed));assert passed;innerchecks+=1
j.close()
(O/'saved-branch-candidate-failure-summary.json').write_text(json.dumps(g.plain(candidate_summary),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.independent-continuous-index-domain-records.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),
    full_binary_partition_cover_verified=True,processed_nodes=len(before),binary_splits=len(splits),necessary_LP_empty_leaves=len(terminal),remaining_unresolved_leaves=len(leaves),
    new_actual_URDF_Taylor_samples=checks,new_actual_URDF_source_inner_box_samples=innerchecks,
    maximum_actual_URDF_analytic_camera_closure_m=maxclosure,maximum_saved_remainder_excess_m=maxexcess,
    maximum_inner_box_sample_source_excess_px=maxerr,sampled_Taylor_and_inner_source_checks_passed=True,
    original_branch_LP_returns_not_recomputed=True,old_source_and_native_jobs_not_repeated=True,
    no_saved_branch_candidate_passes_all_source_rows=all(v['failed_source_frames'] for v in candidate_summary),
    numerical_empty_LP_status_not_formal_dual_certificate=True,current_verified_reference_and_source_authority_unchanged=True,
    new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_task_qualified=False,
    all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

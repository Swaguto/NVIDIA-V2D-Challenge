"""NEW convex source-cone workspace homotopies and four-joint temporal path.

Interpolated ROBOT workspace points are inverted through actual 2R kinematics;
old joint poses are not interpolated or mandated. Source cones are convex at
fixed root; the robot workspace need not be, so every inverse is checked.
"""
from pathlib import Path
import dataclasses,datetime,hashlib,itertools,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-2R-workspace-source-continuation-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
import thumb_conic_geometry_v1 as g
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.kinematics import URDFKinematics
from v2d_dex3_flashchord.temporal_pose_graph import finite_temporal_reachability
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();j=(O/'all-workspace-homotopy-proposals-and-temporal-records.jsonl').open('w');controls=[];l1,l2=.0458,.035
def fk(a,b):return np.array([l1*np.cos(a)+l2*np.cos(a+b),l1*np.sin(a)+l2*np.sin(a+b)])
def inverse(p,box):
    cosine=(p@p-l1*l1-l2*l2)/(2*l1*l2)
    if abs(cosine)>1+1e-12:return []
    b=float(np.arccos(np.clip(cosine,-1,1)));a=float(np.arctan2(p[1],p[0])-np.arctan2(l2*np.sin(b),l1+l2*np.cos(b)));out=[]
    for k in [-2*np.pi,0.,2*np.pi]:
        v=np.array([a+k,b])
        if (v>=box[:,0]-1e-8).all() and (v<=box[:,1]+1e-8).all():out.append(v)
    return out
point=.5*(fk(.1,.5)+fk(.2,.6));branches=inverse(point,np.array([[0.,.3],[0.,.9]]));ok=bool(branches) and max(abs(fk(*branches[0])-point))<1e-14;controls.append(dict(name='workspace_homotopy_has_checked_nonlinear_inverse',passed=ok));assert ok
point=.5*(fk(-1.3,.2)+fk(1.3,.2));branches=inverse(point,np.array([[-1.4,1.4],[.1,.3]]));ok=not branches;controls.append(dict(name='convex_workspace_blend_outside_robot_domain_rejected',passed=ok));assert ok
C1=np.array([.05,0.,1.]);C2=np.array([0.,.15,3.]);C=.4*C1+.6*C2;ok=np.linalg.norm(C[:2])<=.1*C[2];controls.append(dict(name='positive_depth_camera_source_cone_is_convex',passed=bool(ok)));assert ok
for v in controls:g.journal(j,dict(kind='new_workspace_source_homotopy_analytic_contract',record=v))
P=S/'e037-closed-2R-independent-witnesses-v1';path=P/'verified-2R-source-point-domains.json';assert sha(path)==json.loads((P/'report.json').read_text())['all_outputs_sha256'][path.name];domains=json.loads(path.read_text());kin=URDFKinematics(c.F/'robot/urdfs/g1/dex3_right.urdf')
with np.load(S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz') as z:commands=z['robot_qpos'].copy();weights=z['weights'].copy()
source_rows={v['frame']:v for v in json.loads((S/'e037-thumb-ray-independent-validation-v1/report.json').read_text())['source_rows'] if v['hand']=='right'};output=[];checks=0;rejected_inverse=0;rejected_actual=0;maxclosure=0.
for domain in domains:
    frame=domain['frame'];digit=domain['digit'];ji=0 if digit=='index' else 2;sign=1 if digit=='index' else -1;seed=domain['accepted_source_points'];assert seed;lo,hi=c.bounds(frame);box=np.c_[lo[ji:ji+2],hi[ji:ji+2]];seed=sorted(seed,key=lambda r:(float(np.sum((np.array(r['angles_rad'])-c.aligned[frame,1,7+ji:9+ji])**2)),tuple(r['angles_rad'])));anchor=seed[0];anchorxy=fk(*anchor['angles_rad']);pool=list(seed);camera=c.cameras[frame];observed=c.uv((c.human[frame,1,[8 if digit=='index' else 12]]-camera[:3,3])@camera[:3,:3])[0];cap=source_rows[frame]['source_before_after_px'][0][2 if digit=='index' else 3];seen={tuple(np.round(v['angles_rad'],12)) for v in pool}
    for endpoint_index,endpoint in enumerate(seed[1:]):
        endpointxy=fk(*endpoint['angles_rad'])
        for fraction in [.25,.5,.75]:
            point=(1-fraction)*anchorxy+fraction*endpointxy;branches=inverse(point,box);g.journal(j,dict(kind='workspace_source_cone_homotopy_inverse',frame=frame,digit=digit,anchor_angles=anchor['angles_rad'],endpoint_angles=endpoint['angles_rad'],fraction=fraction,workspace_point=point,actual_joint_box=box,inverse_branches=branches,not_joint_pose_or_observed_depth_interpolation=True));rejected_inverse+=not bool(branches)
            for angles in branches:
                key=tuple(np.round(angles,12))
                if key in seen:continue
                seen.add(key);cmd=commands[frame,1].copy();cmd[7+ji:9+ji]=angles;g.journal(j,dict(kind='BEFORE_new_URDF_workspace_homotopy_point',frame=frame,digit=digit,command=cmd,workspace_point=point));poses=kin.poses(cmd[:7],dict(zip(c.manifest['joint_names']['right'],cmd[7:],strict=True)));world=poses['right_'+digit+'_tip'][:3,3];expected=cmd[:3]+Rotation.from_quat(cmd[3:7]).apply(np.array([.0777,-.0016,sign*.0285])+np.r_[point,0.]);closure=float(abs(expected-world).max());C=(world@c.stage-camera[:3,3])@camera[:3,:3];error=float(np.linalg.norm(c.uv(C[None])[0]-observed)) if C[2]>.01 else None;passed=bool(error is not None and error<=cap+1e-6 and closure<1e-8);row=dict(frame=frame,digit=digit,angles_rad=angles,command=cmd,workspace_homotopy_point=point,source_error_px=error,ORIGINAL_source_cap_px=cap,URDF_workspace_closure_m=closure,source_position_binding_passed=passed);g.journal(j,dict(kind='AFTER_new_URDF_workspace_homotopy_point',record=row));checks+=1;rejected_actual+=not passed;maxclosure=max(maxclosure,closure)
                if passed:pool.append(row)
    output.append(dict(frame=frame,digit=digit,source_points=pool,continuous_workspace_homotopies_are_finitely_discretized=True))
lookup={(v['frame'],v['digit']):v['source_points'] for v in output};banks=[];costs=[];node_commands=[];cap_per_clock=24
for frame in range(len(c.q)):
    if weights[frame]<=1e-14:pool=[commands[frame,1].copy()]
    else:
        pairs=itertools.product(lookup[(frame,'index')],lookup[(frame,'middle')]);pool=[]
        for a,b in pairs:
            cmd=commands[frame,1].copy();cmd[7:9]=a['angles_rad'];cmd[9:11]=b['angles_rad'];pool.append(cmd)
        pool=sorted(pool,key=lambda cmd:(float(np.sum((cmd[7:11]-c.aligned[frame,1,7:11])**2)),tuple(cmd[7:11])))[:cap_per_clock]
    node_commands.append(pool);bank=np.array([cmd[7:11] for cmd in pool]).reshape(-1,4);banks.append(bank);costs.append(np.sum((bank-c.aligned[frame,1,7:11])**2,axis=1));g.journal(j,dict(kind='frozen_new_four_joint_source_bank',frame=frame,commands=pool,finite_node_cap=cap_per_clock))
graph=finite_temporal_reachability(banks,c.aligned[:,1,7:11],c.ts,c.limits[:4,0],c.limits[:4,1],np.ones(4)*12,component_cap=.25,adjacent_added_cap=.05,second_added_cap=.05,node_costs=costs,tolerance=1e-7);grow=dataclasses.asdict(graph);g.journal(j,dict(kind='new_four_joint_source_temporal_reachability_return',record=grow));selected=None;motion=None
if graph.path_indices is not None:
    selected=commands.copy()
    for frame,index in enumerate(graph.path_indices):selected[frame,1,7:11]=node_commands[frame][index][7:11]
    delta=selected[:,1,7:11]-c.aligned[:,1,7:11];speed=abs(np.diff(selected[:,1,7:11],axis=0))/np.diff(c.ts)[:,None];motion=dict(component_rad=float(abs(delta).max()),added_adjacent_rad=float(abs(np.diff(delta,axis=0)).max()),added_second_rad=float(abs(np.diff(delta,n=2,axis=0)).max()),maximum_URDF_velocity_rad_s=float(speed.max()));g.journal(j,dict(kind='direct_four_joint_original_motion_checks',record=motion));assert motion['component_rad']<=.25+1e-7 and motion['added_adjacent_rad']<=.05+1e-7 and motion['added_second_rad']<=.05+1e-7 and motion['maximum_URDF_velocity_rad_s']<=12+1e-7;np.savez_compressed(O/'UNQUALIFIED-four-joint-source-continuation.npz',robot_qpos=selected)
j.close();(O/'all-new-workspace-source-domains.json').write_text(json.dumps(g.plain(output),allow_nan=False,indent=2)+'\n');(O/'four-joint-source-temporal-return.json').write_text(json.dumps(g.plain(grow),allow_nan=False,indent=2)+'\n');out=dict(schema='v2d.four-joint-workspace-source-continuation.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),input_manifest_sha256=sha(c.F/'manifest.json'),source_domain_input_sha256=sha(path),new_analytic_contracts=controls,new_independent_URDF_homotopy_states=checks,rejected_workspace_inverse_proposals=rejected_inverse,rejected_actual_source_points=rejected_actual,maximum_URDF_workspace_closure_m=maxclosure,new_four_joint_temporal_path_exists=graph.path_indices is not None,direct_four_joint_original_motion=motion,finite_four_joint_node_cap=cap_per_clock,finite_scope_not_global_unreachability=True,thumb_source_native_self_motor_grasp_and_whole_reference_unqualified=True,current_verified_three_component_freeze_unchanged=True,no_native_position_or_physics=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_task_qualified=False,all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

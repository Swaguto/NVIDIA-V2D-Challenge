"""NEW common-root/source/compiled-face branch and ORIGINAL temporal path.

Axial sampling is finite; proximal clearance and distal/source domains remain
continuous within each section. No native call occurs without a whole-clock
source/URDF/original-motion candidate. No old bank or graph is rerun.
"""
from pathlib import Path
import dataclasses,datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-coupled-source-face-path-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
import thumb_conic_geometry_v1 as g
import harmonic_source_domains_v1 as h
import compiled_thumb_face_chart_v1 as f
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.kinematics import URDFKinematics
from v2d_dex3_flashchord.temporal_pose_graph import finite_temporal_reachability
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();j=(O/'all-chart-sections-source-proposals-and-path-records.jsonl').open('w');controls=[]
for name,coef,bounds,expect in [('cosine_positive_arc',[[1,0,-.5]],[-1.5,1.5],2*np.pi/3),('empty_negative_constant',[[0,0,-1]],[-.5,.5],0.),('unrestricted_positive_constant',[[0,0,1]],[-.5,.5],1.)]:
    d=f.sinusoid_nonnegative_intervals(coef,bounds);length=sum(v['interval_rad'][1]-v['interval_rad'][0] for v in d['intervals'] if v['inside_sufficient_plane']);passed=abs(length-expect)<1e-12;g.journal(j,dict(kind='new_compiled_face_angle_contract',name=name,domain=d,passed=passed));controls.append(dict(name=name,passed=passed));assert passed
P=S/'e037-strict-source-wrist-mode-v1';report=json.loads((P/'report.json').read_text());assert report['original_source_model_root_component_and_full_clock_motion_passed'];path=P/'UNQUALIFIED-strict-source-root-proposal.npz';assert sha(path)==report['all_outputs_sha256'][path.name]
with np.load(path) as z:rootcommands=z['robot_qpos'].copy();weights=z['weights'].copy()
G=S/'e037-thumb-conic-clearance-v1';geometrypath=G/'compiled-parent-geometry.json';assert sha(geometrypath)==json.loads((G/'report.json').read_text())['all_outputs_sha256'][geometrypath.name];geometry=json.loads(geometrypath.read_text());olda=geometry['fixed_axial_thumb_rad'];Ryold=Rotation.from_rotvec([0,olda,0]).as_matrix();palm=np.array(geometry['palm_vertices_in_thumb0_frame'])@Ryold.T;thumb=np.array(geometry['thumb1_vertices_in_own_frame']);origin=np.array(geometry['thumb1_joint_origin']);normals=np.unique(np.round(ConvexHull(palm).equations[:,:3],12),axis=0);normals/=np.linalg.norm(normals,axis=1)[:,None];kin=URDFKinematics(c.F/'robot/urdfs/g1/dex3_right.urdf');baseline=json.loads((S/'e037-thumb-ray-independent-validation-v1/report.json').read_text());source_rows={row['frame']:row for row in baseline['source_rows'] if row['hand']=='right'};banks=[];costs=[];nodes=[];empty=[];a_sections=9;node_cap=12;inside_margin=.02;constructor_overlap=.00075;actual_vel=np.array([12,12,12,12,6.857,12,12]);kinchecks=[]
def write(name,value):(O/name).write_text(json.dumps(g.plain(value),allow_nan=False,indent=2)+'\n')
for frame in range(len(c.q)):
    base=rootcommands[frame,1].copy();original=c.q[frame,1];lo,hi=c.bounds(frame);frame_nodes=[]
    if weights[frame]<=1e-14:
        frame_nodes=[dict(command=base,untouched_reference=True,cost=0.)]
    else:
        camera=c.cameras[frame];A=camera[:3,:3].T@c.stage.T;offset=camera[:3,:3].T@camera[:3,3];R=Rotation.from_quat(base[3:7]).as_matrix();observed=c.uv((c.human[frame,1,[0,4,8,12]]-camera[:3,3])@camera[:3,:3]);caps=np.array(source_rows[frame]['source_before_after_px'][0]);joint=base[7:].copy();other_ok=True
        # Exact continuous proximal source sections, original distal retained.
        for digit,ji,sign in [('index',0,1),('middle',2,-1)]:
            dist=joint[ji+1];vx=.0458+.035*np.cos(dist);vy=.035*np.sin(dist);C0=A@(base[:3]+R@np.array([.0777,-.0016,.0285*sign]))-offset;Cc=A@R@np.array([vx,vy,0]);Cs=A@R@np.array([-vy,vx,0]);domain=h.harmonic_image_domain(C0,Cc,Cs,c.K,observed[2 if ji==0 else 3],max(caps[2 if ji==0 else 3]-inside_margin,1e-8),[lo[ji],hi[ji]]);choice=h.closest_interior_angle(domain,joint[ji]);g.journal(j,dict(kind='new_common_root_other_finger_source_section',frame=frame,digit=digit,domain=domain,selected_angle=None if choice is None else choice[1],fixed_distal_is_one_section_not_continuous_exclusion=True))
            if choice is None:other_ok=False
            else:joint[ji]=choice[1]
        if other_ok:
            axials=sorted(set([float(np.clip(original[11],lo[4],hi[4]))]+[float(v) for v in np.linspace(lo[4],hi[4],a_sections)]))
            for a_index,a in enumerate(axials):
                chart=f.face_chart(palm,thumb,normals,origin,a,[lo[5],hi[5]],constructor_overlap);g.journal(j,dict(kind='new_axial_compiled_face_chart',frame=frame,axial_section=a_index,axial_angle_rad=a,compiled_palm_face_count=len(chart),all_face_domains=chart,finite_axial_scope_not_exclusion=True));proposals=[]
                for plane in chart:
                    for interval in plane['domain']['intervals']:
                        if not interval['inside_sufficient_plane']:continue
                        left,right=interval['interval_rad'];pad=min(1e-7,(right-left)/10);b=float(np.clip(original[12],left+pad,right-pad));proposals.append(dict(proximal=b,face=plane['palm_face_index'],interval=[left,right],plane=plane))
                unique={}
                for proposal in sorted(proposals,key=lambda p:(abs(p['proximal']-original[12]),p['face'])):
                    key=round(proposal['proximal'],10)
                    if key not in unique:unique[key]=proposal
                for proposal in list(unique.values())[:4]:
                    b=proposal['proximal'];Ry=Rotation.from_rotvec([0,a,0]).as_matrix();Rz=Rotation.from_rotvec([0,0,b]).as_matrix();local0=np.array([.0255,0,0])+Ry@(origin+Rz@np.array([0,.0458,0]));C0=A@(base[:3]+R@local0)-offset;Cc=A@R@Ry@Rz@np.array([0,.035,0]);Cs=A@R@Ry@Rz@np.array([-.035,0,0]);domain=h.harmonic_image_domain(C0,Cc,Cs,c.K,observed[1],max(caps[1]-inside_margin,1e-8),[lo[6],hi[6]]);choice=h.closest_interior_angle(domain,original[13]);g.journal(j,dict(kind='new_compiled_face_and_distal_source_chart',frame=frame,axial_angle_rad=a,proximal_face_proposal=proposal,source_domain=domain,selected_distal=None if choice is None else choice[1]));
                    if choice is None:continue
                    cmd=base.copy();cmd[7:]=joint;cmd[11:]=[a,b,choice[1]];g.journal(j,dict(kind='BEFORE_new_URDF_source_chart_point',frame=frame,command=cmd,not_native_or_reference_qualification=True));poses=kin.poses(cmd[:7],dict(zip(c.manifest['joint_names']['right'],cmd[7:],strict=True)));world=np.array([poses[name][:3,3] for name in ['right_hand_palm_link','right_thumb_tip','right_index_tip','right_middle_tip']]);C=np.einsum('ij,nj->ni',A,world,optimize=False)-offset;errors=np.linalg.norm(c.uv(C)-observed,axis=1);predicted=C0+Cc*np.cos(cmd[13])+Cs*np.sin(cmd[13]);closure=float(np.max(abs(predicted-C[1])));n=np.array(proposal['plane']['normal_in_thumb0_frame']);moving=thumb@Rz.T+origin;gap=float(np.min(moving@n)-proposal['plane']['palm_support']);passed=bool((errors<=caps+1e-6).all() and (cmd[7:]>=lo-1e-8).all() and (cmd[7:]<=hi+1e-8).all() and gap>=-constructor_overlap-1e-12 and closure<1e-8);entry=dict(command=cmd,frame=frame,source_errors_px=errors,ORIGINAL_saved_native_source_caps_px=caps,compiled_face_gap_m=gap,face_index=proposal['face'],thumb_harmonic_URDF_closure_m=closure,source_and_actual_URDF_point_passed=passed,cost=float(np.sum((cmd[7:]-c.aligned[frame,1,7:])**2)));g.journal(j,dict(kind='AFTER_new_URDF_source_chart_point',record=entry));kinchecks.append(closure)
                    if passed:frame_nodes.append(entry)
            # Source/geometry-only deterministic pruning; no native result used.
            frame_nodes=sorted(frame_nodes,key=lambda v:(v['cost'],tuple(v['command'][7:])));seen=set();distinct=[]
            for row in frame_nodes:
                key=tuple(np.round(row['command'][7:],10))
                if key not in seen:distinct.append(row);seen.add(key)
            frame_nodes=distinct[:node_cap]
    nodes.append(frame_nodes);bank=np.array([v['command'][7:] for v in frame_nodes]).reshape(-1,7);banks.append(bank);costs.append(np.array([v['cost'] for v in frame_nodes]));g.journal(j,dict(kind='frozen_new_frame_chart_bank',frame=frame,count=len(frame_nodes),nodes=frame_nodes))
    if not len(frame_nodes):empty.append(frame)
write('all-new-compiled-source-chart-nodes.json',nodes);graph=finite_temporal_reachability(banks,c.aligned[:,1,7:],c.ts,c.limits[:,0],c.limits[:,1],actual_vel,component_cap=.25,adjacent_added_cap=.05,second_added_cap=.05,node_costs=costs,tolerance=1e-7);graphrow=dataclasses.asdict(graph);g.journal(j,dict(kind='NEW_whole_clock_original_finger_temporal_reachability',record=graphrow));write('new-chart-temporal-reachability.json',graphrow);selected=None
if graph.path_indices is not None:
    selected=rootcommands.copy()
    for frame,index in enumerate(graph.path_indices):selected[frame,1]=nodes[frame][index]['command']
    delta=selected[:,1,7:]-c.aligned[:,1,7:];speed=np.abs(np.diff(selected[:,1,7:],axis=0))/np.diff(c.ts)[:,None];checks=dict(component_rad=float(abs(delta).max()),added_adjacent_rad=float(abs(np.diff(delta,axis=0)).max()),added_second_rad=float(abs(np.diff(delta,n=2,axis=0)).max()),URDF_velocity_max_ratio=float((speed/actual_vel).max()));g.journal(j,dict(kind='independent_direct_original_whole_clock_motion_arrays_before_any_native',checks=checks));assert checks['component_rad']<=.25+1e-7 and checks['added_adjacent_rad']<=.05+1e-7 and checks['added_second_rad']<=.05+1e-7 and checks['URDF_velocity_max_ratio']<=1+1e-7;np.savez_compressed(O/'UNQUALIFIED-source-face-motion-path.npz',robot_qpos=selected)
j.close();out=dict(schema='v2d.coupled-common-source-face-path.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),input_manifest_sha256=sha(c.F/'manifest.json'),root_mode_proposal_sha256=sha(path),compiled_geometry_sha256=sha(geometrypath),new_analytic_controls=controls,finite_axial_sections=a_sections,source_geometry_nodes_cap_per_clock=node_cap,full_clock_rows=len(banks),new_actual_URDF_source_chart_points=len(kinchecks),maximum_thumb_harmonic_URDF_closure_m=max(kinchecks,default=0.),empty_chart_frames=empty,new_temporal_path_exists=graph.path_indices is not None,all_original_source_root_finger_motion_prerequisites_passed_for_new_path=selected is not None,all_new_states_domains_roots_and_return_records_durable=True,finite_axial_distal_sections_face_subset_and_pruning_not_global_exclusion=True,no_prior_isolated_neighbor_control_mandated=True,current_verified_reference_unchanged=True,no_native_position_or_physics=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_task_qualified=False,all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);write('report.json',out);print(json.dumps(g.plain(out)))

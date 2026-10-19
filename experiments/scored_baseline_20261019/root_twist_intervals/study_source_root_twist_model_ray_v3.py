"""NEW source-model-directed ray at114; full remaining twist, unchanged gates."""
from pathlib import Path
import datetime,hashlib,itertools,json,math,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');O=S/'e037-root-twist-model-ray-study-v3';start=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
np=c.np
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.kinematics import URDFKinematics
from v2d_dex3_flashchord.planar_thumb_ik import PlanarThumbChain,thumb_point
from v2d_dex3_flashchord.source_ray_kinematics import ray_bound_root_rotations
from v2d_dex3_flashchord.source_root_twist import partition_root_twist_domain,rotation_at_twist
library_path=S/'e037-root-twist-library-v1/report.json';library=json.loads(library_path.read_text());assert library['new_library_controls_passed'];original=c.q.copy();chain=PlanarThumbChain((.0255,0,0),(-.0025,.0193),.0458,.035);kin=URDFKinematics(c.F/'robot/urdfs/g1/dex3_right.urdf');B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037';source_hashes={r['path']:r['sha256'] for r in json.loads((B/'manifest.json').read_text())['sources']}
cached_path=S/'e037-coupled-source-disk-pose-bank-v2/report.json';cached=json.loads(cached_path.read_text());assert sha(c.F/'manifest.json')==cached['input_manifest_sha256'];cached_rows={r['frame']:r for r in cached['rows']};seed_path=S/'e037-source-orientation-coupled-fingers-v1/report.json';seed=json.loads(seed_path.read_text());seeds={r['frame']:np.array(r['command']) for r in seed['candidate_commands']};legacy_path=S/'e037-right-thumb-bracketed-v2';legacy=json.loads((legacy_path/'report.json').read_text());assert not legacy['all_original_motion_caps_passed'] and not legacy['whole_reference_or_physical_task_qualified'];assert sha(legacy_path/'right-thumb-reference-commands.npz')==legacy['commands_sha256']
with np.load(legacy_path/'right-thumb-reference-commands.npz') as z:legacy_commands=z['robot_qpos'].copy()
def planar_branches(point,lo,hi):
 x,y=point;co=(x*x+y*y-.0458**2-.035**2)/(2*.0458*.035)
 if abs(co)>1+1e-12:return []
 out=[]
 for b in [math.acos(np.clip(co,-1,1)),-math.acos(np.clip(co,-1,1))]:
  a=math.atan2(y,x)-math.atan2(.035*math.sin(b),.0458+.035*math.cos(b))
  for aa,bb in itertools.product([a-2*math.pi,a,a+2*math.pi],[b-2*math.pi,b,b+2*math.pi]):
   v=np.array([aa,bb])
   if (v>=lo-1e-9).all() and (v<=hi+1e-9).all() and not any(abs(v-z).max()<1e-10 for z in out):out.append(v)
 return out
def project_disk(pixel,center,radius):
 delta=pixel-center;length=float(np.linalg.norm(delta));return pixel.copy() if length<=radius else center+delta*(radius/length)
def planar_point(v,z):
 a,b=v;return np.array([.0777+.0458*math.cos(a)+.035*math.cos(a+b),-.0016+.0458*math.sin(a)+.035*math.sin(a+b),z])
cases=[];proposals=0;accepted=[];partitions=[];seen=set()
with (O/'all-proposals.jsonl').open('w') as journal:
 for frame,kind in [(114,'rejected_path_thumb_component_hypothesis')]:
  if time.monotonic()-start>205:raise TimeoutError('New root twist study wall cap')
  base=original[frame,1].copy();preferred=seeds[frame].copy() if kind=='retained_isolated_control' else base.copy()
  if kind=='rejected_path_thumb_component_hypothesis':preferred[11:]=legacy_commands[frame,1,11:]
  lo,hi=c.bounds(frame);assert (preferred[7:]>=lo-1e-8).all() and (preferred[7:]<=hi+1e-8).all();camera=c.cameras[frame];A=camera[:3,:3].T@c.stage.T;offset=camera[:3,:3].T@camera[:3,3];origin=A.T@offset;alignedR=Rotation.from_quat(c.aligned[frame,1,3:7]).as_matrix();oldR=Rotation.from_quat(base[3:7]).as_matrix();p=B/f'source/hands/right/{frame:06d}.json';assert sha(p)==source_hashes[str(p.relative_to(B))];source=json.loads(p.read_text());assert source['directly_observed'] and source['visual_valid'];sourceR=c.stage@camera[:3,:3]@Rotation.from_rotvec(source['mano']['global_orient']).as_matrix()@np.diag([-1.,-1.,1.]);ori_cap=float(Rotation.from_matrix(sourceR@oldR.T).magnitude());before=np.array(cached_rows[frame]['before_source_errors_px']);observed=c.uv(np.einsum('ni,ij->nj',c.human[frame,1,[4,8,12]]-camera[:3,3],camera[:3,:3],optimize=False));caps=before[1:]-np.minimum(.0001,.01*before[1:]);local=thumb_point(chain,preferred[11:]);anchors=[]
  if kind=='rejected_path_thumb_component_hypothesis':
   preferred_pixel=c.uv((A@(base[:3]+sourceR@local)-offset)[None])[0];pixel=project_disk(preferred_pixel,observed[0],caps[0]);ray=A.T@np.r_[(pixel-[c.K[0,2],c.K[1,2]])/[c.K[0,0],c.K[1,1]],1.]
   for b in ray_bound_root_rotations(local,base[:3],origin,ray,oldR,(.0100001,np.inf)).branches:anchors.append((b.rotation,dict(kind='new_source_ray_anchor',hypothesized_camera_depth_m=b.parameter)))
  else:anchors.append((Rotation.from_quat(preferred[3:7]).as_matrix(),dict(kind='retained_point_mapping_only')))
  summary=dict(frame=frame,kind=kind,source_record_sha256=sha(p),source_direct_frame_flag=bool(c.flags['hand_directly_observed'][frame,1]),source_metric_frame_flag=bool(c.flags['hand_metric_valid'][frame,1]),source_orientation_cap_rad=ori_cap,source_image_caps_px=before.tolist(),preferred_command=preferred.tolist(),old_whole_path_still_rejected=kind=='rejected_path_thumb_component_hypothesis',anchors=len(anchors),intervals=0,unresolved_leaves=0,proposed_points=0,accepted_points=0);cases.append(summary)
  for anchor_index,(anchor,association) in enumerate(anchors):
   axis=anchor@local;axis/=np.linalg.norm(axis);partition=partition_root_twist_domain(anchor,axis,sourceR,ori_cap,alignedR,[.08]*3,minimum_width_rad=1e-5,maximum_boxes=1024);partition_record=dict(frame=frame,kind=kind,anchor_index=anchor_index,anchor_rotation=anchor.tolist(),axis_world=axis.tolist(),local_thumb_point=local.tolist(),source_rotation=sourceR.tolist(),aligned_rotation=alignedR.tolist(),source_orientation_cap_rad=ori_cap,root_component_caps_rad=[.08]*3,association=association,partition=partition);partitions.append(partition_record);nodes={n['id']:n for n in partition['nodes']};leaves=[nodes[i] for i in partition['leaf_ids']];active=[n for n in leaves if n['classification']!='outside_component_box'];summary['intervals']+=len(active);summary['unresolved_leaves']+=sum(n['classification']=='unresolved' for n in leaves);theta_records={}
   for leaf in active:
    left,right=leaf['interval'];samples=np.r_[np.linspace(left,right,max(2,int(math.ceil((right-left)/.01))+1)),(left+right)/2]
    if left<=0<=right:samples=np.r_[samples,0.]
    for theta in samples:theta_records.setdefault(round(float(theta),13),dict(theta=float(theta),leaf_ids=[]))['leaf_ids'].append(leaf['id'])
   for value in theta_records.values():
    if time.monotonic()-start>215:raise TimeoutError('Root twist witness cap')
    theta=value['theta'];rootR=rotation_at_twist(anchor,axis,theta);residual=Rotation.from_matrix(rootR@alignedR.T).as_rotvec();ori=float(Rotation.from_matrix(sourceR@rootR.T).magnitude());record=dict(frame=frame,kind=kind,anchor_index=anchor_index,theta_rad=theta,leaf_ids=value['leaf_ids'],root_original_residual_rad=residual.tolist(),source_model_orientation_error_rad=ori,source_model_orientation_not_measured=True,local_thumb_point=local.tolist(),mapped_thumb_point_native=(base[:3]+rootR@local).tolist(),proposed_root_quaternion_xyzw=Rotation.from_matrix(rootR).as_quat().tolist());summary['proposed_points']+=1;proposals+=1
    if abs(residual).max()>.08+1e-9 or ori>ori_cap+1e-9:record.update(passed=False,rejection='exact root/source-orientation gate');journal.write(json.dumps(record)+'\n');continue
    digitbanks=[]
    for di,(z,cols) in enumerate([(.0285,[0,1]),(-.0285,[2,3])],start=1):
     desired=preferred[7+np.array(cols)];camera_point=A@(base[:3]+rootR@planar_point(desired,z))-offset
     if camera_point[2]<=.01:digitbanks.append([]);continue
     predicted=c.uv(camera_point[None])[0];pixel=project_disk(predicted,observed[di],caps[di]);ray=np.r_[(pixel-[c.K[0,2],c.K[1,2]])/[c.K[0,0],c.K[1,1]],1.];local_origin=rootR.T@(origin-base[:3]);local_ray=rootR.T@(A.T@ray)
     if abs(local_ray[2])<1e-10:digitbanks.append([]);continue
     depth=(z-local_origin[2])/local_ray[2];point=local_origin+depth*local_ray;digitbanks.append(planar_branches(point[:2]-[.0777,-.0016],lo[cols],hi[cols]) if depth>.01 else [])
    record['planar_digit_branch_counts']=[len(b) for b in digitbanks]
    if any(not b for b in digitbanks):record.update(passed=False,rejection='actual bounded planar finger inverse');journal.write(json.dumps(record)+'\n');continue
    for index,middle in itertools.product(*digitbanks):
     trial=dict(record);command=preferred.copy();command[:3]=base[:3];command[3:7]=Rotation.from_matrix(rootR).as_quat();command[7:9]=index;command[9:11]=middle;key=(frame,*np.round(np.r_[residual,command[7:]],11));trial['command']=command.tolist()
     if key in seen:trial.update(passed=False,rejection='duplicate proposed state, prior result retained');journal.write(json.dumps(trial)+'\n');continue
     seen.add(key);c.q=original.copy();c.q[frame,1]=command;v=c.evaluate(frame,command[7:]);depth=max(0.,float(-v['gaps'].min()));source_ok=bool((v['errors']<=before+1e-6).all());position_ok=bool((command[7:]>=lo-1e-8).all() and (command[7:]<=hi+1e-8).all());target=[]
     for ct in c.data.contact[:c.data.ncon]:
      a,b=int(c.model.geom_bodyid[ct.geom1]),int(c.model.geom_bodyid[ct.geom2]);ra=c.bn[a].startswith('dex3_right_');rb=c.bn[b].startswith('dex3_right_')
      if ra!=rb and not (c.bn[b] if ra else c.bn[a]).startswith('dex3_'):target.append(dict(bodies=[c.bn[a],c.bn[b]],signed_gap_m=float(ct.dist)))
     target_depth=max([max(0.,-r['signed_gap_m']) for r in target],default=0.);passed=bool(source_ok and position_ok and depth<=.001 and target_depth<=.001);trial.update(source_image_before_after_px=[before.tolist(),v['errors'].tolist()],source_image_nonregression=source_ok,original_URDF_component_bounds_passed=position_ok,native_self_penetration_m=depth,native_self_contacts=v['self_contacts'],native_hand_target_contacts=target,native_hand_target_penetration_m=target_depth,passed=passed,rejection=None if passed else 'source/actual URDF/native self or hand-target gate',no_force_or_dynamic_qualification=True);journal.write(json.dumps(trial)+'\n')
     if passed:accepted.append(trial);summary['accepted_points']+=1
  print(json.dumps(summary),flush=True)
c.q=original;part_path=O/'all-interval-partitions.json';part_path.write_text(json.dumps(partitions,indent=2)+'\n');accepted_path=O/'accepted-finite-witnesses.json';accepted_path.write_text(json.dumps(accepted,indent=2)+'\n');report=dict(schema='v2d.source-root-twist-model-ray-study.v3',new_ray_from_unchanged_source_model_projection=True,previous_current_projection_family_not_repeated=True,previous_source_disk_report_sha256=sha(S/'e037-root-twist-source-disk-study-v2/report.json'),prior_narrow_point_ray_family_retained=True,positive_control_intervals_and_witnesses_not_repeated=True,prior_interval_report_sha256=sha(S/'e037-root-twist-interval-study-v1/report.json'),thumb_anchor_is_projection_of_preferred_point_into_original_image_disk=True,source_caps_not_changed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),context_sha256=sha(S/'compute/source_orientation_context_v1.py'),input_manifest_sha256=sha(c.F/'manifest.json'),native_model_sha256=sha(c.MP),library_sha256=library['library_sha256'],cached_source_caps_report_sha256=sha(cached_path),retained_isolated_controls_sha256=sha(seed_path),rejected_path_report_sha256=sha(legacy_path/'report.json'),rejected_path_commands_sha256=legacy['commands_sha256'],cases=cases,all_intervals_sha256=sha(part_path),all_proposals_sha256=sha(O/'all-proposals.jsonl'),accepted_witnesses_sha256=sha(accepted_path),total_interval_anchors=len(partitions),total_proposed_root_points=proposals,accepted_finite_witnesses=len(accepted),full_rejected_and_unresolved_interval_records_saved=True,all_evaluated_states_and_rejections_saved=True,numerical_enclosures_not_formal_global_exclusion=True,twist_preserves_hypothesized_robot_tip_not_measured_human_depth=True,legacy_whole_path_rejected_no_certificate_reused=True,source_flags_original_bounds_and_native_parameters_unchanged=True,current_three_component_reference_unchanged=True,source_fit_native_model_or_old_control_jobs_not_repeated=True,no_native_compile_or_physics_step=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_task_or_policy_qualified=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(accepted_witnesses=len(accepted),accepted_frames=sorted(set(r['frame'] for r in accepted)),wall_seconds=report['wall_seconds']),indent=2))

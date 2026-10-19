"""NEW joint-continuous Rodrigues/thumb2/index/middle source domain at114.

The fixed upstream thumb pair is an explicit retained robot hypothesis, not
observed human articulation. Every split, center, rejected and unresolved box
is retained. Only independently checked point states can be constructive.
"""
from pathlib import Path
import collections,datetime,hashlib,heapq,json,math,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-coupled-articulation-domain-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import coupled_articulation_intervals_v1 as b
import source_orientation_context_v1 as c
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.planar_thumb_ik import PlanarThumbChain,thumb_point
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
binding=json.loads((S/'e037-coupled-articulation-binding-v1/report.json').read_text());assert all(x['passed'] for x in binding['controls']) and binding['library_sha256']==sha(S/'compute/coupled_articulation_intervals_v1.py')
original=c.q.copy();frame=114;base=original[frame,1].copy();lo,hi=c.bounds(frame);chain=PlanarThumbChain((.0255,0,0),(-.0025,.0193),.0458,.035)
legacy=S/'e037-right-thumb-bracketed-v2';legacy_report=json.loads((legacy/'report.json').read_text());assert not legacy_report['all_original_motion_caps_passed']
with np.load(legacy/'right-thumb-reference-commands.npz') as z:upstream=z['robot_qpos'][frame,1,11:13].copy()
B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037';sourcefile=B/f'source/hands/right/{frame:06d}.json';hashes={r['path']:r['sha256'] for r in json.loads((B/'manifest.json').read_text())['sources']};assert sha(sourcefile)==hashes[str(sourcefile.relative_to(B))];source=json.loads(sourcefile.read_text());camera=c.cameras[frame];A=camera[:3,:3].T@c.stage.T;offset=camera[:3,:3].T@camera[:3,3];alignedR=Rotation.from_quat(c.aligned[frame,1,3:7]).as_matrix();oldR=Rotation.from_quat(base[3:7]).as_matrix();sourceR=c.stage@camera[:3,:3]@Rotation.from_rotvec(source['mano']['global_orient']).as_matrix()@np.diag([-1.,-1.,1.]);cap=float(Rotation.from_matrix(sourceR@oldR.T).magnitude());qsa=Rotation.from_matrix(sourceR@alignedR.T).as_quat();radius=math.tan(cap/2);rv=Rotation.from_matrix(oldR@sourceR.T).as_rotvec();preferred_root=np.tan(np.linalg.norm(rv)/2)*rv/np.linalg.norm(rv);preferred=np.r_[preferred_root,base[13],base[7:11]]
capsreport=S/'e037-coupled-source-disk-pose-bank-v2/report.json';cached=json.loads(capsreport.read_text());before=np.array(next(r for r in cached['rows'] if r['frame']==frame)['before_source_errors_px']);observed=c.uv((c.human[frame,1,[4,8,12]]-camera[:3,3])@camera[:3,:3]);source_pixels=c.uv((c.human[frame,1,[0,4,8,12]]-camera[:3,3])@camera[:3,:3]);position_tol=1e-8;source_tol=1e-6;orientation_tol=1e-9
initial=np.r_[np.tile([-radius,radius],(3,1)),[[lo[6],hi[6]]],np.c_[lo[:4],hi[:4]]];variable_names=['world_Rodrigues_x','world_Rodrigues_y','world_Rodrigues_z','thumb2','index0','index1','middle0','middle1'];counts=collections.Counter();accepted=[];seen=set();native_records=[];nodes=[];pending=[];maximum_boxes=12000;minimum_scaled_width=.0002;maximum_new_native_states=64;maximum_accepted=8;wall_budget=150
def rotation(r):
 length=np.linalg.norm(r);return Rotation.from_rotvec(2*np.arctan(length)*r/max(length,1e-300)).as_matrix()@sourceR
def planar(a,d,z):return np.array([.0777+.0458*np.cos(a)+.035*np.cos(a+d),-.0016+.0458*np.sin(a)+.035*np.sin(a+d),z])
def point(x):
 root=rotation(x[:3]);angles=base[7:].copy();angles[4:6]=upstream;angles[6]=x[3];angles[:4]=x[4:];command=base.copy();command[3:7]=Rotation.from_matrix(root).as_quat();command[7:]=angles;local=np.array([[0.,0.,0.],thumb_point(chain,angles[4:]),planar(*angles[:2],.0285),planar(*angles[2:4],-.0285)]);C=(A@(base[:3]+(root@local.T).T).T).T-offset;errors=np.linalg.norm(c.uv(C)-source_pixels,axis=1);residual=Rotation.from_matrix(root@alignedR.T).as_rotvec();ori=2*np.arctan(np.linalg.norm(x[:3]));merit=float(np.maximum(errors/before-1,0).sum()+10*np.maximum(abs(residual)-.08,0).sum()+10*max(0.,ori-cap));return command,errors,residual,ori,merit
def box_bound(box):
 norm2=b.sum_intervals(b.square(box[:3]));rvbox=b.root_rotvec_box(box[:3],qsa)
 if norm2[0]>radius**2+2e-13:return dict(rejection='outside_original_source_model_orientation',root_norm_squared=norm2.tolist())
 if rvbox is not None and ((rvbox[:,0]>.08+orientation_tol)|(rvbox[:,1]<-.08-orientation_tol)).any():return dict(rejection='outside_original_world_component_root_box',root_rotvec_enclosure=rvbox.tolist())
 points=[b.thumb_point_box(upstream,box[3]),b.planar_point_box(box[4:6],.0285),b.planar_point_box(box[6:8],-.0285)];pixelboxes=[];low=[]
 for i,p in enumerate(points):
  C=b.camera_point_box(box[:3],p,sourceR,base[:3],A,offset);pixels=b.image_box(C,c.K);pixelboxes.append(None if pixels is None else pixels.tolist());low.append(b.disk_lower_bound(pixels,observed[i]))
 result=dict(root_rotvec_enclosure=None if rvbox is None else rvbox.tolist(),source_image_lower_bounds_px=low,source_pixel_enclosures=pixelboxes)
 if any(v>before[i+1]+source_tol for i,v in enumerate(low)):result['rejection']='outside_original_source_image_disk'
 return result
def native(command,errors):
 c.q=original.copy();c.q[frame,1]=command;v=c.evaluate(frame,command[7:]);all_contacts=[];maxima=dict(right_self=0.,external=0.,interhand=0.)
 for ct in c.data.contact[:c.data.ncon]:
  ga,gb=int(ct.geom1),int(ct.geom2);a,bb=int(c.model.geom_bodyid[ga]),int(c.model.geom_bodyid[gb]);ra=c.bn[a].startswith('dex3_right_');rb=c.bn[bb].startswith('dex3_right_')
  if not (ra or rb):continue
  other=c.bn[bb] if ra else c.bn[a];kind='right_self' if ra and rb else ('interhand' if other.startswith('dex3_left_') else 'external');depth=max(0.,-float(ct.dist));maxima[kind]=max(maxima[kind],depth);all_contacts.append(dict(kind=kind,bodies=[c.bn[a],c.bn[bb]],native_geom_ids=[ga,gb],signed_gap_m=float(ct.dist)))
 closure=float(abs(v['errors']-errors).max());passed=bool(max(maxima.values())<=.001 and (v['errors']<=before+source_tol).all() and closure<1e-5);record=dict(command=command.tolist(),source_errors_px=v['errors'].tolist(),analytic_native_source_error_closure_px=closure,contacts=all_contacts,native_penetrations_m=maxima,passed=passed,point_checks_only_no_temporal_motor_or_task_certificate=True);native_records.append(record);counts['native_point_states']+=1
 if passed:accepted.append(record)
 return record
def make(box,parent=None):
 idx=len(nodes);node=dict(id=idx,parent=parent,box=box.tolist());bound=box_bound(box);node.update(bound)
 if 'rejection' in bound:node['classification']='rejected';counts[bound['rejection']]+=1;nodes.append(node);return
 center=(box[:,0]+box[:,1])/2;fixed=np.clip(preferred,box[:,0],box[:,1]);point_records=[]
 for label,x in [('midpoint',center),('clipped_current_preference',fixed)]:
  command,errors,residual,ori,merit=point(x);key=tuple(np.round(x,13));r=dict(kind=label,parameters=x.tolist(),command=command.tolist(),source_errors_px=errors.tolist(),root_world_residual_rad=residual.tolist(),source_model_orientation_error_rad=float(ori),merit=merit)
  ready=bool((errors<=before+source_tol).all() and abs(residual).max()<=.08+orientation_tol and ori<=cap+orientation_tol and (command[7:]>=lo-position_tol).all() and (command[7:]<=hi+position_tol).all())
  r['source_root_URDF_point_prerequisites_passed']=ready
  if ready and key not in seen and counts['native_point_states']<maximum_new_native_states:
   seen.add(key);r['native_record_index']=len(native_records);r['native_passed']=native(command,errors)['passed']
  point_records.append(r)
 node['proposed_points']=point_records;node['classification']='pending';nodes.append(node);best=min(v['merit'] for v in point_records);width=(box[:,1]-box[:,0]);heuristic=best-.00001*float(width.sum());heapq.heappush(pending,(heuristic,idx))
 return
make(initial)
while pending and len(nodes)<maximum_boxes and time.monotonic()-start<wall_budget and len(accepted)<maximum_accepted and counts['native_point_states']<maximum_new_native_states:
 _,idx=heapq.heappop(pending);node=nodes[idx];box=np.array(node['box']);width=box[:,1]-box[:,0];weights=np.array([3.,3.,3.,.035,.08,.035,.08,.035]);score=width*weights
 if score.max()<minimum_scaled_width:node['classification']='unresolved_minimum_width';counts['unresolved_minimum_width']+=1;continue
 axis=int(score.argmax());mid=float((box[axis,0]+box[axis,1])/2);left=box.copy();right=box.copy();left[axis,1]=mid;right[axis,0]=mid;node['classification']='split';node['split_axis']=variable_names[axis];node['children']=[len(nodes),len(nodes)+1];make(left,idx);make(right,idx)
for _,idx in pending:nodes[idx]['classification']='unresolved_budget_or_constructive_stop'
leaves=[n for n in nodes if n['classification']!='split'];assert all('classification'in n for n in nodes);c.q=original
npth=O/'all-domain-nodes-and-states.json';npth.write_text(json.dumps(nodes,separators=(',',':'))+'\n');ap=O/'all-native-point-records.json';ap.write_text(json.dumps(native_records,indent=2)+'\n');wp=O/'accepted-isolated-witnesses.json';wp.write_text(json.dumps(accepted,indent=2)+'\n')
out=dict(schema='v2d.coupled-articulation-domain.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),library_sha256=sha(S/'compute/coupled_articulation_intervals_v1.py'),binding_report_sha256=sha(S/'e037-coupled-articulation-binding-v1/report.json'),input_manifest_sha256=sha(c.F/'manifest.json'),native_model_sha256=sha(c.MP),source_record_sha256=sha(sourcefile),original_source_caps_report_sha256=sha(capsreport),frame=frame,variable_names=variable_names,initial_box=initial.tolist(),source_image_caps_px=before.tolist(),source_model_orientation_cap_rad=cap,fixed_upstream_thumb_pair=upstream.tolist(),upstream_is_robot_hypothesis_not_measured_source_truth=True,translation_and_source_authority_unchanged=True,both_index_and_middle_angles_continuous=True,distal_thumb_articulation_continuous=True,source_model_Rodrigues_ball_continuous=True,numerical_interval_padding_not_formal_global_proof=True,original_reference_position_tolerance_rad=position_tol,original_source_image_tolerance_px=source_tol,root_component_tolerance_rad=orientation_tol,all_nodes_sha256=sha(npth),all_native_records_sha256=sha(ap),accepted_witnesses_sha256=sha(wp),counts=dict(counts),nodes=len(nodes),leaves=len(leaves),unresolved_leaves=sum(n['classification'].startswith('unresolved') for n in leaves),all_proposed_states_and_rejected_unresolved_boxes_saved=True,accepted_isolated_points=len(accepted),new_native_states=len(native_records),no_completed_family_or_generic_optimizer_repeated=True,no_global_infeasibility_or_trajectory_claim=True,current_three_component_freeze_unchanged=True,new_GPU_jobs=0,new_PPO_steps=0,no_native_compile_or_physics_step=True,whole_reference_or_task_qualified=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

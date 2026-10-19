"""NEW conditional continuous thumb1/thumb2 preimage of original image disk.

The root/index/middle are a retained source-residual proposal, not a source
pose winner or qualified reference. Moving thumb1 also changes its native
palm clearance, so every potential constructive point gets fresh contacts.
"""
from pathlib import Path
import collections,datetime,hashlib,heapq,json,math,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-thumb-proximal-distal-slice-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import coupled_articulation_intervals_v1 as b
import source_orientation_context_v1 as c
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.planar_thumb_ik import PlanarThumbChain,thumb_point
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();P=S/'e037-coupled-articulation-independent-v1';ir=json.loads((P/'report.json').read_text());bestpath=P/'best-retained-proposals-diagnostic-only.json';assert sha(bestpath)==ir['best_diagnostic_only_sha256'];parent=json.loads(bestpath.read_text())[0];study=json.loads((S/'e037-coupled-articulation-domain-v1/report.json').read_text());base=np.array(parent['command']);frame=114;original=c.q.copy();lo,hi=c.bounds(frame);before=np.array(study['source_image_caps_px']);camera=c.cameras[frame];A=camera[:3,:3].T@c.stage.T;offset=camera[:3,:3].T@camera[:3,3];rootR=Rotation.from_quat(base[3:7]).as_matrix();observed=c.uv((c.human[frame,1,[0,4,8,12]]-camera[:3,3])@camera[:3,:3]);chain=PlanarThumbChain((.0255,0,0),(-.0025,.0193),.0458,.035);fixed_axial=float(base[11]);initial=np.c_[lo[5:7],hi[5:7]];nodes=[];pending=[];native=[];accepted=[];seen=set();maximum_boxes=2048;native_cap=64;counts=collections.Counter()
def local_box(box):
 proximal,distal=box;total=b.add(proximal,distal);x=b.sub(b.exact(-.0025),b.add(b.mul(b.exact(.0458),b.sine(proximal)),b.mul(b.exact(.035),b.sine(total))));y=b.add(b.exact(.0193),b.add(b.mul(b.exact(.0458),b.cosine(proximal)),b.mul(b.exact(.035),b.cosine(total))));return np.array([b.add(b.exact(.0255),b.mul(b.exact(np.cos(fixed_axial)),x)),y,b.mul(b.exact(-np.sin(fixed_axial)),x)])
def point(v):
 cmd=base.copy();cmd[12:14]=v;local=thumb_point(chain,cmd[11:]);C=A@(cmd[:3]+rootR@local)-offset;error=float(np.linalg.norm(c.uv(C[None])[0]-observed[1]));return cmd,error
def check(cmd,error,node_id):
 c.q=original.copy();c.q[frame,1]=cmd;value=c.evaluate(frame,cmd[7:]);contacts=[];depths=dict(right_self=0.,external=0.,interhand=0.)
 for ct in c.data.contact[:c.data.ncon]:
  aa,bb=int(c.model.geom_bodyid[ct.geom1]),int(c.model.geom_bodyid[ct.geom2]);ra=c.bn[aa].startswith('dex3_right_');rb=c.bn[bb].startswith('dex3_right_')
  if not(ra or rb):continue
  other=c.bn[bb] if ra else c.bn[aa];kind='right_self' if ra and rb else ('interhand' if other.startswith('dex3_left_') else 'external');depths[kind]=max(depths[kind],max(0.,-float(ct.dist)));contacts.append(dict(kind=kind,bodies=[c.bn[aa],c.bn[bb]],geom_ids=[int(ct.geom1),int(ct.geom2)],signed_gap_m=float(ct.dist)))
 passed=bool((value['errors']<=before+1e-6).all() and max(depths.values())<=.001);rec=dict(node_id=node_id,command=cmd.tolist(),source_errors_px=value['errors'].tolist(),source_caps_px=before.tolist(),analytic_native_thumb_image_error_closure_px=abs(float(value['errors'][1])-error),native_contacts=contacts,native_penetrations_m=depths,passed=passed,isolated_point_no_motor_grasp_or_motion_certificate=True);native.append(rec)
 if passed:accepted.append(rec)
 return len(native)-1
def make(box,parent_id=None):
 idx=len(nodes);node=dict(id=idx,parent=parent_id,box=box.tolist());C=b.sub(b.matvec(b.exact(A),b.add(b.exact(base[:3]),b.matvec(b.exact(rootR),local_box(box)))),b.exact(offset));pixels=b.image_box(C,c.K);lower=b.disk_lower_bound(pixels,observed[1]);node.update(camera_enclosure=C.tolist(),pixel_enclosure=None if pixels is None else pixels.tolist(),image_error_lower_bound_px=lower)
 if lower>before[1]+1e-6:node.update(classification='rejected_source_image_disk');nodes.append(node);return
 proposals=[]
 for label,v in [('midpoint',box.mean(axis=1)),('clipped_parent',np.clip(base[12:14],box[:,0],box[:,1]))]:
  cmd,error=point(v);rec=dict(kind=label,parameters=v.tolist(),command=cmd.tolist(),source_thumb_error_px=error,source_thumb_passed=error<=before[1]+1e-6);key=tuple(np.round(v,13))
  if rec['source_thumb_passed'] and key not in seen and len(native)<native_cap:seen.add(key);rec['native_record_index']=check(cmd,error,idx)
  proposals.append(rec)
 node.update(proposed_states=proposals,classification='pending');nodes.append(node);heapq.heappush(pending,(min(x['source_thumb_error_px'] for x in proposals),idx))
make(initial)
while pending and len(nodes)<maximum_boxes and len(native)<native_cap and len(accepted)<8 and time.monotonic()-start<100:
 _,idx=heapq.heappop(pending);node=nodes[idx];box=np.array(node['box']);weights=np.array([.0808,.035]);axis=int(((box[:,1]-box[:,0])*weights).argmax());mid=float(box[axis].mean())
 if box[axis,1]-box[axis,0]<1e-5:node['classification']='unresolved_minimum_width';continue
 left=box.copy();right=box.copy();left[axis,1]=mid;right[axis,0]=mid;node.update(classification='split',split_axis=axis,children=[len(nodes),len(nodes)+1]);make(left,idx);make(right,idx)
for _,idx in pending:nodes[idx]['classification']='unresolved_budget_or_constructive_stop'
c.q=original;paths=[]
for name,value in [('all-nodes-and-proposals.json',nodes),('all-new-native-points.json',native),('accepted-isolated-witnesses.json',accepted)]:
 p=O/name;p.write_text(json.dumps(value,indent=2)+'\n');paths.append(p)
out=dict(schema='v2d.thumb-proximal-distal-slice.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),interval_library_sha256=sha(S/'compute/coupled_articulation_intervals_v1.py'),parent_geometric_proposal_sha256=sha(bestpath),parent_node_id=parent['node_id'],parent_selected_by_source_violation_not_physical_success=True,root_index_middle_axial_thumb_fixed_conditionally=True,source_pose_not_selected_as_measured_truth=True,initial_proximal_distal_box=initial.tolist(),source_caps_px=before.tolist(),nodes=len(nodes),unresolved_boxes=sum(n['classification'].startswith('unresolved') for n in nodes),native_point_states=len(native),isolated_witnesses=len(accepted),all_states_and_rejected_unresolved_boxes_saved=True,all_outputs_sha256={p.name:sha(p) for p in paths},numerical_interval_not_formal_global_proof=True,no_optimization_restart_or_old_finite_family_repeated=True,original_source_root_finger_bounds_and_native_parameters_unchanged=True,current_combined_reference_unchanged=True,no_native_compile_physics_or_GPU=True,whole_reference_or_task_qualified=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

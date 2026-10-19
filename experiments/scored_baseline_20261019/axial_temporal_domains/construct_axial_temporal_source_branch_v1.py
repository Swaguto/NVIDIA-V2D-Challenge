"""NEW source/temporal branch: continuous axial-thumb/index angle domains.

One correlated neighbor box and one conditional root are fixed by declared
source/kinematic criteria; no source pose is selected by physical score.
"""
from pathlib import Path
import datetime,hashlib,itertools,json,math,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-axial-temporal-source-branch-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import thumb_conic_geometry_v1 as g
import harmonic_source_domains_v1 as h
import source_orientation_context_v1 as c
from scipy.spatial.transform import Rotation
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();journal=(O/'all-domains-proposals-native-outcomes.jsonl').open('w');controls=[]
def write(name,value):(O/name).write_text(json.dumps(g.plain(value),allow_nan=False,indent=2)+'\n')
def check(name,passed,**details):
 record=dict(kind='new_harmonic_contract_control',name=name,passed=bool(passed),**details);g.journal(journal,record);controls.append(record);assert passed,name
K0=np.diag([100.,100.,1.]);domain=h.harmonic_image_domain([0,0,-1],[.05,0,0],[0,.05,0],K0,np.zeros(2),10.,[-.3,.3]);check('behind_camera_cone_is_never_inside_image_domain',all(r['classification']=='positive_depth_failure' for r in domain['intervals']),domain=domain)
domain=h.harmonic_image_domain([0,0,1],[.05,0,0],[0,.05,0],K0,np.zeros(2),10.,[.2,.2]);choice=h.closest_interior_angle(domain,.2);check('fixed_angle_contract_retains_singleton_domain',choice is not None and choice[1]==.2,domain=domain)
P=S/'e037-temporal-box-source-cones-v1';report=json.loads((P/'report.json').read_text());selected_path=P/'selected-source-temporal-box.json';assert sha(selected_path)==report['all_outputs_sha256'][selected_path.name];selected=json.loads(selected_path.read_text());assert selected is not None;entry=selected['correlated_box'];box=np.array(entry['joint_angle_box_rad']);base=np.array(selected['root_control']['command']);frame=114;original=c.q.copy();preferred=original[frame,1,7:];rootR=Rotation.from_quat(base[3:7]).as_matrix();camera=c.cameras[frame];A=camera[:3,:3].T@c.stage.T;offset_camera=camera[:3,:3].T@camera[:3,3];observed=c.uv((c.human[frame,1,[0,4,8,12]]-camera[:3,3])@camera[:3,:3]);contract=json.loads((S/'e037-thumb-conic-clearance-v1/source-conic-contract.json').read_text());caps=np.array(contract['source_image_caps_px']);inside_margin=.02;closest=np.clip(preferred,box[:,0],box[:,1])
# Rebuild only saved neighbor IDs; this never regenerates or native-rechecks a bank.
bank=json.loads((S/'e037-coupled-source-disk-pose-bank-v2/pose-bank.json').read_text());extra=json.loads((S/'e037-root-twist-interval-study-v1/accepted-finite-witnesses.json').read_text());neighbors={f:[] for f in [113,115]}
for p in bank+extra:
 if p['frame'] not in neighbors:continue
 key=tuple(np.round(p['command'],12))
 if not any(v['key']==key for v in neighbors[p['frame']]):neighbors[p['frame']].append(dict(key=key,command=np.array(p['command'])))
neighbor_commands=[neighbors[113][entry['neighbor113_index']]['command'],neighbors[115][entry['neighbor115_index']]['command']];lo,hi=c.bounds(frame);all_domains=[]
def domain_record(name,C0,Cc,Cs,disk_index,bounds,pref):
 d=h.harmonic_image_domain(C0,Cc,Cs,c.K,observed[disk_index],caps[disk_index]-inside_margin,bounds);record=dict(kind='actual_new_harmonic_domain',name=name,domain=d,original_source_radius_px=caps[disk_index],constructive_inward_margin_px=inside_margin);g.journal(journal,record);all_domains.append(record)
 # Independent direct camera projection checks this new domain's polynomial.
 error=0.
 for angle in np.linspace(*bounds,17):
  C=C0+Cc*np.cos(angle)+Cs*np.sin(angle);poly=np.polynomial.polynomial.polyval(np.tan(angle/2),d['tan_half_angle_polynomial'])/(1+np.tan(angle/2)**2)**2;pixel=c.uv(C[None])[0];direct=(np.sum((pixel-observed[disk_index])**2)-(caps[disk_index]-inside_margin)**2)*C[2]**2;error=max(error,abs(float(poly)-float(direct)))
 check(name+'_perspective_polynomial_closure',error<1e-7,maximum_algebraic_error=error);return d,h.closest_interior_angle(d,pref)
# Index distal is retained at the correlated box midpoint, not a free depth hint.
distal=float(box[1].mean());vx=.0458+.035*np.cos(distal);vy=.035*np.sin(distal);C0=A@(base[:3]+rootR@np.array([.0777,-.0016,.0285]))-offset_camera;Cc=A@rootR@np.array([vx,vy,0.]);Cs=A@rootR@np.array([-vy,vx,0.]);index_domain,index_choice=domain_record('continuous_index0_at_temporal_distal_section',C0,Cc,Cs,2,box[0],preferred[0]);index_bank=[] if index_choice is None else [np.array([index_choice[1],distal])]
# Axial thumb is continuous in its ORIGINAL temporal interval for two frozen
# proximal/distal geometric sections. No previous failed interval slice repeats.
thumb_bank=[];bc_sections=[]
for seed in [closest[5:7],box[5:7].mean(axis=1)]:
 if not any(abs(seed-v).max()<1e-12 for v in bc_sections):bc_sections.append(seed)
for si,(prox,dist) in enumerate(bc_sections):
 X=-.0025-.0458*np.sin(prox)-.035*np.sin(prox+dist);Y=.0193+.0458*np.cos(prox)+.035*np.cos(prox+dist);C0=A@(base[:3]+rootR@np.array([.0255,Y,0.]))-offset_camera;Cc=A@rootR@np.array([X,0.,0.]);Cs=A@rootR@np.array([0.,0.,-X]);domain,choice=domain_record('continuous_axial_thumb_section_'+str(si),C0,Cc,Cs,1,box[4],preferred[4])
 if choice is not None:thumb_bank.append(np.array([choice[1],prox,dist]))
# Middle source conic uses a positive-distal actual planar branch. The exact
# boundary is partitioned by this correlated joint box; interior seeds retained.
middle_bank=[];middle_proposals=[]
def middle_point(v):
 a,bb=v;return np.array([.0458*np.cos(a)+.035*np.cos(a+bb),.0458*np.sin(a)+.035*np.sin(a+bb)])
E=np.array([[1.,0.],[0.,1.],[0.,0.]]);H=np.c_[A@rootR@E,A@(base[:3]+rootR@np.array([.0777,-.0016,-.0285]))-offset_camera];conic=g.conic_from_camera_plane(H,c.K,observed[3],caps[3]-inside_margin);ellipse=g.conic_ellipse(conic);g.journal(journal,dict(kind='actual_middle_correlated_conic',conic=conic,ellipse=ellipse,joint_box=box[2:4]));middle_breaks=[];root_records=[]
for label,v in [('clipped_original',closest[2:4]),('box_midpoint',box[2:4].mean(axis=1))]:middle_proposals.append(dict(kind=label,angles=v))
if ellipse['kind']=='ellipse':
 center=np.array(ellipse['center']);factor=np.array(ellipse['factor']);circles=[]
 for dist in box[3]:circles.append(dict(kind='distal_joint_bound',center=np.zeros(2),radius=float(np.sqrt(.0458**2+.035**2+2*.0458*.035*np.cos(dist)))))
 for prox in box[2]:circles.append(dict(kind='proximal_joint_bound',center=.0458*np.array([np.cos(prox),np.sin(prox)]),radius=.035))
 breaks=[0.,2*np.pi]
 for circle in circles:
  roots=g.ellipse_circle_roots(center,factor,circle['center'],circle['radius']);root_records.append(dict(circle=circle,roots=roots));breaks.extend(roots['angles']);g.journal(journal,dict(kind='actual_middle_conic_root',circle=circle,roots=roots))
 middle_breaks=sorted(set(breaks))
 for left,right in zip(middle_breaks,middle_breaks[1:]):
  angle=(left+right)/2;xy=center+factor@np.array([np.cos(angle),np.sin(angle)]);branches=g.thumb_planar_inverse(xy,[0.,0.],(.0458,.035),box[2]-np.pi/2,box[3]);g.journal(journal,dict(kind='actual_middle_conic_boundary_arc',interval=[left,right],planar_point=xy,shifted_branches=branches))
  for branch in branches:middle_proposals.append(dict(kind='analytic_conic_arc_midpoint',angles=branch+[np.pi/2,0.]))
seen=set()
for proposal in middle_proposals:
 angles=np.array(proposal['angles']);C=H@np.r_[middle_point(angles),1.];error=float(np.linalg.norm(c.uv(C[None])[0]-observed[3]));key=tuple(np.round(angles,12));valid=bool((angles>=box[2:4,0]-1e-8).all() and (angles<=box[2:4,1]+1e-8).all() and error<=caps[3]-inside_margin+1e-6);g.journal(journal,dict(kind='middle_source_point',proposal=proposal,camera_point=C,source_error_px=error,passed_original_box_and_strict_source=valid))
 if valid and key not in seen:seen.add(key);middle_bank.append(angles)
middle_bank=sorted(middle_bank,key=lambda v:float(np.sum((v-preferred[2:4])**2)))[:4];queue=[]
for index,middle,thumb in itertools.product(index_bank,middle_bank,thumb_bank):
 command=base.copy();command[7:9]=index;command[9:11]=middle;command[11:]=thumb;queue.append(dict(command=command,conditional_neighbor113=neighbor_commands[0],conditional_neighbor115=neighbor_commands[1],source_box_index=selected['source_bound']['box_index']))
queue=sorted(queue,key=lambda r:float(np.sum((r['command'][7:]-preferred)**2)));write('frozen-new-branch-point-queue.json',queue);g.journal(journal,dict(kind='frozen_before_native_queue',count=len(queue),sha256=sha(O/'frozen-new-branch-point-queue.json')));native=[];accepted=[]
for i,proposal in enumerate(queue[:8]):
 command=proposal['command'];g.journal(journal,dict(kind='BEFORE_new_native_position',queue_index=i,proposal=proposal));c.q=original.copy();c.q[frame,1]=command;v=c.evaluate(frame,command[7:]);contacts=[];depths=dict(right_self=0.,external=0.,interhand=0.)
 for ct in c.data.contact[:c.data.ncon]:
  aa,bb=int(c.model.geom_bodyid[ct.geom1]),int(c.model.geom_bodyid[ct.geom2]);ra=c.bn[aa].startswith('dex3_right_');rb=c.bn[bb].startswith('dex3_right_')
  if not(ra or rb):continue
  other=c.bn[bb] if ra else c.bn[aa];category='right_self' if ra and rb else ('interhand' if other.startswith('dex3_left_') else 'external');depths[category]=max(depths[category],max(0.,-float(ct.dist)));contacts.append(dict(kind=category,bodies=[c.bn[aa],c.bn[bb]],signed_gap_m=float(ct.dist)))
 q3=np.array([neighbor_commands[0][7:],command[7:],neighbor_commands[1][7:]]);delta=q3-c.aligned[[113,114,115],1,7:];adj=float(abs(np.diff(delta,axis=0)).max());second=float(abs(np.diff(delta,n=2,axis=0)).max());position=bool((command[7:]>=lo-1e-8).all() and (command[7:]<=hi+1e-8).all());source_ok=bool((v['errors']<=caps+1e-6).all());temporal=bool(adj<=.05+1e-7 and second<=.05+1e-7);passed=bool(position and source_ok and temporal and max(depths.values())<=.001);record=dict(kind='AFTER_new_native_position',queue_index=i,command=command,source_original_before_after_px=[caps,v['errors']],native_contacts=contacts,native_penetrations_m=depths,original_position_passed=position,original_source_nonregression_passed=source_ok,local_triplet_added_adjacent_rad=adj,local_triplet_added_second_rad=second,local_triplet_added_motion_passed=temporal,source_native_local_added_motion_passed=passed,whole_clock_root_velocity_swept_motor_grasp_unqualified=True);g.journal(journal,record);native.append(record)
 if passed:accepted.append(record)
c.q=original;journal.close();write('all-harmonic-domains.json',all_domains);write('all-new-native-position-records.json',native);write('accepted-new-local-branch-witnesses.json',accepted);out=dict(schema='v2d.axial-temporal-source-branch.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),harmonic_library_sha256=sha(S/'compute/harmonic_source_domains_v1.py'),input_manifest_sha256=sha(c.F/'manifest.json'),source_temporal_box_sha256=sha(selected_path),analytic_control_records=controls,index_source_sections=len(index_bank),thumb_source_sections=len(thumb_bank),middle_source_sections=len(middle_bank),frozen_branch_points=len(queue),new_native_points=len(native),local_source_native_added_motion_witnesses=len(accepted),all_domains_roots_states_and_native_outcomes_durable=True,finite_bc_sections_and_one_neighbor_box_not_global_exclusion=True,current_combined_reference_and_source_authority_unchanged=True,no_completed_optimizer_bank_graph_slice_or_source_fit_repeated=True,new_GPU_jobs=0,new_PPO_steps=0,no_native_compile_or_physics_step=True,whole_reference_or_task_qualified=False,all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);write('report.json',out);print(json.dumps(g.plain(out),indent=2))

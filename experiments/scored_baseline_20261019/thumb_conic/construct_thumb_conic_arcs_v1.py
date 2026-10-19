"""NEW analytic conic/2R boundary arcs, then fresh isolated native points.

This is not the prior failed conditional two-dimensional interval slice.
Roots partition the continuous image ellipse by actual joint/clearance bounds.
No absent results are reconstructed and no full reference is mutated.
"""
from pathlib import Path
import datetime,hashlib,json,math,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-thumb-conic-arcs-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import thumb_conic_geometry_v1 as g
import source_orientation_context_v1 as c
from scipy.spatial.transform import Rotation
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();P=S/'e037-thumb-conic-clearance-v1';report=json.loads((P/'report.json').read_text());assert report['full_original_proximal_domain_covered'] and len(report['analytic_controls'])==7
for n,h in report['all_outputs_sha256'].items():assert sha(P/n)==h
contract=json.loads((P/'source-conic-contract.json').read_text());clearance=json.loads((P/'all-compiled-parent-clearance-intervals.json').read_text());leaves=[r for r in clearance if r['classification']!='split'];base=np.array(contract['conditional_command']);frame=114;original=c.q.copy();lo,hi=c.bounds(frame);caps=np.array(contract['source_image_caps_px']);offset=np.array(contract['thumb_planar_offset']);l1,l2=contract['link_lengths_m'];pb=contract['original_proximal_bounds_rad'];db=contract['original_distal_bounds_rad'];H=np.array(contract['plane_camera_transform']);actual_R=Rotation.from_quat(base[3:7]).as_matrix();journal=(O/'all-root-arc-and-native-records.jsonl').open('w');arcs=[];roots_all=[];native_records=[];accepted=[];source_points=[]
def write(name,v):(O/name).write_text(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n')
def classify(b):
 rows=[r['classification'] for r in leaves if r['angle_interval'][0]-1e-12<=b<=r['angle_interval'][1]+1e-12]
 return 'compiled_parent_support_overlap_within_gate' if rows and all(x=='compiled_parent_support_overlap_within_gate' for x in rows) else ('compiled_parent_translation_penetration_over_gate' if rows and all(x=='compiled_parent_translation_penetration_over_gate' for x in rows) else 'unresolved_or_boundary')
for kind,key in [('original','original_ellipse'),('strict_subconic','strict_ellipse')]:
 ellipse=contract[key]
 if ellipse['kind']!='ellipse':g.journal(journal,dict(kind='unsupported_conic',family=kind,ellipse=ellipse));continue
 center=np.array(ellipse['center']);factor=np.array(ellipse['factor']);breaks=[0.,2*np.pi];circles=[]
 for distal in db:
  radius=float(np.sqrt(l1*l1+l2*l2+2*l1*l2*np.cos(distal)));circles.append(dict(kind='actual_distal_limit',angle_rad=distal,center=offset,radius=radius))
 blevels=sorted(set([*pb,*[v for row in leaves for v in row['angle_interval']]]))
 for proximal in blevels:
  C=offset+np.array([-l1*np.sin(proximal),l1*np.cos(proximal)]);circles.append(dict(kind='proximal_or_compiled_clearance_partition',angle_rad=proximal,center=C,radius=l2))
 for circle in circles:
  roots=g.ellipse_circle_roots(center,factor,np.array(circle['center']),circle['radius']);entry=dict(kind='conic_circle_root_equation',family=kind,circle=circle,roots=roots);roots_all.append(entry);g.journal(journal,entry);breaks.extend(roots['angles'])
 # Positive depth must hold over each retained arc, not just its midpoint.
 depth_constant=float(H[2]@np.r_[center,1.]);depth_cos=float(H[2,:2]@factor[:,0]);depth_sin=float(H[2,:2]@factor[:,1]);threshold=.01;amplitude=np.hypot(depth_cos,depth_sin)
 if amplitude>0 and abs((threshold-depth_constant)/amplitude)<=1:
  phase=np.arctan2(depth_sin,depth_cos);width=np.arccos(np.clip((threshold-depth_constant)/amplitude,-1,1));breaks.extend([float((phase-width)%(2*np.pi)),float((phase+width)%(2*np.pi))])
 breaks=sorted(breaks);unique=[]
 for t in breaks:
  if not unique or abs(t-unique[-1])>1e-9:unique.append(float(t))
 for left,right in zip(unique,unique[1:]):
  mid=(left+right)/2;xy=center+factor@np.array([np.cos(mid),np.sin(mid)]);inv=g.thumb_planar_inverse(xy,offset,(l1,l2),pb,db);depth_bounds=g.sinusoid_bounds(depth_cos,depth_sin,depth_constant,[left,right]);entry=dict(kind='continuous_conic_arc',family=kind,ellipse_parameter_interval_rad=[left,right],midpoint_parameter_rad=mid,planar_point=xy,actual_bounded_2R_branches=inv,depth_enclosure_m=[float(depth_bounds[0]),float(depth_bounds[1])],classification='outside_actual_joint_workspace' if not inv else ('unresolved_depth' if depth_bounds[0]<=.01 else 'actual_joint_workspace_boundary_arc'))
  if inv:entry['midpoint_parent_clearance_classes']=[classify(float(v[0])) for v in inv]
  arcs.append(entry);g.journal(journal,entry)
  if kind=='strict_subconic' and inv and depth_bounds[0]>.01:
   for branch in inv:
    cmd=base.copy();cmd[12:14]=branch;record=dict(family=kind,source_arc_index=len(arcs)-1,source_arc_parameter_rad=mid,command=cmd,planar_point=xy,parent_clearance_class=classify(float(branch[0])),original_source_image_caps_px=caps,strict_subconic_inward_margin_px=contract['constructive_inward_margin_px'],conditional_robot_pose_not_measured_source_truth=True);source_points.append(record)
# Freeze queue before examining physical outcomes; no source-pose winner.
queue=sorted(source_points,key=lambda x:float(np.sum((np.asarray(x['command'])[7:]-original[frame,1,7:])**2)));write('frozen-analytic-source-point-queue.json',queue);g.journal(journal,dict(kind='frozen_source_point_queue',points=len(queue),queue_sha256=sha(O/'frozen-analytic-source-point-queue.json')))
for index,proposal in enumerate(queue[:16]):
 if time.monotonic()-start>180:raise TimeoutError('Predeclared analytic-source point cap')
 if proposal['parent_clearance_class']=='compiled_parent_translation_penetration_over_gate':g.journal(journal,dict(kind='compiled_parent_bound_rejection',queue_index=index,proposal=proposal));continue
 cmd=np.array(proposal['command']);g.journal(journal,dict(kind='BEFORE_new_native_position',queue_index=index,proposal=proposal));c.q=original.copy();c.q[frame,1]=cmd;v=c.evaluate(frame,cmd[7:]);contacts=[];depths=dict(right_self=0.,external=0.,interhand=0.)
 for ct in c.data.contact[:c.data.ncon]:
  aa,bb=int(c.model.geom_bodyid[ct.geom1]),int(c.model.geom_bodyid[ct.geom2]);ra=c.bn[aa].startswith('dex3_right_');rb=c.bn[bb].startswith('dex3_right_')
  if not(ra or rb):continue
  other=c.bn[bb] if ra else c.bn[aa];category='right_self' if ra and rb else ('interhand' if other.startswith('dex3_left_') else 'external');depths[category]=max(depths[category],max(0.,-float(ct.dist)));contacts.append(dict(kind=category,bodies=[c.bn[aa],c.bn[bb]],native_geom_ids=[int(ct.geom1),int(ct.geom2)],signed_gap_m=float(ct.dist)))
 camera=c.cameras[frame];B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037';source=json.loads((B/f'source/hands/right/{frame:06d}.json').read_text());sourceR=c.stage@camera[:3,:3]@Rotation.from_rotvec(source['mano']['global_orient']).as_matrix()@np.diag([-1.,-1.,1.]);oldR=Rotation.from_quat(original[frame,1,3:7]).as_matrix();newR=Rotation.from_quat(cmd[3:7]).as_matrix();old_ori=float(Rotation.from_matrix(sourceR@oldR.T).magnitude());new_ori=float(Rotation.from_matrix(sourceR@newR.T).magnitude());rootres=Rotation.from_matrix(newR@Rotation.from_quat(c.aligned[frame,1,3:7]).as_matrix().T).as_rotvec();pos_ok=bool((cmd[7:]>=lo-1e-8).all() and (cmd[7:]<=hi+1e-8).all() and abs(rootres).max()<=.08+1e-9);source_ok=bool((v['errors']<=caps+1e-6).all() and new_ori<=old_ori+1e-9);passed=bool(pos_ok and source_ok and max(depths.values())<=.001);record=dict(kind='AFTER_new_native_position',queue_index=index,command=cmd,original_position_root_bounds_passed=pos_ok,source_image_before_after_px=[caps,v['errors']],source_model_orientation_before_after_rad=[old_ori,new_ori],original_source_nonregression_passed=source_ok,native_contacts=contacts,native_penetrations_m=depths,isolated_point_passed=passed,no_motion_motor_or_grasp_certificate=True);g.journal(journal,record);native_records.append(record)
 if passed:accepted.append(record)
c.q=original;journal.close();write('all-analytic-conic-arc-records.json',arcs);write('all-quartic-circle-root-records.json',roots_all);write('all-new-native-position-records.json',native_records);write('accepted-isolated-point-witnesses.json',accepted)
out=dict(schema='v2d.thumb-conic-arcs.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),library_sha256=sha(S/'compute/thumb_conic_geometry_v1.py'),conic_contract_sha256=sha(P/'source-conic-contract.json'),clearance_contract_sha256=sha(P/'all-compiled-parent-clearance-intervals.json'),input_manifest_sha256=sha(c.F/'manifest.json'),native_model_sha256=sha(c.MP),root_equations=len(roots_all),original_arcs=sum(r['family']=='original' for r in arcs),strict_subconic_arcs=sum(r['family']=='strict_subconic' for r in arcs),frozen_source_point_queue=len(queue),new_native_point_states=len(native_records),accepted_isolated_point_states=len(accepted),all_roots_arcs_proposals_and_native_outcomes_durable=True,whole_conic_boundary_partition_not_full_interior_global_exclusion=True,old_failed_slice_states_not_reconstructed_or_rerun=True,no_optimizer_bank_or_source_fit_repeated=True,current_combined_reference_unchanged=True,new_GPU_jobs=0,new_PPO_steps=0,no_native_compile_or_physics_step=True,whole_reference_or_task_qualified=False,all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);write('report.json',out);print(json.dumps(g.plain(out),indent=2))

"""NEW source-image enclosures on correlated temporal seven-joint boxes.

Root is the retained114 robot control. Axial thumb and both other digits vary
inside EACH correlated box; coordinate hulls are not treated as feasible.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-temporal-box-source-cones-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import thumb_conic_geometry_v1 as g
import coupled_articulation_intervals_v1 as b
import source_orientation_context_v1 as c
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.kinematics import URDFKinematics
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();P=S/'e037-axial-temporal-box-union-v1';report=json.loads((P/'report.json').read_text());path=P/'all-eligible-correlated-boxes.json';assert sha(path)==report['all_outputs_sha256'][path.name];boxes=json.loads(path.read_text());control=json.loads((P/'selected-correlated-box.json').read_text())['new114_root_control'];cmd=np.array(control['command']);frame=114;rootR=Rotation.from_quat(cmd[3:7]).as_matrix();camera=c.cameras[frame];A=camera[:3,:3].T@c.stage.T;off=camera[:3,:3].T@camera[:3,3];observed=c.uv((c.human[frame,1,[0,4,8,12]]-camera[:3,3])@camera[:3,:3]);contract=json.loads((S/'e037-thumb-conic-clearance-v1/source-conic-contract.json').read_text());caps=np.array(contract['source_image_caps_px']);journal=(O/'all-correlated-source-bounds.jsonl').open('w');rows=[];eligible=[];binding_pad_px=.0001
def thumb_box(v):
 a,prox,dist=v;total=b.add(prox,dist);x=b.sub(b.exact(-.0025),b.add(b.mul(b.exact(.0458),b.sine(prox)),b.mul(b.exact(.035),b.sine(total))));y=b.add(b.exact(.0193),b.add(b.mul(b.exact(.0458),b.cosine(prox)),b.mul(b.exact(.035),b.cosine(total))));return np.array([b.add(b.exact(.0255),b.mul(b.cosine(a),x)),y,b.mul(b.neg(b.sine(a)),x)])
for index,entry in enumerate(boxes):
 angles=np.array(entry['joint_angle_box_rad']);points=[thumb_box(angles[4:]),b.planar_point_box(angles[:2],.0285),b.planar_point_box(angles[2:4],-.0285)];pixels=[];low=[]
 for i,p in enumerate(points):
  native=b.add(b.exact(cmd[:3]),b.matvec(b.exact(rootR),p));C=b.sub(b.matvec(b.exact(A),native),b.exact(off));pixel=b.image_box(C,c.K);pixels.append(None if pixel is None else pixel);low.append(max(0.,b.disk_lower_bound(pixel,observed[i+1])-binding_pad_px))
 bad=[i+1 for i,v in enumerate(low) if v>caps[i+1]+1e-6];row=dict(box_index=index,neighbor113_index=entry['neighbor113_index'],neighbor115_index=entry['neighbor115_index'],joint_angle_box_rad=angles,source_pixel_enclosures=pixels,conservative_source_error_lower_bounds_px=low,source_original_caps_px=caps,interval_model_binding_outward_allowance_px=binding_pad_px,source_point_gate_tolerance_unchanged_px=1e-6,rejected_source_anchor_indices=bad,classification='necessary_source_image_exclusion' if bad else 'unresolved_or_constructively_possible',original_joint_distance_to_box_rad=entry['original_joint_distance_to_box_rad']);rows.append(row);g.journal(journal,row)
 if not bad:eligible.append(row)
journal.close();kin=URDFKinematics(c.F/'robot/urdfs/g1/dex3_right.urdf');tipnames=['right_thumb_tip','right_index_tip','right_middle_tip'];rng=np.random.default_rng(39819);indices=np.unique(np.linspace(0,len(rows)-1,min(16,len(rows)),dtype=int));sample_journal=(O/'all-new-URDF-enclosure-samples.jsonl').open('w');maximum_excess=0.;sample_count=0
for index in indices:
 row=rows[int(index)];box=np.array(row['joint_angle_box_rad'])
 for sample in range(8):
  angles=rng.uniform(box[:,0],box[:,1]);proposal=cmd.copy();proposal[7:]=angles;poses=kin.poses(proposal[:7],dict(zip(c.manifest['joint_names']['right'],angles,strict=True)));points=np.stack([poses[n][:3,3] for n in tipnames]);C=(points@c.stage-camera[:3,3])@camera[:3,:3];actual_pixels=c.uv(C);excess=0.
  for i,v in enumerate(row['source_pixel_enclosures']):
   if v is None:continue
   interval=np.array(v);excess=max(excess,float((interval[:,0]-actual_pixels[i]).max()),float((actual_pixels[i]-interval[:,1]).max()))
  errors=np.linalg.norm(actual_pixels-observed[1:],axis=1);lower=np.array(row['conservative_source_error_lower_bounds_px']);passed=bool(excess<=binding_pad_px and (errors>=lower-binding_pad_px).all());g.journal(sample_journal,dict(box_index=int(index),sample=sample,command=proposal,URDF_pixels=actual_pixels,URDF_source_errors_px=errors,maximum_enclosure_excess_px=excess,passed=passed));assert passed;maximum_excess=max(maximum_excess,excess);sample_count+=1
sample_journal.close();selected=min(eligible,key=lambda r:(r['original_joint_distance_to_box_rad'],r['neighbor113_index'],r['neighbor115_index'])) if eligible else None
for name,value in [('all-source-domain-boxes.json',rows),('all-unexcluded-source-domain-boxes.json',eligible),('selected-source-temporal-box.json',None if selected is None else dict(source_bound=selected,correlated_box=boxes[selected['box_index']],root_control=control,source_pose_not_selected_by_physical_outcome=True,conditional_finite_neighbors_not_measured_mandatory_nodes=True))]:
 (O/name).write_text(json.dumps(g.plain(value),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.temporal-box-source-cones.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),input_manifest_sha256=sha(c.F/'manifest.json'),saved_correlated_boxes_sha256=sha(path),conditional_boxes=len(boxes),excluded_boxes=len(boxes)-len(eligible),unexcluded_unresolved_boxes=len(eligible),rejected_source_anchor_counts={str(i):sum(i in r['rejected_source_anchor_indices'] for r in rows) for i in [1,2,3]},new_independent_URDF_source_enclosure_samples=sample_count,maximum_sampled_enclosure_excess_px=maximum_excess,source_point_gate_not_relaxed=True,all_seven_joint_ranges_continuous_and_correlated=True,numerical_bounds_not_formal_global_exclusion=True,current_reference_and_source_authority_unchanged=True,no_old_source_bank_graph_native_control_or_fit_repeated=True,new_GPU_jobs=0,new_PPO_steps=0,no_native_or_physics_steps=True,whole_reference_or_task_qualified=False,all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out),indent=2))

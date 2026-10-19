"""Independent saved-box cover, actual URDF image enclosure and residual audit."""
from pathlib import Path
import collections,datetime,hashlib,json,math,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');O=S/'e037-coupled-articulation-independent-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.kinematics import URDFKinematics
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();P=S/'e037-coupled-articulation-domain-v1';rep=json.loads((P/'report.json').read_text());assert sha(P/'all-domain-nodes-and-states.json')==rep['all_nodes_sha256'];nodes=json.loads((P/'all-domain-nodes-and-states.json').read_text());indexed={n['id']:n for n in nodes};leaf_ids=[]
for node in nodes:
 if node['classification']=='split':
  parent=np.array(node['box']);a,bb=[np.array(indexed[i]['box']) for i in node['children']];axis=rep['variable_names'].index(node['split_axis']);mask=np.arange(8)!=axis;assert np.array_equal(a[mask],parent[mask]) and np.array_equal(bb[mask],parent[mask]);assert a[axis,0]==parent[axis,0] and a[axis,1]==bb[axis,0] and bb[axis,1]==parent[axis,1];assert all(indexed[i]['parent']==node['id'] for i in node['children'])
 else:leaf_ids.append(node['id'])
assert len(nodes)==2*len(leaf_ids)-1 and len(nodes)==rep['nodes'];known={0}
for node in nodes:
 assert node['id'] in known
 if node['classification']=='split':known.update(node['children'])
assert known==set(indexed)
frame=114;kin=URDFKinematics(c.F/'robot/urdfs/g1/dex3_right.urdf');camera=c.cameras[frame];stage=c.stage;B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037';source=json.loads((B/f'source/hands/right/{frame:06d}.json').read_text());sourceR=stage@camera[:3,:3]@Rotation.from_rotvec(source['mano']['global_orient']).as_matrix()@np.diag([-1.,-1.,1.]);alignR=Rotation.from_quat(c.aligned[frame,1,3:7]).as_matrix();base=c.q[frame,1].copy();upstream=np.array(rep['fixed_upstream_thumb_pair']);caps=np.array(rep['source_image_caps_px']);observed=c.uv((c.human[frame,1,[0,4,8,12]]-camera[:3,3])@camera[:3,:3]);names=c.manifest['joint_names']['right'];tipnames=['right_hand_palm_link','right_thumb_tip','right_index_tip','right_middle_tip'];rng=np.random.default_rng(7821);unresolved=[n for n in nodes if n['classification'].startswith('unresolved')];rejected=[n for n in nodes if n['classification']=='rejected'];selected=[*unresolved,*[rejected[i] for i in np.linspace(0,len(rejected)-1,min(32,len(rejected)),dtype=int)]];sample_records=[];max_excess=0.;source_rejection_counts=collections.Counter()
for node in selected:
 box=np.array(node['box'])
 for mode in ['midpoint','random']:
  x=box.mean(axis=1) if mode=='midpoint' else rng.uniform(box[:,0],box[:,1]);r=x[:3];length=np.linalg.norm(r);rootR=Rotation.from_rotvec(2*np.arctan(length)*r/max(length,1e-300)).as_matrix()@sourceR;cmd=base.copy();cmd[3:7]=Rotation.from_matrix(rootR).as_quat();cmd[7:11]=x[4:];cmd[11:13]=upstream;cmd[13]=x[3];poses=kin.poses(cmd[:7],dict(zip(names,cmd[7:],strict=True)));points=np.stack([poses[n][:3,3] for n in tipnames]);C=(points@stage-camera[:3,3])@camera[:3,:3];pixels=c.uv(C);errors=np.linalg.norm(pixels-observed,axis=1);residual=Rotation.from_matrix(rootR@alignR.T).as_rotvec();bounds=node.get('root_rotvec_enclosure');violations=[]
  if bounds is not None:
   v=np.array(bounds);excess=float(max((v[:,0]-residual).max(),(residual-v[:,1]).max()));max_excess=max(max_excess,excess);assert excess<1e-10
  if node.get('source_pixel_enclosures') is not None:
   for i,p in enumerate(node['source_pixel_enclosures'],1):
    if p is None:continue
    v=np.array(p);excess=float(max((v[:,0]-pixels[i]).max(),(pixels[i]-v[:,1]).max()));max_excess=max(max_excess,excess);assert excess<1e-4,(node['id'],i,excess)
   lower=np.array(node['source_image_lower_bounds_px']);assert (errors[1:]>=lower-1e-4).all()
  source_rejection_counts.update(str(i) for i in np.flatnonzero(errors>caps+1e-6));sample_records.append(dict(node_id=node['id'],mode=mode,parameters=x.tolist(),command=cmd.tolist(),URDF_camera_points=C.tolist(),URDF_source_errors_px=errors.tolist(),root_world_residual_rad=residual.tolist(),source_model_orientation_error_rad=float(Rotation.from_matrix(sourceR@rootR.T).magnitude()),saved_box_exclusion=node.get('rejection'),saved_box_classification=node['classification']))
all_proposals=[dict(node_id=n['id'],**p) for n in nodes for p in n.get('proposed_points',[])];best=sorted(all_proposals,key=lambda r:r['merit'])[:16];part=O/'all-independent-URDF-enclosure-samples.json';part.write_text(json.dumps(sample_records,indent=2)+'\n');bestpath=O/'best-retained-proposals-diagnostic-only.json';bestpath.write_text(json.dumps(best,indent=2)+'\n');unpath=O/'all-unresolved-domains.json';unpath.write_text(json.dumps(unresolved,separators=(',',':'))+'\n')
out=dict(schema='v2d.coupled-articulation-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),study_report_sha256=sha(P/'report.json'),all_saved_nodes_sha256=sha(P/'all-domain-nodes-and-states.json'),complete_binary_partition_preserved=True,nodes=len(nodes),leaves=len(leaf_ids),unresolved_boxes=len(unresolved),independent_actual_URDF_projection_samples=len(sample_records),maximum_sampled_enclosure_excess=max_excess,all_sampled_enclosures_passed=True,independent_samples_sha256=sha(part),best_diagnostic_only_sha256=sha(bestpath),unresolved_domains_sha256=sha(unpath),sampled_source_anchor_violations=dict(source_rejection_counts),best_recorded_merit=best[0]['merit'],best_proposal_source_errors_px=best[0]['source_errors_px'],source_caps_px=caps.tolist(),all_proposals_remain_rejected_not_promoted=True,continuous_budget_remaining_not_global_exclusion=True,current_reference_unchanged=True,no_new_native_physics_or_GPU=True,whole_reference_task_or_policy_qualified=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

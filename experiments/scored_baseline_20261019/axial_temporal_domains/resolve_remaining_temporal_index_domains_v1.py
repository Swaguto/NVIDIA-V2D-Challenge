"""NEW original-tolerance resolution of139 previously unverified pair domains.

The173 existing separator certificates are reused by hash, never recomputed.
Original numeric tolerances produce necessary OUTER boxes only; no candidate
gate is relaxed and no point/native success is inferred from them.
"""
from pathlib import Path
import datetime,hashlib,json,math,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-remaining-temporal-index-domains-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import thumb_conic_geometry_v1 as g
import source_orientation_context_v1 as c
from planar_support_critical_points_v1 import point,support
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.kinematics import URDFKinematics
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();old=S/'e037-temporal-index-separators-v1';orep=json.loads((old/'report.json').read_text());certpath=old/'all-independent-separator-certificates.json';assert sha(certpath)==orep['all_outputs_sha256'][certpath.name];oldcerts=json.loads(certpath.read_text());known={(r['neighbor113_index'],r['neighbor115_index']):r for r in oldcerts};assert len(known)==173;P=S/'e037-conic-temporal-middle-boxes-v1';preport=json.loads((P/'report.json').read_text());recordpath=P/'all-neighbor-pair-boxes.jsonl';assert sha(recordpath)==preport['all_outputs_sha256'][recordpath.name];rows=[json.loads(v) for v in recordpath.read_text().splitlines()];rows=[r for r in rows if r['new_control_index']==0];assert len(rows)==500
T=S/'e037-temporal-index-conic-domains-v1';line=json.loads((T/'all-controls-domains-and-witnesses.jsonl').read_text().splitlines()[4]);Q=np.array(line['Q']);ellipse=line['ellipse'];center=np.array(ellipse['center']);factor=np.array(ellipse['factor']);A=Q[:2,:2];control=json.loads((S/'e037-thumb-conic-independent-v1/selected-conditional-robot-control.json').read_text());base=np.array(control['command']);rootR=Rotation.from_quat(base[3:7]).as_matrix();frame=114;camera=c.cameras[frame];observed=c.uv((c.human[frame,1,[8]]-camera[:3,3])@camera[:3,:3])[0];kin=URDFKinematics(c.F/'robot/urdfs/g1/dex3_right.urdf');journal=(O/'all-tolerance-domain-and-new-plane-records.jsonl').open('w');rng=np.random.default_rng(93710);records=[];counts=dict(necessary_outer_joint_box_empty=0,reused_existing_source_separator=0,new_source_separator=0,unresolved=0);minimum_margin=math.inf;max_closure=0.;samples=0;motion_pad=1e-7;binding_guard=1e-8
for row in rows:
 key=(row['neighbor113_index'],row['neighbor115_index']);lower=np.array(row['lower_original_correction_rad'])-motion_pad;upper=np.array(row['upper_original_correction_rad'])+motion_pad;absolute=np.c_[c.aligned[frame,1,7:]+lower,c.aligned[frame,1,7:]+upper];entry=dict(neighbor113_index=key[0],neighbor115_index=key[1],necessary_outer_joint_box_rad=absolute,original_motion_tolerance_rad=motion_pad,outer_box_not_a_candidate_gate=True)
 if (lower>upper).any():entry['classification']='necessary_outer_joint_box_empty';entry['still_empty_joint_indices']=np.flatnonzero(lower>upper);counts[entry['classification']]+=1;records.append(entry);g.journal(journal,entry);continue
 if key in known:
  certificate=known[key];assert np.allclose(absolute[:2],np.array(certificate['outer_index_box_rad']),atol=2e-15,rtol=0);assert certificate['numerically_separated'];entry.update(classification='reused_existing_source_separator',prior_separator_input_sha256=sha(certpath),prior_plane=certificate['plane'],old_source_domains_not_recomputed=True);minimum_margin=min(minimum_margin,certificate['plane']['separation_margin_m']);counts[entry['classification']]+=1;records.append(entry);g.journal(journal,entry);continue
 box=absolute[:2]
 if not 0<box[1,0]<=box[1,1]<np.pi:entry['classification']='unresolved';entry['reason']='positive branch chart not established';counts['unresolved']+=1;records.append(entry);g.journal(journal,entry);continue
 directions=[]
 for a in [box[0,0],box[0].mean(),box[0,1]]:
  for b in [box[1,0],box[1].mean(),box[1,1]]:
   n=A@(point(a,b)-center)
   if np.linalg.norm(n)>1e-15:directions.append(n/np.linalg.norm(n))
 candidates=[]
 for n in directions:
  extremum,critical=support(n,box[0],box[1]);es=float(n@center+np.linalg.norm(factor.T@n));candidates.append(dict(direction=n,workspace_minimum=extremum,all_boundary_critical_points=critical,ellipse_maximum_support=es,metric_binding_outward_guard_m=binding_guard,separation_margin_m=extremum['projection']-es-binding_guard))
 best=max(candidates,key=lambda e:e['separation_margin_m']);separated=best['separation_margin_m']>1e-12;entry.update(classification='new_source_separator' if separated else 'unresolved',plane=best,other_tested_directions=candidates);checks=[]
 for sample in range(3):
  angles=rng.uniform(box[:,0],box[:,1]);cmd=base.copy();cmd[7:9]=angles;poses=kin.poses(cmd[:7],dict(zip(c.manifest['joint_names']['right'],cmd[7:],strict=True)));native_point=poses['right_index_tip'][:3,3];local=rootR.T@(native_point-cmd[:3])-np.array([.0777,-.0016,.0285]);closure=float(abs(local[:2]-point(*angles)).max());max_closure=max(max_closure,closure);C=(native_point@c.stage-camera[:3,3])@camera[:3,:3];error=float(np.linalg.norm(c.uv(C[None])[0]-observed));projection=float(best['direction']@local[:2]);passed=bool(closure<binding_guard and projection>=best['workspace_minimum']['projection']-binding_guard);record=dict(command=cmd,planar_URDF_point=local,source_error_px=error,model_closure_m=closure,passed=passed);checks.append(record);samples+=1;assert passed
 entry['new_independent_URDF_checks']=checks;minimum_margin=min(minimum_margin,best['separation_margin_m']);counts[entry['classification']]+=1;records.append(entry);g.journal(journal,entry)
journal.close();assert len(records)==500;(O/'all-complete-conditional-domain-records.json').write_text(json.dumps(g.plain(records),allow_nan=False,indent=2)+'\n');out=dict(schema='v2d.remaining-temporal-index-domains.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),input_manifest_sha256=sha(c.F/'manifest.json'),saved_pair_records_sha256=sha(recordpath),reused_certificate_sha256=sha(certpath),all_finite_neighbor_pairs=500,classifications=counts,all_finite_conditional_domains_rejected_numerically=counts['unresolved']==0,minimum_recorded_source_plane_separation_m=minimum_margin,new_independent_actual_URDF_checks=samples,maximum_new_URDF_planar_closure_m=max_closure,old173domains_and_their_controls_not_recomputed=True,original_numeric_motion_tolerance_explicit_necessary_outer_box_only=True,actual_source_position_motion_gates_unchanged=True,no_global_or_free_root_or_continuous_neighbor_unreachability_claim=True,all_new_domain_witnesses_and_unresolved_records_durable=True,current_three_component_reference_unchanged=True,no_native_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_task_qualified=False,all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out),indent=2))

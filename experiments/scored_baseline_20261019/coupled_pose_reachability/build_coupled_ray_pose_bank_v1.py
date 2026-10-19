"""NEW finite pose bank: thumb configuration + ray-bound common root rotation."""
from pathlib import Path
import datetime,hashlib,itertools,json,math,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');O=S/'e037-coupled-ray-pose-bank-v1';sys.path.insert(0,str(S/'compute'));start=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
import source_orientation_context_v1 as c
np=c.np
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.planar_thumb_ik import PlanarThumbChain,thumb_point
from v2d_dex3_flashchord.source_ray_kinematics import ray_bound_root_rotations
LP=S/'e037-coupled-seed-temporal-compatibility-v1/report.json';linear=json.loads(LP.read_text());assert linear['necessary_finger_motion_compatible'];lib=json.loads((S/'e037-ray-bound-root-library-v1/report.json').read_text());assert lib['new_library_controls_passed'];isolated_path=S/'e037-source-coupled-isolated-independent-v1/report.json';isolated=json.loads(isolated_path.read_text());assert isolated['all_two_isolated_source_self_bounds_prerequisites_passed'];seed_path=S/'e037-source-orientation-coupled-fingers-v1/report.json';seed_report=json.loads(seed_path.read_text());assert sha(seed_path)==isolated['seed_report_sha256'];seeds={r['frame']:np.array(r['command']) for r in seed_report['candidate_commands']};prior_self_path=S/'e037-thumb-ray-independent-validation-v1/fresh-candidate-self-contacts.json';prior_self=json.loads(prior_self_path.read_text());assert sha(c.F/'manifest.json')==isolated['input_manifest_sha256']==linear['input_manifest_sha256']
original=c.q.copy();N=len(original);chain=PlanarThumbChain((.0255,0,0),(-.0025,.0193),.0458,.035);B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037';source_hashes={r['path']:r['sha256'] for r in json.loads((B/'manifest.json').read_text())['sources']};ranges={(r['frame'],r['joint']):(r['minimum'],r['maximum']) for r in linear['ranges']}
with np.load(S/'e037-coupled-seed-temporal-compatibility-v1/scalar-motion-witness-not-reference.npz') as z:witness=z['right_joint_angles'].copy()
def planar_branches(point,lo,hi):
 x,y=point;cosine=(x*x+y*y-.0458**2-.035**2)/(2*.0458*.035)
 if abs(cosine)>1+1e-12:return []
 out=[]
 for b in [math.acos(np.clip(cosine,-1,1)),-math.acos(np.clip(cosine,-1,1))]:
  a=math.atan2(y,x)-math.atan2(.035*math.sin(b),.0458+.035*math.cos(b))
  for aa,bb in itertools.product([a-2*math.pi,a,a+2*math.pi],[b-2*math.pi,b,b+2*math.pi]):
   v=np.array([aa,bb])
   if (v>=lo-1e-9).all() and (v<=hi+1e-9).all() and not any(abs(v-z).max()<1e-10 for z in out):out.append(v)
 return out
def contacts():
 rows=[]
 for ct in c.data.contact[:c.data.ncon]:
  ga,gb=int(ct.geom1),int(ct.geom2)
  if ga in c.rgeoms and gb in c.rgeoms:rows.append(dict(bodies=[c.bn[int(c.model.geom_bodyid[ga])],c.bn[int(c.model.geom_bodyid[gb])]],signed_gap_m=float(ct.dist)))
 return rows
def thumb_palm_depth():
 rows=contacts();eligible=[r for r in rows if all(('thumb' in n or 'palm' in n) for n in r['bodies'])]
 return max([max(0.,-r['signed_gap_m']) for r in eligible],default=0.)
bank=[];rows=[];candidate_counter=0
for frame in range(102,120):
 if time.monotonic()-start>210:raise TimeoutError('Ray-bound finite bank wall cap')
 c.q=original.copy();base=original[frame,1].copy();before=c.evaluate(frame,base[7:]);alignedR=Rotation.from_quat(c.aligned[frame,1,3:7]).as_matrix();baseR=Rotation.from_quat(base[3:7]).as_matrix();eligible=[];seen=set();counts=dict(thumb_configurations=0,thumb_palm_rejected=0,sphere_branches=0,antiparallel_branches=0,root_bound_or_orientation_rejected=0,planar_digit_rejected=0,full_native_evaluations=0,source_rejected=0,self_rejected=0,duplicate_eligible=0)
 def store(command,errors,self_contacts,association):
  key=tuple(np.round(np.r_[Rotation.from_matrix(Rotation.from_quat(command[3:7]).as_matrix()@alignedR.T).as_rotvec(),command[7:]],10))
  if key in seen:counts['duplicate_eligible']+=1;return
  seen.add(key);eligible.append(dict(frame=frame,command=command.tolist(),source_image_errors_px=list(map(float,errors)),native_self_contacts=self_contacts,maximum_self_penetration_m=max([max(0.,-r['signed_gap_m']) for r in self_contacts],default=0.),association=association))
 if frame<104 or frame>=118:
  depth=max(0.,float(-before['gaps'].min()));assert depth<=.001;store(base,before['errors'],[dict(bodies=r['native_bodies'],signed_gap_m=r['signed_gap_m']) for r in before['self_contacts']],dict(kind='unchanged_temporal_boundary'))
 elif frame in seeds:
  # Reuse saved independent evidence; do not repeat the old two-pose audit.
  r=next(r for r in isolated['rows'] if r['frame']==frame);assert r['isolated_source_self_bounds_prerequisites_passed'];store(seeds[frame],r['source_image_before_after_px'][1],r['self_contacts'],dict(kind='retained_converged_seed',verification_report_sha256=sha(isolated_path)))
 else:
  p=B/f'source/hands/right/{frame:06d}.json';assert sha(p)==source_hashes[str(p.relative_to(B))];record=json.loads(p.read_text());assert record['directly_observed'] and record['visual_valid'] and before['source_mask'].all();sourceR=c.stage@c.cameras[frame,:3,:3]@Rotation.from_rotvec(record['mano']['global_orient']).as_matrix()@np.diag([-1.,-1.,1.]);ori_before=Rotation.from_matrix(sourceR@baseR.T).magnitude();camera=c.cameras[frame];A=camera[:3,:3].T@c.stage.T;offset=camera[:3,:3].T@camera[:3,3];origin_native=A.T@offset
  c.set_state(frame);oldpoints=c.data.xpos[c.tipids[1:]].copy();current_pixels=c.uv(np.einsum('ij,nj->ni',A,oldpoints,optimize=False)-offset);observed=c.uv(np.einsum('ni,ij->nj',c.human[frame,1,[4,8,12]]-camera[:3,3],camera[:3,:3],optimize=False));distance=np.linalg.norm(current_pixels-observed,axis=1);margin=np.minimum(.0001,.01*distance);pixels=current_pixels+(observed-current_pixels)*(margin/np.maximum(distance,1e-12))[:,None]
  lo=np.array([ranges[(frame,n)][0] for n in c.manifest['joint_names']['right']]);hi=np.array([ranges[(frame,n)][1] for n in c.manifest['joint_names']['right']]);position_lo,position_hi=c.bounds(frame);lo=np.maximum(lo,position_lo);hi=np.minimum(hi,position_hi)
  if (base[7:]>=lo-1e-9).all() and (base[7:]<=hi+1e-9).all() and max(0.,float(-before['gaps'].min()))<=.001:store(base,before['errors'],[dict(bodies=r['native_bodies'],signed_gap_m=r['signed_gap_m']) for r in before['self_contacts']],dict(kind='unchanged_valid_current_pose'))
  levels=[]
  for j in range(4,7):
   values=list(np.linspace(lo[j],hi[j],5))+[float(witness[frame,j])]
   if lo[j]<=base[7+j]<=hi[j]:values.append(float(base[7+j]))
   levels.append(sorted(set(round(v,13) for v in values)))
  thumb_ray_camera=np.r_[(pixels[0]-[c.K[0,2],c.K[1,2]])/[c.K[0,0],c.K[1,1]],1.];thumb_ray_native=A.T@thumb_ray_camera
  for thumb_angles in itertools.product(*levels):
   counts['thumb_configurations']+=1
   if time.monotonic()-start>215:raise TimeoutError('Ray-bound finite bank evaluation cap')
   angles=base[7:].copy();angles[4:]=thumb_angles;c.q=original.copy();c.set_state(frame,angles)
   if thumb_palm_depth()>.001+1e-9:counts['thumb_palm_rejected']+=1;continue
   point=thumb_point(chain,thumb_angles)
   for convention,prior in [('closest_current_rotation',baseR),('closest_source_model_rotation',sourceR)]:
    solutions=ray_bound_root_rotations(point,base[:3],origin_native,thumb_ray_native,prior,(.0100001,np.inf));counts['sphere_branches']+=len(solutions.branches);counts['antiparallel_branches']+=solutions.antiparallel_alignment_count
    for branch in solutions.branches:
     rootR=branch.rotation;relative=Rotation.from_matrix(rootR@alignedR.T).as_rotvec();ori=Rotation.from_matrix(sourceR@rootR.T).magnitude()
     if abs(relative).max()>.08+1e-9 or ori>ori_before+1e-9:counts['root_bound_or_orientation_rejected']+=1;continue
     digitbanks=[]
     for di,(z,cols) in enumerate([(.0285,[0,1]),(-.0285,[2,3])],start=1):
      ray=np.r_[(pixels[di]-[c.K[0,2],c.K[1,2]])/[c.K[0,0],c.K[1,1]],1.];local_origin=rootR.T@(origin_native-base[:3]);local_ray=rootR.T@(A.T@ray)
      if abs(local_ray[2])<1e-10:digitbanks.append([]);continue
      depth=(z-local_origin[2])/local_ray[2];local=local_origin+depth*local_ray;digitbanks.append(planar_branches(local[:2]-[.0777,-.0016],lo[cols],hi[cols]) if depth>.01 else [])
     if any(not b for b in digitbanks):counts['planar_digit_rejected']+=1;continue
     for index,middle in itertools.product(*digitbanks):
      command=base.copy();command[3:7]=Rotation.from_matrix(rootR).as_quat();command[7:9]=index;command[9:11]=middle;command[11:]=thumb_angles;c.q[frame,1]=command;v=c.evaluate(frame,command[7:]);counts['full_native_evaluations']+=1
      if not (v['errors']<=before['errors']+1e-6).all():counts['source_rejected']+=1;continue
      if max(0.,float(-v['gaps'].min()))>.001+1e-9:counts['self_rejected']+=1;continue
      store(command,v['errors'],[dict(bodies=r['native_bodies'],signed_gap_m=r['signed_gap_m']) for r in v['self_contacts']],dict(kind='thumb_configuration_ray_bound_common_root',twist_convention=convention,hypothesized_camera_depth_m=branch.parameter,source_record_sha256=sha(p),source_orientation_error_rad=float(ori)))
 for node in eligible:node['bank_index']=len(bank);bank.append(node)
 indices=[n['bank_index'] for n in eligible]
 if len(indices)>48:
  values=np.array([np.r_[Rotation.from_matrix(Rotation.from_quat(n['command'][3:7]).as_matrix()@baseR.T).as_rotvec()/.08,(np.array(n['command'][7:])-witness[frame])/.25] for n in eligible]);chosen=[int(np.argmin(np.sum(values**2,axis=1)))];nearest=np.sum((values-values[chosen[0]])**2,axis=1)
  baseline=next((i for i,n in enumerate(eligible) if n['association']['kind']=='unchanged_valid_current_pose'),None)
  if baseline is not None and baseline not in chosen:chosen.append(baseline);nearest=np.minimum(nearest,np.sum((values-values[baseline])**2,axis=1))
  while len(chosen)<48:
   nearest[chosen]=-1;i=int(np.argmax(nearest));chosen.append(i);nearest=np.minimum(nearest,np.sum((values-values[i])**2,axis=1))
  selected=[indices[i] for i in chosen]
 else:selected=indices
 row=dict(frame=frame,source_direct=bool(c.flags['hand_directly_observed'][frame,1]),source_metric_frame_flag=bool(c.flags['hand_metric_valid'][frame,1]),before_source_errors_px=before['errors'].tolist(),counts=counts,eligible_nodes=len(indices),selected_nodes=len(selected),selected_bank_indices=selected);rows.append(row);print(json.dumps(row),flush=True)
c.q=original;bank_path=O/'pose-bank.json';bank_path.write_text(json.dumps(bank,indent=2)+'\n');report=dict(schema='v2d.coupled-ray-root-finger-bank.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),context_sha256=sha(S/'compute/source_orientation_context_v1.py'),input_manifest_sha256=sha(c.F/'manifest.json'),ray_root_library_sha256=lib['library_sha256'],necessary_temporal_report_sha256=sha(LP),isolated_seed_verification_sha256=sha(isolated_path),pose_bank_sha256=sha(bank_path),method='Five scalar-viability levels plus witness/current thumb angles; exact sphere/ray root mapping with two geometric twist conventions; exact planar image-ray other fingers; actual native self/source/original boxes; deterministic geometric48node cover before pair-state reachability',rows=rows,total_eligible_nodes=len(bank),all_frames_have_selected_nodes=all(r['selected_nodes'] for r in rows),finite_bank_and_twist_conventions_not_global_reachability=True,no_scalar_LP_witness_promoted_to_source_or_collision_truth=True,source_model_orientation_not_measured_robot_pose=True,hypothesized_depth_not_observed_human_depth_or_contact=True,all_three_components_source_flags_objects_and_native_model_unchanged=True,no_old_source_fits_or_completed_trajectory_controls_repeated=True,no_native_compile_or_physics_step=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_task_or_policy_qualified=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(total_eligible_nodes=len(bank),all_frames_have_nodes=report['all_frames_have_selected_nodes'],wall_seconds=report['wall_seconds']),indent=2))

"""NEW finite native-hand facet atlas; no local wrist/finger optimization."""
from pathlib import Path
import datetime,hashlib,itertools,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();start=time.monotonic();O=S/'e037-compiled-contact-reachability-atlas-v1';plan_path=S/'compute/compiled-contact-reachability-CPU-plan-v1.json';plan=json.loads(plan_path.read_text())
# Exact immutable setup only, excluding every old iteration/optimization loop.
setup=S/'compute/solve_compiled_thumb_index_mode_v2.py';previous=json.loads((S/'e037-compiled-thumb-index-mode-v2/report.json').read_text());assert sha(setup)==previous['producer_sha256'];prefix=setup.read_text().split("for proposal in bank['contact_mode_hypotheses']:")[0];assert 'for iteration in range' not in prefix;ns={'__file__':str(setup)};exec(compile(prefix,str(setup),'exec'),ns)
np=ns['np'];from scipy.spatial import cKDTree
from trimesh.triangles import closest_point
model=ns['model'];data=ns['data'];T=ns['T'];commands=ns['commands'];kin=ns['kin'];names=ns['joint_names'];bank=ns['bank'];patches=ns['patches'];q0=commands[1].copy();translation_box=np.array([[-.008,.008]]*3);gap=plan['target_signed_gap_m'];patch_world={}
for key,patch in patches.items():
 tri=np.einsum('nvi,ji->nvj',patch.triangles_local_m,T[:3,:3],optimize=False)+T[:3,3];normal=np.einsum('ni,ji->nj',patch.outward_normals_local,T[:3,:3],optimize=False);patch_world[key]=(tri,normal,cKDTree(tri.mean(1)))
rows={};summary={};query_count=0;necessary_facet_pairs=0
for digit in ['thumb','index','middle']:
 active=np.array([i for i,n in enumerate(names) if '_hand_'+digit+'_' in n]);limits=np.array([kin.limits[names[i]] for i in active]);levels=[np.linspace(lo,hi,plan['joint_grid_levels']) for lo,hi in limits];samples=list(itertools.product(*levels));current=tuple(q0[7+active]);samples.append(current);records={key:[] for key in patches};best_per_configuration={}
 for sample_id,values in enumerate(samples):
  if time.monotonic()-start>220:raise TimeoutError('Finite atlas cap')
  q=q0.copy();q[7+active]=values;ns['set_state'](q);best={};joint_cost=float(np.sum(((np.asarray(values)-current)/(limits[:,1]-limits[:,0]))**2))
  for g in ns['right_geoms']:
   body=int(model.geom_bodyid[g])
   if ns['digit_of'](ns['bn'][body])!=digit:continue
   tri,normal=ns['meshes'][g];G=data.geom_xmat[g].reshape(3,3);H=np.einsum('nvi,ji->nvj',tri,G,optimize=False)+data.geom_xpos[g];hn=np.einsum('ni,ji->nj',normal,G,optimize=False);centers=H.mean(1)
   for key,(B,on,tree) in patch_world.items():
    query_count+=len(H);_,nearest=tree.query(centers,k=min(plan['object_centroid_neighbors_per_hand_facet'],len(B)));nearest=np.asarray(nearest).reshape(len(H),-1);hi=np.repeat(np.arange(len(H)),nearest.shape[1]);oi=nearest.reshape(-1);facing=np.einsum('ni,ni->n',hn[hi],on[oi],optimize=False)<=-.5;hi,oi=hi[facing],oi[facing]
    if not len(hi):continue
    lower=B[oi].min(1)-H[hi].max(1)+gap*on[oi];upper=B[oi].max(1)-H[hi].min(1)+gap*on[oi];feasible=(np.maximum(lower,translation_box[:,0])<=np.minimum(upper,translation_box[:,1])).all(1);hi,oi,lower,upper=hi[feasible],oi[feasible],lower[feasible],upper[feasible];necessary_facet_pairs+=len(hi)
    if not len(hi):continue
    object_points=closest_point(B[oi],centers[hi]);preferred=object_points-centers[hi]+gap*on[oi];bound_violation=np.maximum(abs(preferred)-.008,0).max(1);score=np.sum((preferred/.008)**2,axis=1)+plan['joint_prior_score_weight']*joint_cost+1000*bound_violation;pick=int(np.argmin(score));hidx=int(hi[pick]);oidx=int(oi[pick]);record=dict(digit=digit,patch=key,grid_sample=sample_id,active_joint_indices=active.tolist(),active_joint_values_rad=list(values),hand_geom=int(g),hand_body=ns['bn'][body],hand_facet=hidx,object_patch_facet=oidx,hand_triangle_world_m=H[hidx].tolist(),hand_outward_normal_world=hn[hidx].tolist(),object_triangle_world_m=B[oidx].tolist(),object_outward_normal_world=on[oidx].tolist(),necessary_translation_lower_m=lower[pick].tolist(),necessary_translation_upper_m=upper[pick].tolist(),preferred_translation_m=preferred[pick].tolist(),rank_score=float(score[pick]),sampled_joint_prior_squared_norm=joint_cost)
    if key not in best or record['rank_score']<best[key]['rank_score']:best[key]=record
  for key,r in best.items():records[key].append(r)
 rows[digit]={key:sorted(records[key],key=lambda a:(a['rank_score'],a['grid_sample'],a['hand_geom'],a['hand_facet'],a['object_patch_facet']))[:plan['retained_configurations_per_digit_patch']] for key in patches};summary[digit]=dict(active_joint_indices=active.tolist(),joint_limits_rad=limits.tolist(),grid_levels=plan['joint_grid_levels'],sampled_joint_configurations=len(samples),eligible_distinct_configurations_per_patch={k:len(v) for k,v in records.items()},retained_configurations_per_patch={k:len(v) for k,v in rows[digit].items()})
queue=[];pair_counts={}
for mode,other in [('thumb_index','index'),('thumb_middle','middle')]:
 candidates=[]
 for thumb_pool in rows['thumb'].values():
  for a in thumb_pool:
   for other_pool in rows[other].values():
    for b in other_pool:
     if np.dot(a['object_outward_normal_world'],b['object_outward_normal_world'])>-.5:continue
     lo=np.maximum(np.maximum(a['necessary_translation_lower_m'],b['necessary_translation_lower_m']),translation_box[:,0]);hi=np.minimum(np.minimum(a['necessary_translation_upper_m'],b['necessary_translation_upper_m']),translation_box[:,1])
     if (lo>hi).any():continue
     score=float(np.sum(((np.asarray(a['preferred_translation_m'])-b['preferred_translation_m'])/.008)**2)+.1*(a['rank_score']+b['rank_score']));candidates.append(dict(mode=mode,rank_score=score,digits=[a,b],necessary_common_translation_lower_m=lo.tolist(),necessary_common_translation_upper_m=hi.tolist()))
 candidates.sort(key=lambda a:(a['rank_score'],a['digits'][0]['grid_sample'],a['digits'][1]['grid_sample']));pair_counts[mode]=len(candidates);queue.extend(candidates[:plan['maximum_facet_pair_trials_per_mode']])
atlas_path=O/'atlas.json';atlas_path.write_text(json.dumps(dict(schema='v2d.finite-compiled-finger-contact-atlas.v1',digit_patches=rows),indent=2)+'\n');queue_path=O/'fixed-facet-pair-queue.json';queue_path.write_text(json.dumps(queue,indent=2)+'\n');report=dict(schema='v2d.compiled-hand-reachability-atlas-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),inputs_sha256={str(p):sha(p) for p in [setup,ns['bank_path'],ns['patch_path'],ns['mp'],ns['f']/'targets.npz',R/'scripts/dex3_training/v2d_dex3_flashchord/native_mesh.py']},geometry_sha256=bank['geometry_sha256'],atlas_sha256=sha(atlas_path),queue_sha256=sha(queue_path),source_region_and_source_authority_unchanged=True,root_orientation_fixed_to_current_center=True,root_translation_component_bounds_m=.008,root_center_not_changed=True,summary=summary,nearest_centroid_queries=query_count,necessary_facet_pairs=necessary_facet_pairs,common_translation_AABB_eligible_pair_counts=pair_counts,queued_facet_pair_trials=len(queue),centroid_neighbor_and_retention_pruning_is_finite_not_global=True,no_local_wrist_or_finger_optimizer=True,no_native_compile_or_physics_steps=True,no_prior_jobs_repeated=True,new_GPU_jobs=0,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

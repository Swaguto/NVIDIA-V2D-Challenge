"""New frozen-pose witness/native-gap forensics, no optimization or physics."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
producer=S/'compute/solve_native_contact_patch_modes_v1.py'
old_report_path=S/'e037-native-contact-patch-modes-v1/report.json'
old=json.loads(old_report_path.read_text());assert sha(producer)==old['producer_sha256']
# Restore only immutable setup and pure witness functions from exact executed
# source. The mode/iteration loop is deliberately excluded, never re-executed.
text=producer.read_text();prefix=text.split("for proposal in bank['contact_mode_hypotheses']:")[0]
assert 'for iteration in range' not in prefix
ns={'__file__':str(producer)};exec(compile(prefix,str(producer),'exec'),ns)
np=ns['np'];mj=ns['mujoco'];model=ns['model'];data=ns['data'];bn=ns['bn'];bank=ns['bank'];patches=ns['patches'];T=ns['T'];ob=ns['ob'];O=S/'e037-contact-patch-witness-audit-v1';start=time.monotonic()
plan_path=S/'compute/contact-patch-witness-audit-CPU-plan-v1.json';plan=json.loads(plan_path.read_text())
for doc,key in [(old,'inputs_sha256'),(bank,'input_sha256')]:
 for path,h in doc[key].items():assert sha(Path(path))==h
assert sha(ns['patch_path'])==bank['patch_file_sha256']
with np.load(ns['patch_path']) as z:part_ids={k:z[k+'_part_ids'].copy() for k in bank['patches']}
object_geoms=np.flatnonzero(model.geom_bodyid==ob);rows=[]
def local_hull(g):
 mid=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);return ns['ConvexHull'](np.asarray(model.mesh_vert[a:a+n],float))
object_hulls={int(g):local_hull(int(g)) for g in object_geoms}
def signed_halfspace(point,g):
 G=data.geom_xmat[g].reshape(3,3);local=np.einsum('i,ij->j',point-data.geom_xpos[g],G,optimize=False);h=object_hulls[g];return float((np.einsum('ki,i->k',h.equations[:,:3],local,optimize=False)+h.equations[:,3]).max())
body_T=np.eye(4)
for old_mode in old['mode_reports']:
 path=S/'e037-native-contact-patch-modes-v1'/old_mode['mode_id']/'single-knot-patch-command.npz'
 with np.load(path) as z:q=z['right_command'].copy()
 ns['set_state'](q);body_T[:3,:3]=data.xmat[ob].reshape(3,3);body_T[:3,3]=data.xpos[ob];np.testing.assert_allclose(body_T,T,atol=1e-12)
 mode=ns['RobotContactMode'](old_mode['mode_id'],'right',bank['object_id'],bank['geometry_sha256'],old_mode['digit_patches'])
 pairs=ns['pairs_for_mode'](mode);items=[]
 for p in pairs:
  g=int(p['geom']);hit=patches[p['patch']].closest_world(p['hand_point'][None],T);tri_id=int(hit.triangle_indices[0]);parent=int(object_geoms[int(part_ids[p['patch']][tri_id])]);gaps=[]
  for og in object_geoms:
   segment=np.zeros(6);gap=float(mj.mj_geomDistance(model,data,g,int(og),1.,segment));gaps.append(dict(object_geom=int(og),signed_gap_m=gap,fromto_world_m=segment.tolist(),broadphase_masks_allow_pair=bool((model.geom_contype[g]&model.geom_conaffinity[og])or(model.geom_contype[og]&model.geom_conaffinity[g]))))
  h=local_hull(g);G=data.geom_xmat[g].reshape(3,3);local=np.einsum('i,ij->j',p['hand_point']-data.geom_xpos[g],G,optimize=False);hand_surface=float((np.einsum('ki,i->k',h.equations[:,:3],local,optimize=False)+h.equations[:,3]).max())
  distances=sorted(gaps,key=lambda a:a['signed_gap_m']);active=[dict(geom_pair=[int(c.geom1),int(c.geom2)],dist_m=float(c.dist)) for c in data.contact[:data.ncon] if g in [int(c.geom1),int(c.geom2)] and (int(c.geom1) in object_geoms or int(c.geom2) in object_geoms)]
  patch_point=hit.points[0];normal=hit.outward_normals[0];delta=p['hand_point']-patch_point;signed=float(np.dot(delta,normal));tangent=float(np.linalg.norm(delta-signed*normal));original_hit=patches[p['patch']].closest_world(p['object_point'][None],T)
  # Distinguish a last-iteration stale object witness from a native gap error.
  nearest_original_delta=float(np.linalg.norm(hit.points[0]-p['object_point']))
  items.append(dict(digit=p['digit'],hand_geom=g,hand_body=bn[int(model.geom_bodyid[g])],assigned_patch=p['patch'],nearest_patch_triangle=tri_id,parent_object_geom=parent,hand_point_world_m=p['hand_point'].tolist(),old_object_witness_world_m=p['object_point'].tolist(),nearest_patch_point_world_m=patch_point.tolist(),hand_hull_halfspace_residual_m=hand_surface,nearest_patch_distance_m=float(hit.distances_m[0]),normal_offset_m=signed,tangential_offset_m=tangent,old_target_error_m=float(np.linalg.norm(p['error'])),nearest_vs_old_object_witness_m=nearest_original_delta,object_parent_halfspace_at_hand_point_m=signed_halfspace(p['hand_point'],parent),minimum_union_halfspace_at_hand_point_m=min(signed_halfspace(p['hand_point'],int(og)) for og in object_geoms),parent_gap=next(r for r in gaps if r['object_geom']==parent),minimum_native_gap=distances[0],all_native_pair_gaps=gaps,native_actual_contacts_for_selected_geom=active,hand_mesh_margin_m=float(model.geom_margin[g]),parent_mesh_margin_m=float(model.geom_margin[parent])))
 rows.append(dict(mode_id=mode.mode_id,saved_command_sha256=sha(path),pairs=items,native_contacts_not_reclassified_or_dropped=True))
report=dict(schema='v2d.contact-patch-witness-native-gap-audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),inputs_sha256={str(p):sha(p) for p in [producer,old_report_path,ns['mp'],ns['patch_path']]},all_recorded_bank_and_mode_inputs_verified=True,rows=rows,source_flags_and_heldout_failures_unchanged=True,saved_candidates_not_mutated=True,no_optimizer_or_prior_fit_rerun=True,no_native_compile_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
for r in rows:
 print(json.dumps(dict(mode_id=r['mode_id'],pairs=[{k:p[k] for k in ['digit','hand_body','old_target_error_m','nearest_patch_distance_m','normal_offset_m','tangential_offset_m','nearest_vs_old_object_witness_m','minimum_union_halfspace_at_hand_point_m','parent_gap','minimum_native_gap','native_actual_contacts_for_selected_geom']} for p in r['pairs']]),indent=2),flush=True)

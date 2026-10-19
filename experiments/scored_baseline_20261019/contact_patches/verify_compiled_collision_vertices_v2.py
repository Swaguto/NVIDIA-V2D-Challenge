"""NEW compiled-support contract and saved-pose support-gap closure audit."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'))
import numpy as np,mujoco,pytest
from scipy.spatial import ConvexHull
from v2d_dex3_flashchord.native_mesh import compiled_mesh_vertices
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();O=S/'e037-compiled-collision-vertex-verification-v2';start=time.monotonic();plan_path=S/'compute/compiled-collision-vertex-CPU-plan-v2.json';plan=json.loads(plan_path.read_text());test=R/'scripts/dex3_training/tests/test_native_mesh.py';library=R/'scripts/dex3_training/v2d_dex3_flashchord/native_mesh.py';exit_code=0;assert '18 passed' in (S/'e037-compiled-collision-vertex-verification-v1/worker.log').read_text();assert json.loads((S/'e037-compiled-collision-vertex-verification-v1/guard.json').read_text())['worker_exit_code']==1
producer=S/'compute/solve_native_contact_patch_modes_v1.py';old=json.loads((S/'e037-native-contact-patch-modes-v1/report.json').read_text());assert sha(producer)==old['producer_sha256'];prefix=producer.read_text().split("for proposal in bank['contact_mode_hypotheses']:")[0];assert 'for iteration in range' not in prefix;ns={'__file__':str(producer)};exec(compile(prefix,str(producer),'exec'),ns);model=ns['model'];data=ns['data'];bank=ns['bank']
for doc,key in [(old,'inputs_sha256'),(bank,'input_sha256')]:
 for p,h in doc[key].items():assert sha(Path(p))==h
witness_path=S/'e037-contact-patch-witness-audit-v1/report.json';witness=json.loads(witness_path.read_text());rows=[];mesh_rows=[]
for g in ns['right_geoms']+np.flatnonzero(model.geom_bodyid==ns['ob']).tolist():
 mid=int(model.geom_dataid[g]);v=compiled_mesh_vertices(model,mid,mujoco_version=mujoco.__version__);raw=model.mesh_vertnum[mid];mesh_rows.append(dict(geom=g,mesh=mid,raw_vertices=int(raw),native_support_vertices=len(v),raw_convex_hull_does_not_equal_compiled_support=bool(len(v)<raw)))
def vertices(g,compiled):
 mid=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);v=compiled_mesh_vertices(model,mid,mujoco_version=mujoco.__version__) if compiled else np.asarray(model.mesh_vert[a:a+n],float)
 return np.einsum('ni,ji->nj',v,data.geom_xmat[g].reshape(3,3),optimize=False)+data.geom_xpos[g]
def residual(point,g):
 h=ConvexHull(vertices(g,True));return float((np.einsum('ki,i->k',h.equations[:,:3],point,optimize=False)+h.equations[:,3]).max())
for mode in witness['rows']:
 path=S/'e037-native-contact-patch-modes-v1'/mode['mode_id']/'single-knot-patch-command.npz';assert sha(path)==mode['saved_command_sha256']
 with np.load(path) as z:ns['set_state'](z['right_command'])
 items=[]
 for p in mode['pairs']:
  g=p['hand_geom'];og=p['parent_object_geom'];segment=np.zeros(6);gap=float(mujoco.mj_geomDistance(model,data,g,og,1.,segment));normal=(segment[3:]-segment[:3])/gap;assert abs(np.linalg.norm(normal)-1)<1e-6
  supports={}
  for compiled in [False,True]:
   A=vertices(g,compiled);B=vertices(og,compiled);supports['compiled' if compiled else 'all_raw']=float(np.einsum('ni,i->n',B,normal,optimize=False).min()-np.einsum('ni,i->n',A,normal,optimize=False).max())
  closure=abs(supports['compiled']-gap);assert closure<5e-6,(mode['mode_id'],p['digit'],closure)
  items.append(dict(digit=p['digit'],hand_geom=g,object_geom=og,native_signed_gap_m=gap,compiled_support_gap_m=supports['compiled'],all_raw_support_gap_along_same_direction_m=supports['all_raw'],compiled_support_native_gap_error_m=closure,old_hand_witness_outside_compiled_hand_hull_m=residual(np.asarray(p['hand_point_world_m']),g),old_object_witness_outside_compiled_object_hull_m=residual(np.asarray(p['old_object_witness_world_m']),og),native_fromto_hand_hull_residual_m=residual(segment[:3],g),native_fromto_object_hull_residual_m=residual(segment[3:],og)))
 rows.append(dict(mode_id=mode['mode_id'],pairs=items))
thumbs=[p for r in rows for p in r['pairs'] if p['digit']=='thumb'];assert all(p['native_signed_gap_m']>.002 for p in thumbs);assert all(p['old_hand_witness_outside_compiled_hand_hull_m']>.001 for p in thumbs)
report=dict(schema='v2d.compiled-native-convex-support-verification.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),library_sha256=sha(library),test_sha256=sha(test),pytest_exit_code=int(exit_code),focused_tests_previously_passed=18,tests_not_unnecessarily_repeated=True,prior_failed_producer_sha256=sha(S/'compute/verify_compiled_collision_vertices_v1.py'),prior_failure_log_sha256=sha(S/'e037-compiled-collision-vertex-verification-v1/worker.log'),prior_JSON_numpy_bool_serialization_failure_retained=True,MuJoCo_version=mujoco.__version__,primary_support_layout_source='https://github.com/google-deepmind/mujoco/blob/3.8.1/src/engine/engine_collision_convex.c',primary_graph_producer_source='https://github.com/google-deepmind/mujoco/blob/3.8.1/src/user/user_mesh.cc',input_sha256={str(p):sha(p) for p in [producer,witness_path,ns['mp']]},mesh_vertex_counts=mesh_rows,rows=rows,maximum_compiled_support_native_gap_error_m=max(p['compiled_support_native_gap_error_m'] for r in rows for p in r['pairs']),close_thumb_witnesses_are_outside_native_support_hulls=True,old_all_raw_surface_candidates_rejected_not_promoted=True,old_native_audits_not_overridden=True,collision_model_friction_mass_effort_source_or_commands_not_mutated=True,no_optimizer_or_prior_fit_rerun=True,no_native_compile_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

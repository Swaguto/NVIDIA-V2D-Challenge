"""NEW independent authority bindings and full-clock translation map, no fit."""
from pathlib import Path
from fractions import Fraction as Q
import ast,datetime,hashlib,json,os
import numpy as np
from object_translation_maps_v1 import exact_translation_motion_maps,scalar_apply
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');O=S/'e037-object-translation-binding-v1';A=S/'e037-object-source-authority-v1';C=S/'e037-object-translation-contract-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
def qrec(v):return dict(numerator=v.numerator,denominator=v.denominator)
def parse(v):return Q(v['numerator'],v['denominator'])
with (O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,v)
 emit(dict(kind='BEFORE_NEW_object_authority_and_motion_map_binding',authority_report_sha256=sha(A/'report.json'),NEW14contracts_reused_BY_HASH=sha(C/'report.json'),no_contract_source_native_audit_or_solver_rerun=True))
 old_sources={p:read(Path(p)) for p in {r['current_source_receipt'] for r in read(A/'all151-source-authority-inventory.json')}}
 rows=read(A/'all151-source-authority-inventory.json');copies=0;stricts=0
 track=read(S/'plane-aligned-pose-temporal-path-v4/report.json')
 for row in rows:
  f=row['frame'];source=old_sources[row['current_source_receipt']];raw=source
  for component in row['current_source_JSON_path']:raw=raw[component]
  assert raw==row['current_accepted_source_thresholds_COPIED'];assert row['old_strict_source_comparison_COPIED']==track['rows'][f]['source_fit_before'];assert row['original_object_observed_flag']==track['rows'][f]['original_object_observed_flag'];copies+=3;stricts+=3
  emit(dict(kind='AFTER_NEW_independent_source_threshold_origin',frame=f,current_receipt=row['current_source_receipt'],current_receipt_sha256=sha(row['current_source_receipt']),exact_JSON_copy_matches=True,old_strict_floor_COPIED_not_rescored=True))
 bindings=read(A/'original-gate-source-bindings.json');functionchecks=0
 for name,rec in bindings['functions'].items():
  p=Path(rec['path']);node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name);assert ast.get_source_segment(p.read_text(),node)==rec['exact_function_source'];assert sha(p)==rec['source_sha256'];assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==rec['AST_sha256'];functionchecks+=1
 def nums(name):return {float(n.value)for n in ast.walk(ast.parse(bindings['functions'][name]['exact_function_source']))if isinstance(n,ast.Constant)and isinstance(n.value,(int,float))}
 assert {.05,.15,40.}.issubset(nums('pose_rejection'));assert {.01,.1,2.5,4096.,100.}.issubset(nums('observed_fit'));assert {.02,.1,.5,200.}.issubset(nums('fusion_depth'))
 save('independent-original-gate-constant-bindings.json',dict(exact_function_AST_bindings=functionchecks,source_threshold_copy_checks=copies,original_strict_floor_components=stricts,original_tracker_constants_bound=True,NOTE_no_object_pose_box_found_ONLY_in_these_inspected_original_gates=True,NOT_blanket_permission_to_move_objects_arbitrarily=True))
 frame=read(A/'NEW-source-world-native-frame-return.json');times=np.asarray(frame['timestamps_s']);T=[Q.from_float(float(v))for v in times];maps=exact_translation_motion_maps(times)
 with np.load(S/'e037-joint-pose-translation-repair-v1/unqualified-combined-object-proposal.npz')as z:world=z['world_from_object'];assert np.array_equal(z['timestamps_s'],times)
 variables=[dict(column=3*f+k,name=f'd_world_{f:03d}_{"xyz"[k]}',frame=f,component=k,units='metres',continuous_derived_object_translation_NOT_observation=True)for f in range(len(T))for k in range(3)]
 save('UNQUALIFIED-continuous-object-translation-symbols.json',variables)
 derived=[];coefficients=0;constantchecks=0
 for family in ['velocity','acceleration']:
  for i,row in enumerate(maps[family]):
   emit(dict(kind='BEFORE_NEW_wholeclock_object_motion_affine_map',family=family,clock_index=i,unassigned_translation_variables=True))
   if family=='velocity':expected={i:-1/(T[i+1]-T[i]),i+1:1/(T[i+1]-T[i])}
   else:
    dt0=T[i+1]-T[i];dt1=T[i+2]-T[i+1];mid=(dt0+dt1)/2;expected={i:1/(dt0*mid),i+1:-(1/dt0+1/dt1)/mid,i+2:1/(dt1*mid)}
   assert row==expected and sum(row.values())==0
   constants=scalar_apply(row,world[:,:3,3]);serialized=[]
   for k in range(3):
    terms=[dict(column=3*f+k,coefficient=qrec(v))for f,v in sorted(row.items())];coefficients+=len(terms)
    direct=([ (Q.from_float(float(world[i+1,k,3]))-Q.from_float(float(world[i,k,3])))/(T[i+1]-T[i]) ] if family=='velocity'else[( (Q.from_float(float(world[i+2,k,3]))-Q.from_float(float(world[i+1,k,3])))/(T[i+2]-T[i+1])-(Q.from_float(float(world[i+1,k,3]))-Q.from_float(float(world[i,k,3])))/(T[i+1]-T[i]) )/((T[i+2]-T[i])/2)])[0]
    assert direct==constants[k];constantchecks+=1;serialized.append(dict(component=k,constant=qrec(constants[k]),terms=terms))
   result=dict(family=family,index=i,scalar_maps=serialized,units='metres_per_second'if family=='velocity'else'metres_per_second_squared',bound_origin='ORIGINAL object motion receipt; not robot root caps',gate='Euclidean norm <= ORIGINAL cap + original1e-9 comparison tolerance',no_evaluation_of_new_delta=True)
   derived.append(result);save(f'{family}-{i:03d}-affine-map.json',result);emit(dict(kind='AFTER_NEW_wholeclock_object_motion_affine_map',returned=result,independent_exact_timestamp_formula_passed=True))
 save('all299-UNQUALIFIED-object-motion-affine-maps.json',derived)
 clarification=dict(source_authority_report_sha256=sha(A/'report.json'),partition_metadata_scope='Even/odd labels describe a DECLARED FUTURE training-only protocol consistent with prior50/47/49 experiments. This inventory did NOT create 151historical partitions, read new pixel labels, or show historical heldout independence at every frame.',original_observed_fit_fullpixel_gate_is_postconstruction_only=True,original_gate_includes_both_parities_do_NOT_optimize_it=True,first_surface_fixed_ray_partitions_exist_in_saved50_and47_49_artifacts_only=True,all_prior_masks_depth_validity_misses_and_heldout_evidence_retained=True,old50mm_5mm_fitting_boxes_are_constructor_limits_NOT_original_object_pose_authority=True,object_box_absence_is_scope_limited_not_feasibility_or_unbounded_movement_permission=True,no_new_source_pose_selected_or_queued=True,source_world_native_closure_not_source_accuracy=True)
 save('authority-split-and-scope-clarification.json',clarification)
 result=dict(schema='v2d.NEW-independent-object-authority-translation-bindings.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_threshold_and_flag_copy_checks=copies,source_floor_components=stricts,original_function_ASTs=functionchecks,continuous_unassigned_object_translation_symbols=len(variables),wholeclock_velocity_maps=len(maps['velocity']),wholeclock_acceleration_maps=len(maps['acceleration']),exact_sparse_scalar_coefficients=coefficients,exact_affine_constants=constantchecks,old14contracts_reused_BY_HASH=True,fullclock_source_world_native_closure_REUSED_BY_HASH=True,original_all_six_object_caps_required=True,original_p95_cannot_be_replaced_by_only_maximum_SOC=True,original_angular_motion_constant_ONLY_for_fixed_rotations=True,no_object_candidate_or_acceptance_or_exclusion=True,new_source_native_fit_solver_GPU_or_physics_calls=0,goal_UNACHIEVED=True)
 save('report.json',result);emit(dict(kind='AFTER_NEW_independent_object_authority_translation_binding',returned=result));print(json.dumps(result))

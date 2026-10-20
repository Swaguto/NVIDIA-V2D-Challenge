"""NEW unreached binding continuation; retains the failed native/URDF identity check."""
from joint_collision_maps_v1 import S,F,M,G,O,sha,save,save_npz
from pathlib import Path
import json,hashlib
import numpy as np
from fractions import Fraction
D=S/'e037-joint-collision-binding-continuation-v1'
B=S/'e037-joint-collision-binding-v1'
paths=[B/'NEW-input-SHA-bindings-RETURN.json',B/'NEW-static-tree-numeric-bindings-RETURN.json',B/'NEW-all-body-chain-bindings-RETURN.json',B/'NEW-URDF-native-static-origin-bindings-RETURN.json',B/'guard.json',B/'worker.log']
receipts={str(p):sha(p) for p in paths}
save(D/'NEW-earlier-bindings-BY-HASH-RETURN.json',receipts)
assert json.loads((B/'guard.json').read_text())['worker_exit_code']==1
assert all(json.loads((B/'NEW-input-SHA-bindings-RETURN.json').read_text()).values())
static_bindings=json.loads((B/'NEW-static-tree-numeric-bindings-RETURN.json').read_text())
assert all(v['exact'] for v in static_bindings.values())
previous_chains=json.loads((B/'NEW-all-body-chain-bindings-RETURN.json').read_text())
assert all(r['ancestor_exact'] and all(all(o[k] for k in ['parent_matches','position_exact','quaternion_exact','joint_order_exact']) for o in r['operations']) for r in previous_chains)
origin_records=json.loads((B/'NEW-URDF-native-static-origin-bindings-RETURN.json').read_text())
# No larger tolerance or repaired metadata. Native and URDF are distinct maps.
# Read the actual saved discrepancy returns and retain both original sources.
scope=dict(native_URDF_bit_or_1e12_identity_FAILED=True,
    native_body_origin_diff_max_m=max(r['origin_translation_error'] for r in origin_records),
    native_joint_range_diff_max_rad=max(r['limit_error'] for r in origin_records),
    native_origin_rotation_diff_max=max(r['origin_rotation_error'] for r in origin_records),
    cause_UNPROVEN=True, no_tolerance_or_metadata_changed=True,
    source_factors_and_URDF_gates_remain_original_URDF=True,
    NEW_collision_maps_use_actual_native_MJB_tree=True,
    actual_candidate_needs_BOTH_source_URDF_and_native_gates=True)
save(D/'NEW-native-URDF-distinct-map-scope-RETURN.json',scope)
assert all(r['child_suffix_matches'] and r['axis_error']==0 and r['origin_rotation_error']==0 and r['native_pivot_norm']==0 and r['native_reference']==0 for r in origin_records)
# Static model adapter uses the already bound saved arrays, no second MJB read.
raw=json.loads((O/'NEW-static-native-tree-RETURN.json').read_text())
class Record:
    def __init__(self,name): self.name=name
class Static:
    def __getattr__(self,k):
        v=raw[k]
        return np.array(v) if isinstance(v,list) else v
    def body(self,i): return Record(raw['body_names'][int(i)])
    def joint(self,i): return Record(raw['joint_names'][int(i)])
model=Static()
program=json.loads((O/'NEW-articulated-chain-program-RETURN.json').read_text())
meta=json.loads((O/'NEW-URDF-native-incidence-RETURN.json').read_text())
expected_ancestors=np.zeros((model.nbody,model.njnt),dtype=np.uint8)
for rec in previous_chains:
    for op in rec['operations']:
        expected_ancestors[rec['body'],np.flatnonzero(model.jnt_bodyid==op['body'])]=1
variables = json.loads((M / 'continuous-angle-variables.json').read_text())
with np.load(O / 'NEW-complete-column-and-pair-incidence-RETURN.npz', allow_pickle=False) as z:
    maps = {k: z[k] for k in z.files}
expected_joint_columns = np.full_like(maps['column_for_frame_joint'], -1)
route_return = []
for rec in variables:
    native = [j for j in range(model.njnt) if model.joint(j).name.endswith('_' + rec['URDF_joint']) or model.joint(j).name == rec['URDF_joint']]
    assert len(native) == 1
    jid = native[0]
    expected_joint_columns[rec['frame'], jid] = rec['column']
    route_return.append([rec['column'], rec['frame'], jid, rec['command_column']])
# The free coordinate affects exactly its descendant geometry bodies at its clock.
expected_geom_columns = np.full_like(maps['geom_robot_columns'], -1)
right_joints = [r['native_joint_matches'][0] for r in meta['bindings'] if r['side'] == 'right']
for g in range(model.ngeom):
    descendant = expected_ancestors[int(model.geom_bodyid[g])]
    for slot, jid in enumerate(right_joints):
        if descendant[jid]:
            expected_geom_columns[:, g, slot] = expected_joint_columns[:, jid]
# Every scalar route is returned before comparisons.
save_npz(D / 'NEW-independent-column-route-RETURN.npz',
    joint_columns=expected_joint_columns, body_joint_incidence=expected_ancestors,
    geom_robot_columns=expected_geom_columns, robot_embed=np.arange(465),
    object_embed=np.arange(465, 1523),
    object_columns=np.arange(465, 1522).reshape(151, 7))
assert np.array_equal(maps['column_for_frame_joint'], expected_joint_columns)
assert np.array_equal(maps['body_joint_incidence'], expected_ancestors)
assert np.array_equal(maps['geom_robot_columns'], expected_geom_columns)
assert np.array_equal(maps['robot_embed'], np.arange(465))
assert np.array_equal(maps['object_embed'], np.arange(465, 1523))
assert np.array_equal(maps['object_columns'], np.arange(465, 1522).reshape(151, 7))
# Independent exhaustive upper-triangle bijection. No native eligibility flag drops rows.
pairs = maps['geom_pair']
ii, jj = np.triu_indices(58, 1)
expected_pairs = np.column_stack([ii, jj])
assert np.array_equal(pairs, expected_pairs)
assert np.array_equal(maps['all_frames'], np.arange(151))
flags = json.loads((O / 'NEW-complete-pair-filter-metadata-RETURN.json').read_text())
flag_checks = []
for row, (a, b) in zip(flags['pairs'], expected_pairs, strict=True):
    ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
    wa, wb = int(model.body_weldid[ba]), int(model.body_weldid[bb])
    signature = 65536 * min(ba, bb) + max(ba, bb)
    expected = dict(geoms=[int(a), int(b)], bodies=[ba,bb], same_body=ba == bb,
        same_weld=wa == wb,
        welded_parent_relation=int(model.body_parentid[wa]) == wb or int(model.body_parentid[wb]) == wa,
        collision_mask_allows=bool((model.geom_contype[a] & model.geom_conaffinity[b]) |
                                   (model.geom_contype[b] & model.geom_conaffinity[a])),
        explicit_body_exclusion=signature in set(int(s) for s in model.exclude_signature),
        packed_body_signature=signature, retained=True)
    flag_checks.append(dict(geoms=[int(a),int(b)], exact=all(row[k] == v for k,v in expected.items())))
save(D / 'NEW-complete-pair-and-self-filter-bindings-RETURN.json', dict(
    per_pair=flag_checks, pair_instances=151 * len(pairs),
    all_28_right_self_pairs_per_clock=flags['categories_per_frame']['right_internal'] == 28,
    no_pair_omitted=True, filter_flags_NOT_native_contact_observations=True))
assert all(r['exact'] for r in flag_checks)
assert flags['categories_per_frame']['right_internal'] == 28
# Numeric compiled BODY supports are bound by every stored component's byte hash.
supports = json.loads((O / 'NEW-compiled-support-map-RETURN.json').read_text())
support_checks = []
with np.load(supports['cache_path'], allow_pickle=False) as z:
    for rec in supports['geoms']:
        v, n = z[rec['vertices_key']], z[rec['normals_key']]
        support_checks.append(dict(geom=rec['geom'], vertices_components=v.size,
            normals_components=n.size,
            vertices_exact=hashlib.sha256(v.tobytes()).hexdigest() == rec['vertices_bytes_sha256'],
            normals_exact=hashlib.sha256(n.tobytes()).hexdigest() == rec['normals_bytes_sha256'],
            body_exact=rec['body'] == int(model.geom_bodyid[rec['geom']]),
            placement_applied_once=rec['geom_placement_ALREADY_applied']))
save(D / 'NEW-all-compiled-support-component-bindings-RETURN.json', support_checks)
assert all(all(r[k] for k in ['vertices_exact','normals_exact','body_exact','placement_applied_once']) for r in support_checks)
# Keep all unchanged object source/motion/cone rows. Old geometry replaced by NEW
# articulated support predicates; no stale fixed-hand inequality is inherited.
assert np.array_equal(maps['inherited_object_row_intervals'], [[0,2],[66593,4618162]])
assert np.array_equal(maps['invalid_old_fixed_hand_geometry_interval'], [2,66593])
declaration = json.loads((O / 'NEW-joint-construction-declaration-RETURN.json').read_text())
q_checks = []
for q in program['q_assignments']:
    jid = q['joint']
    if q['kind'] == 'hinge_angle':
        b = next(x for x in meta['bindings'] if x['native_joint_matches'][0] == jid)
        manifest = json.loads((F / 'manifest.json').read_text())
        si = ['left','right'].index(b['side'])
        command = 7 + manifest['joint_names'][b['side']].index(b['URDF_joint'])
        ok = q['fixed_indices'] == ['frame',si,command] and q['fixed_key'] == 'model_base_qpos'
    elif q['kind'] == 'fixed_saved_root_command':
        name = model.joint(jid).name
        si = 0 if 'left_wrist_' in name else 1
        suffix = name.split('wrist_')[-1]
        ok = q['indices'] == ['frame',si,{'x':[0],'y':[1],'z':[2],'rotvec':[6,3,4,5]}[suffix]]
    elif q['kind'] == 'object_Cayley_from_F':
        ok = model.body(model.jnt_bodyid[jid]).name == 'wooden_spoon_object' and int(model.jnt_type[jid]) == 0
    elif q['kind'] == 'ORIGINAL_qpos0_support':
        ok = model.body(model.jnt_bodyid[jid]).name == 'support' and q['qpos0_address'] == int(model.jnt_qposadr[jid])
    else:
        ok = False
    q_checks.append(dict(joint=jid, exact=ok))
save(D / 'NEW-every-qpos-command-source-binding-RETURN.json', q_checks)
assert len(q_checks) == model.njnt and all(r['exact'] for r in q_checks)
save(D / 'report.json', dict(schema='v2d.NEW-independent-joint-collision-map-binding.v1',
    passed=True, namespace_variables=1523, native_bodies=31, native_joints=24,
    all_free_joint_clock_routes=465, all_dense_geom_routes=int(expected_geom_columns.size),
    static_numeric_entries=sum(r['entries'] for r in static_bindings.values()),
    compiled_vertex_normal_components=sum(r['vertices_components']+r['normals_components'] for r in support_checks),
    all_pair_instances=249603, right_self_instances=28*151,
    maximum_static_URDF_origin_diagnostic=max(r['origin_rotation_error'] for r in origin_records),
    object_source_motion_cones_reused_BY_HASH=True,
    old_fixed_hand_geometry_rows_NOT_inherited=66591,
    all_primal_values_UNASSIGNED=True, no_episode_FK_native_source_solver_evaluation=True))

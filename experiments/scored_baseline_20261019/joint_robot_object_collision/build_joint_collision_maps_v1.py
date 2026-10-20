"""Build NEW symbolic coupled native chains/pairs; never evaluate an episode pose."""
from joint_collision_maps_v1 import *
import itertools
import xml.etree.ElementTree as ET
import mujoco

D = O
model_path = S / 'e037-motion-constrained-wrist-native-v1/native-model.mjb'
variables_path = M / 'continuous-angle-variables.json'
arrays_path = M / 'unqualified-model-fixed-components-and-initial-F.npz'
geometry_path = G / 'NEW-native-body-rotation-factors-RETURN.npz'
source_model = S / 'e037-hard-raster-model-v1'
contracts_path = S / 'e037-joint-collision-contracts-v1/report.json'
input_paths = [model_path, variables_path, arrays_path, geometry_path, contracts_path,
    F / 'manifest.json', F / 'targets.npz',
    F / 'robot/urdfs/g1/dex3_left.urdf', F / 'robot/urdfs/g1/dex3_right.urdf',
    M / 'all-exact-linear-angle-inequalities.json',
    M / 'all-direct-source-factor-proposals.json',
    M / 'retained-halfangle-domains-and-unknown-source-evidence.json',
    M / 'actual-mask-and-immutable-source-bindings.json',
    source_model / 'NEW-canonical-A-RETURN.npz',
    source_model / 'NEW-canonical-P-RETURN.npz',
    source_model / 'NEW-canonical-vectors-RETURN.npz',
    source_model / 'NEW-canonical-metadata-RETURN.json']
# Bind all source-model per-frame rows and strict flags without re-evaluation.
input_paths += sorted(source_model.glob('frame-*-NEW-hard-cell-coefficients-RETURN.npz'))
inputs = {str(p): sha(p) for p in input_paths}
save(D / 'NEW-input-authority-RETURN.json', inputs)
assert inputs[str(model_path)] == '755fc51729d4c4fc03d04108d39bec2b249074351863f096f9aa498c9ac28566'
assert inputs[str(F / 'manifest.json')] == 'f438adf3da846ec1c76c31c7b6d6c12b46bea76004aa7745cfada504378da606'
assert json.loads(contracts_path.read_text())['passed']
assert mujoco.__version__ == '3.8.1'
model = mujoco.MjModel.from_binary_path(str(model_path))
fields = ['body_parentid', 'body_weldid', 'body_pos', 'body_quat', 'body_jntadr',
          'body_jntnum', 'jnt_type', 'jnt_bodyid', 'jnt_pos', 'jnt_axis',
          'jnt_qposadr', 'qpos0', 'jnt_limited', 'jnt_range', 'geom_bodyid',
          'geom_pos', 'geom_quat', 'geom_type', 'geom_dataid', 'geom_contype',
          'geom_conaffinity', 'geom_margin', 'geom_gap', 'exclude_signature',
          'pair_geom1', 'pair_geom2']
static = {name: getattr(model, name).copy() for name in fields}
static.update(body_names=[model.body(i).name for i in range(model.nbody)],
              joint_names=[model.joint(i).name for i in range(model.njnt)],
              geom_names=[model.geom(i).name for i in range(model.ngeom)],
              mujoco_version=mujoco.__version__, nq=model.nq, nv=model.nv,
              nbody=model.nbody, njnt=model.njnt, ngeom=model.ngeom,
              disableflags=int(model.opt.disableflags),
              FILTERPARENT_bit=int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT),
              NO_MjData_or_forward_or_native_distance=True)
save(D / 'NEW-static-native-tree-RETURN.json', static)
assert (model.nbody, model.ngeom) == (31, 58)
body_names = static['body_names']
joint_names = static['joint_names']
parents = static['body_parentid']
chains = [ancestors(parents, b) for b in range(model.nbody)]
# URDF joint/link incidence and origin data only: no URDF poses().
urdfs = {}
bindings = []
for side in ['left', 'right']:
    path = F / f'robot/urdfs/g1/dex3_{side}.urdf'
    root = ET.parse(path).getroot()
    records = []
    for j in root.findall('joint'):
        origin = j.find('origin')
        axis = j.find('axis')
        rec = dict(name=j.attrib['name'], type=j.attrib['type'],
                   parent=j.find('parent').attrib['link'], child=j.find('child').attrib['link'],
                   xyz=[float(v) for v in (origin.attrib.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()],
                   rpy=[float(v) for v in (origin.attrib.get('rpy', '0 0 0') if origin is not None else '0 0 0').split()],
                   axis=None if axis is None else [float(v) for v in axis.attrib['xyz'].split()])
        if j.find('limit') is not None:
            rec['limit'] = {k: float(v) for k, v in j.find('limit').attrib.items()}
        records.append(rec)
        if j.attrib['type'] != 'fixed':
            matches = [i for i, name in enumerate(joint_names) if name == rec['name'] or name.endswith('_' + rec['name'])]
            bindings.append(dict(side=side, URDF_joint=rec['name'], native_joint_matches=matches,
                                 URDF_child=rec['child']))
    urdfs[side] = records
save(D / 'NEW-URDF-native-incidence-RETURN.json', dict(URDF=urdfs, bindings=bindings))
assert len(bindings) == 14 and all(len(b['native_joint_matches']) == 1 for b in bindings)
manifest = json.loads((F / 'manifest.json').read_text())
variables = json.loads(variables_path.read_text())
column_for_frame_joint = np.full((151, model.njnt), -1, dtype=np.int32)
variable_routes = []
for v in variables:
    jid = next(i for i, n in enumerate(joint_names) if n == v['URDF_joint'] or n.endswith('_' + v['URDF_joint']))
    column_for_frame_joint[v['frame'], jid] = v['column']
    variable_routes.append(dict(column=v['column'], frame=v['frame'], native_joint=jid,
                               native_body=int(model.jnt_bodyid[jid]), command_column=v['command_column']))
# Indirections retain exact saved fixed commands; they are not new pose values.
qassign = []
for jid, name in enumerate(joint_names):
    if jid in [b['native_joint_matches'][0] for b in bindings]:
        bind = next(b for b in bindings if b['native_joint_matches'][0] == jid)
        si = 0 if bind['side'] == 'left' else 1
        idx = manifest['joint_names'][bind['side']].index(bind['URDF_joint'])
        qassign.append(dict(joint=jid, kind='hinge_angle', frame_variables_table='column_for_frame_joint',
                            fixed_fallback_path=str(arrays_path), fixed_key='model_base_qpos',
                            fixed_indices=['frame', si, 7 + idx], reference_qpos0_address=int(model.jnt_qposadr[jid])))
    elif 'wrist_' in name:
        side = 'left' if 'left_wrist_' in name else 'right'
        si = 0 if side == 'left' else 1
        suffix = name.split('wrist_')[-1]
        command_indices = {'x': [0], 'y': [1], 'z': [2], 'rotvec': [6, 3, 4, 5]}[suffix]
        qassign.append(dict(joint=jid, kind='fixed_saved_root_command', path=str(arrays_path),
                            key='model_base_qpos', indices=['frame', si, command_indices],
                            quaternion_order='wxyz' if suffix == 'rotvec' else None,
                            unqualified_fixed_model_hypothesis=(side == 'right')))
    elif model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
        body_name = body_names[int(model.jnt_bodyid[jid])]
        qassign.append(dict(joint=jid, kind='object_Cayley_from_F' if body_name == 'wooden_spoon_object' else 'ORIGINAL_qpos0_support',
                            object_columns='465+7*frame+[0:7]' if body_name == 'wooden_spoon_object' else None,
                            qpos0_address=int(model.jnt_qposadr[jid]), body_name=body_name))
    else:
        raise ValueError('unbound native joint ' + name)
# Symbolic chain records bind every ancestor and all q operations in native order.
program = []
body_joint_incidence = np.zeros((model.nbody, model.njnt), dtype=np.uint8)
for bid in range(model.nbody):
    operations = []
    for ancestor in chains[bid]:
        joint_ids = list(range(int(model.body_jntadr[ancestor]), int(model.body_jntadr[ancestor] + model.body_jntnum[ancestor]))) if model.body_jntnum[ancestor] else []
        for jid in joint_ids:
            body_joint_incidence[bid, jid] = 1
        operations.append(dict(body=ancestor, parent=int(parents[ancestor]),
                               local_translation=static['body_pos'][ancestor],
                               local_quaternion_wxyz=static['body_quat'][ancestor],
                               ordered_joints=joint_ids))
    program.append(dict(body=bid, name=body_names[bid], ancestors=chains[bid], operations=operations))
save(D / 'NEW-articulated-chain-program-RETURN.json', dict(body_programs=program,
    q_assignments=qassign, variable_routes=variable_routes,
    semantics=dict(fixed='T_parent * T(body_pos) * R(body_quat)',
                   hinge='T(jnt_pos) * R(axis,q-qpos0) * T(-jnt_pos)',
                   slide='T(axis*(q-qpos0))',
                   ball='T(jnt_pos) * R(q_wxyz) * T(-jnt_pos)',
                   free='world T(q_xyz) R(q_wxyz), replaces local free-body pose',
                   point_J='world_axis CROSS (point-world_joint_pivot)',
                   normal_J='world_axis CROSS normal',
                   moving_support_direction_product_rule='dn.dot(xA-xB) + n.dot(dxA-dxB)',
                   support='max over ALL compiled vertices; retain EVERY tied vertex',
                   ideal_convex_clearance='exists unit n: h_A(n)+h_B(-n)<=0.001m',
                   ideal_scope='Minkowski inradius sign; NOT numerical native EPA/contact identity')))
# A complete dense ancestor route (small) with -1 meaning exact no-free-column.
geom_columns = np.full((151, model.ngeom, 7), -1, dtype=np.int32)
right_hinges = [b['native_joint_matches'][0] for b in bindings if b['side'] == 'right']
for frame in range(151):
    for geom in range(model.ngeom):
        for slot, jid in enumerate(right_hinges):
            if body_joint_incidence[model.geom_bodyid[geom], jid]:
                geom_columns[frame, geom, slot] = column_for_frame_joint[frame, jid]
# Reuse correct compiled BODY supports by hash; never read mesh_vert/extract again.
with np.load(geometry_path, allow_pickle=False) as z:
    supports = []
    for geom in range(model.ngeom):
        vertices, normals = z[f'g{geom}_vertices_body'], z[f'g{geom}_normals_body']
        supports.append(dict(geom=geom, body=int(model.geom_bodyid[geom]),
            vertices_key=f'g{geom}_vertices_body', normals_key=f'g{geom}_normals_body',
            vertices_shape=vertices.shape, normals_shape=normals.shape,
            vertices_bytes_sha256=hashlib.sha256(vertices.tobytes()).hexdigest(),
            normals_bytes_sha256=hashlib.sha256(normals.tobytes()).hexdigest(),
            geom_placement_ALREADY_applied=True, map='T_body_analytic * body_vertices',
            normals_map='R_body_analytic * body_normals',
            native_mesh_reader='native_mesh.compiled_mesh_vertices, pinned3.8.1 BY_HASH cache provenance'))
save(D / 'NEW-compiled-support-map-RETURN.json', dict(cache_path=str(geometry_path),
     cache_sha256=inputs[str(geometry_path)], geoms=supports,
     old_cached_F_world_poses_NOT_used=True, raw_mesh_vert_NOT_read=True))
# All 58 choose 2 pairs retained, including excluded/internal signatures.
groups = [geometry_group(body_names[int(model.geom_bodyid[g])]) for g in range(model.ngeom)]
pairs = np.array(list(itertools.combinations(range(model.ngeom), 2)), dtype=np.int32)
filter_flags = []
category_counts = {}
for a, b in pairs:
    a, b = int(a), int(b)
    ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
    wa, wb = int(model.body_weldid[ba]), int(model.body_weldid[bb])
    signature = (min(ba, bb) << 16) + max(ba, bb)
    cat = pair_category(groups[a], groups[b])
    category_counts[cat] = category_counts.get(cat, 0) + 1
    filter_flags.append(dict(geoms=[a, b], category=cat, bodies=[ba, bb],
        same_body=(ba == bb), same_weld=(wa == wb),
        welded_parent_relation=(int(model.body_parentid[wa]) == wb or int(model.body_parentid[wb]) == wa),
        collision_mask_allows=bool((int(model.geom_contype[a]) & int(model.geom_conaffinity[b])) or
                                   (int(model.geom_contype[b]) & int(model.geom_conaffinity[a]))),
        explicit_body_exclusion=bool(signature in model.exclude_signature),
        packed_body_signature=signature, retained=True,
        native_eligibility='metadata flags ONLY; actual native filters/gate mandatory; no pair omitted'))
save(D / 'NEW-complete-pair-filter-metadata-RETURN.json', dict(pairs=filter_flags,
    categories_per_frame=category_counts, pair_instances=151 * len(pairs),
    every_pair_retained=True, explicit_pairs=int(model.npair),
    filterparent_disabled=bool(int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))))
# Factorized full incidence: pair x frame pointers avoid materializing vertex products.
# All pairs and all clocks retained. Every body and free ancestor maps to one namespace.
object_columns = np.array([[465 + 7 * f + k for k in range(7)] for f in range(151)], dtype=np.int32)
save_npz(D / 'NEW-complete-column-and-pair-incidence-RETURN.npz',
    column_for_frame_joint=column_for_frame_joint, body_joint_incidence=body_joint_incidence,
    geom_robot_columns=geom_columns, geom_bodyid=static['geom_bodyid'],
    geom_pair=pairs, all_frames=np.arange(151, dtype=np.int32),
    object_columns=object_columns, object_embed=np.arange(465, 1523, dtype=np.int32),
    robot_embed=np.arange(465, dtype=np.int32),
    inherited_object_row_intervals=np.array([[0, 2], [66593, 4618162]], dtype=np.int64),
    invalid_old_fixed_hand_geometry_interval=np.array([2, 66593], dtype=np.int64))
save(D / 'NEW-joint-construction-declaration-RETURN.json', dict(
    variables=1523, robot_angle_variables=465, object_variables=1058,
    all_variables_UNASSIGNED=True, optimizer_selected=False,
    robot_source_motion='8114 linear rows/194 source factors/38 domains/fullregion2 and unknowns BY HASH',
    object_restrictions='every unchanged source/clock/motion/cone/strict flag retained via exact old-column embedding',
    invalid_geometry_rows_NOT_inherited=66591,
    new_collision_family='ALL compiled supports under analytic coupled body chains; complete unit-sphere support predicate',
    pair_instances=151 * len(pairs), selected_direction_or_pose_bank=False,
    conditioning='robot original radian coordinates; object saved SI/r,Cayley,lift coordinates; NO new scales',
    fixed_unqualified_wrist_middle='model hypotheses only; F remains sole verified reference',
    left_early_late_thumb_missing_commands='saved exact values by immutable indirection',
    strict_source_native_uncertainties='all retained; no closed-model/source/physical equivalence',
    acceptance=['seal NEW complete proposal and ALL constructor/solver/intermediate returns BEFORE queries',
                'independent whole151 source/actual URDF/ALL original robot and object motion',
                'actual integer cells/strict depth/changed NN/ranks/rounded median/quality/fulltriangle training',
                'original fullpixel and heldout POST only, no fitting or rescue feedback',
                'fresh native FULL hand/object/support AND both-hand self',
                'full swept and actual motor/grasp/whole-source before GPU/PPO'],
    no_episode_pose_source_NN_raster_FK_native_solver_physics_GPU_evaluation=True))
save(D / 'report.json', dict(schema='v2d.NEW-joint-symbolic-collision-construction.v1',
    variables=1523, articulated_bodies=model.nbody, native_joints=model.njnt,
    compiled_geoms=model.ngeom, all_pairs_per_frame=len(pairs), pair_instances=151 * len(pairs),
    categories_per_frame=category_counts, symbolic_chain_maps=True,
    episode_body_positions_distances_NOT_evaluated=True, no_solver_or_pose_selected=True,
    old_constraints_reused_BY_HASH=True, new_maps_durable=True))

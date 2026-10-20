"""NEW explicit gauge binding for the coupled native-object expression."""
from joint_collision_maps_v1 import *
D=S/'e037-joint-native-object-gauge-v1'
meta_path=S/'e037-hard-raster-model-v1/NEW-canonical-metadata-RETURN.json'
geometry_path=G/'NEW-native-body-rotation-factors-RETURN.npz'
meta=json.loads(meta_path.read_text())
raw=json.loads((O/'NEW-static-native-tree-RETURN.json').read_text())
body=raw['body_names'].index('wooden_spoon_object')
with np.load(geometry_path,allow_pickle=False) as z:
 stage=z['stage_world_translation_linear'].copy()
 object_base=z['cached_body_poses'][:,body].copy()
# NEW synthetic noncommuting-frame control: rotation belongs to the native body,
# translation belongs to the declared old world-to-native stage linear map.
q=rod(np.array([0.,1.,0.]),.3)
base=rod(np.array([1.,0.,0.]),-.4)
toy_stage=rod(np.array([0.,0.,1.]),.2)
v=np.array([.2,-.1,.3]);delta=np.array([.03,-.02,.01])
correct=base@q@v+toy_stage@delta
wrong=q@base@v+delta
save(D/'NEW-object-frame-order-contract-RETURN.json',dict(
 correct=correct,wrong=wrong,distinct=float(np.max(np.abs(correct-wrong))),
 delta_native=toy_stage@delta,body_frame_rotation='base_R * Q, not Q * base_R'))
assert np.max(np.abs(correct-wrong))>.001
save_npz(D/'NEW-native-object-gauge-coefficients-RETURN.npz',
 stage_world_translation_linear=stage,cached_NATIVE_object_base=object_base,
 radius=np.array(meta['radius']),scale=np.array(meta['scale']),
 object_body=np.array(body,dtype=np.int32))
save(D/'NEW-native-object-gauge-program-RETURN.json',dict(
 geometry_cache_path=str(geometry_path),geometry_cache_sha256=sha(geometry_path),
 original_F_source_poses_path=str(F/'targets.npz'),original_F_source_poses_sha256=sha(F/'targets.npz'),
 canonical_metadata_path=str(meta_path),canonical_metadata_sha256=sha(meta_path),
 inherited_pose_semantics_path=str(G/'NEW-full-pose-factor-semantics-RETURN.json'),
 inherited_pose_semantics_sha256=sha(G/'NEW-full-pose-factor-semantics-RETURN.json'),
 native_object_body=body,radius=meta['radius'],scale=meta['scale'],
 variable_slice='x[465+7*f:465+7*f+7]',
 phi='u/scale',eta='radius*x_eta',delta='eta/(1+phi.dot(phi))',
 Q='((1-phi.phi)I+2phi*phi.T+2[phi]cross)/(1+phi.phi)',
 native_R='cached_NATIVE_object_base[f,:3,:3] * Q(phi)',
 native_t='cached_NATIVE_object_base[f,:3,3] + stage_world_translation_linear * delta',
 source_world_R='ORIGINAL_F_object_world_R[f] * Q(phi)',
 source_world_t='ORIGINAL_F_object_world_t[f] + delta',
 sigma='construction lift ONLY, no direct actual-geometry coordinate',
 phase='x[1522], physical_rho=radius*x[1522]',
 robot_root='saved model_base_qpos native-command root indirection, NO object-stage applied',
 support='ORIGINAL native qpos0, NO new object map applied',
 compiled_geometry='cached BODY vertices transformed ONCE by this native rigid map',
 old_generic_object_Cayley_from_F_tag_refined_by_THIS_explicit_gauge=True,
 old_cached_F_HAND_world_poses_NOT_used=True,
 no_object_pose_or_source_or_native_position_evaluation=True))
save(D/'report.json',dict(schema='v2d.NEW-joint-native-object-gauge.v1',
 new_synthetic_contracts=1,world_to_native_translation_map_bound=True,
 object_frame_rotation_order_bound=True,sigma_lift_separated=True,
 no_episode_pose_FK_source_native_solver_evaluation=True))

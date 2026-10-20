"""NEW full-pose training wrapper; actual unchanged k2/median/raster helpers."""
import numpy as np
from object_full_pose_maps_v1 import compose,source_dispatch
from object_training_source_semantics_v1 import raster_training_counts,vertex_median_semantics,real_quality,exact_training_iou_nonregression
class FullPoseTrainingOracle:
 def __init__(self,data,load_training,emit,save_arrays):self.d=data;self.load=load_training;self.emit=emit;self.save=save_arrays
 def query(self,f,delta,theta,proposal_id):
  if not proposal_id:raise ValueError('sealed NEW full-pose identity required')
  dispatch=source_dispatch(delta,theta);base=self.d['stats'][f];self.emit(dict(kind='BEFORE_NEW_full_pose_training_query',frame=f,proposal_id=proposal_id,world_translation=delta,object_rotation_coordinates=theta,dispatch=dispatch))
  if dispatch=='CACHED_F_BY_HASH':
   out=dict(frame=f,proposal_id=proposal_id,status=dispatch,passes_TRAINING_construction=True,training_intersection=base['training_intersection'],training_union=base['training_union'],training_median_m=base['training_mesh_vertex_median_m'],training_quality=base['training_quality']);self.emit(dict(kind='AFTER_zero_F_CACHE_RETURN',actual_RETURN=out));return out
  if dispatch!='NEW_actual_training_operators_REQUIRED':
   out=dict(frame=f,proposal_id=proposal_id,status=dispatch,passes_TRAINING_construction=False);self.emit(dict(kind='AFTER_NEW_unavailable_RETURN',actual_RETURN=out));return out
  T=compose(self.d['camera'][f],delta,theta,self.d['Rcw'][f]);camera=self.d['V']@T[:3,:3].T+T[:3,3]
  pose_return=self.save(f'{proposal_id}-f{f:03d}-NEW-camera-pose-RETURN',dict(camera_from_object=T,camera_vertices=camera));self.emit(dict(kind='AFTER_NEW_full_pose_camera_constructor_RETURN',frame=f,array_RETURN=pose_return,numerical_scope='NEW changed-rotation matmul positions; original training k2/einsum local NN; old C0 alias uncertainty retained'))
  data=self.load(f);image=raster_training_counts(camera,self.d['faces'],self.d['K'],data['training_mask'],data['domain']);image_arrays={}
  if image['status']=='training_raster_counted':image_arrays=dict(predicted_training_mask=image['predicted_training_mask'],uv_float=image['projection_return']['actual_uv_float'],uv_integer=image['projection_return']['actual_uv_integer'])
  ip=self.save(f'{proposal_id}-f{f:03d}-image-RETURN',image_arrays);imeta={k:v for k,v in image.items()if k not in ['projection_return','predicted_training_mask']};self.emit(dict(kind='AFTER_NEW_full_pose_raster_RETURN',frame=f,actual_RETURN=imeta,array_RETURN=ip))
  if image['status']!='training_raster_counted':
   out=dict(frame=f,proposal_id=proposal_id,status='UNRESOLVED_training_projection',passes_TRAINING_construction=False);self.emit(dict(kind='AFTER_NEW_full_pose_source_RETURN',actual_RETURN=out));return out
  median=vertex_median_semantics(self.d['V'],data['points_C_m'],T,self.d['Rcw'][f]);keys=['point_local','nearest_vertex_indices','nearest_vertex_distances','second_vertex_distances','central_point_indices','central_distances','central_weights','world_translation_local_gradient','point_world_translation_linear'];mp=self.save(f'{proposal_id}-f{f:03d}-median-RETURN',{k:median[k]for k in keys if k in median});mmeta={k:v for k,v in median.items()if k not in keys};self.emit(dict(kind='AFTER_NEW_full_pose_median_RETURN',frame=f,actual_RETURN=mmeta,array_RETURN=mp))
  if median['status']!='training_mesh_vertex_median':
   out=dict(frame=f,proposal_id=proposal_id,status='UNRESOLVED_training_median',passes_TRAINING_construction=False);self.emit(dict(kind='AFTER_NEW_full_pose_source_RETURN',actual_RETURN=out));return out
  ratio=exact_training_iou_nonregression(image,dict(intersection=base['training_intersection'],union=base['training_union']));q=real_quality(image['iou'],median['median']);dm=float(median['median']-base['training_mesh_vertex_median_m']);dq=float(q-base['training_quality']);passed=bool(ratio['status']=='exact_integer_ratio_comparison'and ratio['nonregressing']and dm<=0 and dq>=0)
  out=dict(frame=f,proposal_id=proposal_id,status='NEW_actual_full_pose_TRAINING_ONLY',dispatch=dispatch,training_intersection=image['intersection'],training_union=image['union'],training_median_m=median['median'],training_quality=q,training_IoU_determinant=ratio.get('determinant'),training_median_change_m=dm,training_quality_change=dq,passes_TRAINING_construction=passed,original_fullpixel_heldout_motion_native_self_swept_motor_grasp_NOT_qualified=True);self.emit(dict(kind='AFTER_NEW_full_pose_source_RETURN',actual_RETURN=out));return out

"""NEW hard training-source predicate for separately declared future proposals.

It is not the original fullpixel gate or a trajectory solver. Zero F queries
reuse saved training values by hash; only NEW nonzero proposals are evaluated.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from object_training_source_semantics_v1 import raster_training_counts,vertex_median_semantics,real_quality,exact_training_iou_nonregression

class TrainingSourceOracle:
 def __init__(self,vertices,faces,K,C0,camera_poses,Rcw,frames,load_training,emit,save_arrays):
  self.V=np.asarray(vertices,float);self.faces=np.asarray(faces,int);self.K=np.asarray(K,float);self.C0=np.asarray(C0,float);self.T=np.asarray(camera_poses,float);self.Rcw=np.asarray(Rcw,float);self.frames=frames;self.load=load_training;self.emit=emit;self.save_arrays=save_arrays
 def query(self,frame,world_translation,proposal_id):
  if not isinstance(proposal_id,str)or not proposal_id:raise ValueError('durable NEW proposal identity required')
  delta=np.asarray(world_translation,float)
  if delta.shape!=(3,)or not np.isfinite(delta).all():raise ValueError('finite derived world translation required')
  base=self.frames[frame];self.emit(dict(kind='BEFORE_NEW_training_hard_constraint_query',frame=frame,proposal_id=proposal_id,derived_world_translation=delta,not_original_fullpixel_gate_or_observation=True))
  if np.array_equal(delta,np.zeros(3)):
   out=dict(status='CACHED_F_training_values_BY_HASH',frame=frame,proposal_id=proposal_id,training_intersection=base['training_intersection'],training_union=base['training_union'],training_median_m=base['training_mesh_vertex_median_m'],training_quality=base['training_quality'],training_IoU_determinant=0,training_median_change_m=0.,training_quality_change=0.,passes_TRAINING_construction=True,old_baseline_raster_point_or_scorer_NOT_replayed=True,original_source_motion_native_whole_trajectory_gates_NOT_qualified=True);self.emit(dict(kind='AFTER_training_constraint_zero_F_REUSED_RETURN',actual_RETURN=out));return out
  data=self.load(frame);C=self.C0[frame]+np.einsum('ij,j->i',self.Rcw[frame],delta,optimize=False);T=self.T[frame].copy();T[:3,3]+=np.einsum('ij,j->i',self.Rcw[frame],delta,optimize=False);image=raster_training_counts(C,self.faces,self.K,data['training_mask'],data['domain']);image_arrays={}
  if image['status']=='training_raster_counted':image_arrays=dict(predicted_training_mask=image['predicted_training_mask'],uv_float=image['projection_return']['actual_uv_float'],uv_integer=image['projection_return']['actual_uv_integer'])
  ip=self.save_arrays(f'{proposal_id}-f{frame:03d}-image-RETURN',image_arrays);imeta={k:v for k,v in image.items()if k not in ['projection_return','predicted_training_mask']};self.emit(dict(kind='AFTER_NEW_training_hard_raster_RETURN',frame=frame,actual_RETURN=imeta,array_RETURN=ip))
  if image['status']!='training_raster_counted':out=dict(status='UNRESOLVED_training_source_projection',frame=frame,proposal_id=proposal_id,passes_TRAINING_construction=False,original_gates_UNCHANGED=True);self.emit(dict(kind='AFTER_NEW_training_hard_constraint_RETURN',actual_RETURN=out));return out
  median=vertex_median_semantics(self.V,data['points_C_m'],T,self.Rcw[frame]);keys=['point_local','nearest_vertex_indices','nearest_vertex_distances','second_vertex_distances','central_point_indices','central_distances','central_weights','world_translation_local_gradient','point_world_translation_linear'];mp=self.save_arrays(f'{proposal_id}-f{frame:03d}-median-RETURN',{k:median[k]for k in keys if k in median});mmeta={k:v for k,v in median.items()if k not in keys};self.emit(dict(kind='AFTER_NEW_training_hard_median_RETURN',frame=frame,actual_RETURN=mmeta,array_RETURN=mp))
  if median['status']!='training_mesh_vertex_median':out=dict(status='UNRESOLVED_training_source_median',frame=frame,proposal_id=proposal_id,passes_TRAINING_construction=False,original_gates_UNCHANGED=True);self.emit(dict(kind='AFTER_NEW_training_hard_constraint_RETURN',actual_RETURN=out));return out
  ratio=exact_training_iou_nonregression(image,dict(intersection=base['training_intersection'],union=base['training_union']));quality=real_quality(image['iou'],median['median']);dmedian=float(median['median']-base['training_mesh_vertex_median_m']);dq=float(quality-base['training_quality']);passed=bool(ratio['status']=='exact_integer_ratio_comparison'and ratio['nonregressing']and dmedian<=0 and dq>=0);out=dict(status='NEW_training_source_predicate_ONLY',frame=frame,proposal_id=proposal_id,training_intersection=image['intersection'],training_union=image['union'],training_median_m=median['median'],training_quality=quality,training_IoU_determinant=ratio.get('determinant'),training_median_change_m=dmedian,training_quality_change=dq,passes_TRAINING_construction=passed,nominal_hard_training_restrictions_NOT_original_candidate_tolerances=True,original_fullpixel_source_comparison_tolerance_UNCHANGED=1e-10,local_median_world_gradient=median['world_translation_local_gradient'],rank_and_NN_ties_NOT_fixed_source_angles=True,raster_predicate_discontinuous_NO_fake_classical_IoU_gradient=True,original_source_motion_native_self_swept_motor_grasp_whole_reference_NOT_qualified=True);self.emit(dict(kind='AFTER_NEW_training_hard_constraint_RETURN',actual_RETURN=out));return out

from pathlib import Path
from fractions import Fraction as Q
import datetime,json,os
import cv2,numpy as np
from object_training_source_semantics_v1 import TRAINING,training_domain,select_training_pixels,projected_integer_vertices,raster_training_counts,exact_training_iou_nonregression,point_cloud_from_pixels,vertex_median_semantics,real_quality
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];O=S/'e037-training-source-contract-v1';passed=[]
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def t(n,proposal,fn):
  journal(log,dict(kind='BEFORE_NEW_training_semantic_contract',name=n,proposal=proposal));r=fn();save(n+'-RETURN.json',r);journal(log,dict(kind='AFTER_NEW_training_semantic_contract_RETURN',name=n,actual_RETURN=r));assert r['passed'],n;passed.append(n)
 def parity():
  d=np.ones((32,32));v=np.ones_like(d,bool);m=np.ones_like(v);domain=training_domain(d.shape);a=select_training_pixels(d,v,m);d[~domain]=99;v[~domain]=False;m[~domain]=False;b=select_training_pixels(d,v,m);return dict(passed=np.array_equal(a['pixels'],b['pixels'])and a['eligible_count']==b['eligible_count']==512,original_odd_values_NOT_optimization_inputs=True)
 t('NEW_odd_source_values_do_not_affect_training_selection',dict(grid=[32,32]),parity)
 def sampling():
  d=np.ones((100,100));v=np.ones_like(d,bool);a=select_training_pixels(d,v,v);expected=np.argwhere(training_domain(d.shape))[np.linspace(0,4999,4096,dtype=int)][:,::-1];return dict(passed=np.array_equal(a['pixels'],expected)and len(np.unique(a['indices']))==4096,selected_count=len(a['pixels']),original_source_cap4096_retained_in_NEW_training_partition=True)
 t('NEW_training_raster_order_sampling_cap',dict(training_count=5000),sampling)
 def quantization():
  uv=np.array([[-1.8,-.2],[-.8,.8],[.2,1.8],[20001.,-20001.]]);C=np.c_[uv,np.ones(len(uv))];p=projected_integer_vertices(C,np.eye(3));expected=np.array([[-1,0],[0,0],[0,1],[20000,-20000]],np.int32);return dict(passed=np.array_equal(p['actual_uv_integer'],expected),actual_RETURN=p,negative_zero_and_saturation_cases_retained=True)
 t('NEW_original_truncation_and_clip_semantics',dict(not_rint=True),quantization)
 def raster():
  C=np.array([[1.,1.,1.],[7.,1.,1.],[1.,7.,1.]]);faces=np.array([[0,1,2]]);domain=training_domain((10,10));mask=np.zeros((10,10),bool);mask[1:8,1:8]=True;a=raster_training_counts(C,faces,np.eye(3),mask,domain);mask[~domain]=~mask[~domain];b=raster_training_counts(C,faces,np.eye(3),mask,domain);manual=np.zeros((10,10),np.uint8);cv2.fillConvexPoly(manual,np.array([[1,1],[7,1],[1,7]],np.int32),255);return dict(passed=a['intersection']==b['intersection']and a['union']==b['union']and np.array_equal(a['predicted_training_mask'],(manual>127)&domain),counts=[a['intersection'],a['union']],odd_mask_unknown_NOT_negative_label=True)
 t('NEW_full_triangle_training_IoU_counts_and_odd_independence',dict(one_triangle=True),raster)
 def ratios():
  base=dict(intersection=3,union=7);up=exact_training_iou_nonregression(dict(intersection=6,union=14),base);down=exact_training_iou_nonregression(dict(intersection=4,union=10),base);empty=exact_training_iou_nonregression(dict(intersection=0,union=0),base);return dict(passed=up['nonregressing']and not down['nonregressing']and empty['status'].startswith('UNRESOLVED'),actual_returns=[up,down,empty],no_IoU_float_tolerance_substitution=True)
 t('NEW_exact_IoU_cross_multiplication',dict(positive_union_required=True),ratios)
 def points():
  d=np.full((4,4),.5);p=np.array([[1,2],[3,0]]);K=np.array([[2.,0,1],[0,4.,2],[0,0,1]]);ret=point_cloud_from_pixels(p,d,K);expected=np.array([[0,0,.5],[.5,-.25,.5]]);return dict(passed=np.array_equal(ret,expected),actual_RETURN=ret,depth_is_CV_z_not_range_or_occluded_joint=True)
 t('NEW_original_camera_surface_point_lift',dict(two_pixels=True),points)
 V=np.array([[0.,0.,0.],[10.,10.,10.],[20.,20.,20.]])
 def evenmedian():
  P=np.c_[np.linspace(1.,2.,100),np.full(100,.1),np.zeros(100)];T=np.eye(4);r=vertex_median_semantics(V,P,T,np.eye(3));ds=np.linalg.norm(P,axis=1);manual=(np.sort(ds)[49]+np.sort(ds)[50])/2;return dict(passed=r['median']==manual and np.array_equal(r['central_point_indices'],[49,50]),actual_RETURN=r,even_median_is_TWO_ranks=True)
 t('NEW_even_training_median_two_rank_identity',dict(points=100),evenmedian)
 def gradient():
  P=np.c_[np.linspace(1.,2.,101),np.full(101,.15),np.full(101,.05)];T=np.eye(4);R=np.array([[0,-1.,0],[1.,0,0],[0,0,1.]]);r=vertex_median_semantics(V,P,T,R);h=1e-6;fd=[]
  for j in range(3):
   a=T.copy();b=T.copy();a[:3,3]+=R[:,j]*h;b[:3,3]-=R[:,j]*h;fd.append((vertex_median_semantics(V,P,a,R)['median']-vertex_median_semantics(V,P,b,R)['median'])/(2*h))
  e=float(abs(np.asarray(fd)-r['world_translation_local_gradient']).max());return dict(passed=e<1e-8,actual_gradient=r['world_translation_local_gradient'],independent_FD=fd,error=e,diagnostic_NOT_candidate_tolerance=True)
 t('NEW_translation_inverse_and_dynamic_median_gradient',dict(unique_NN_and_rank=True),gradient)
 def switching():
  vv=np.array([[0.,0,0],[2.,0,0],[5.,0,0]]);P=np.tile([.9,.1,0.],(101,1));a=vertex_median_semantics(vv,P,np.eye(4),np.eye(3));T=np.eye(4);T[0,3]=-.4;b=vertex_median_semantics(vv,P,T,np.eye(3));return dict(passed=np.all(a['nearest_vertex_indices']==0)and np.all(b['nearest_vertex_indices']==1),old_correspondence_NOT_frozen_as_source_truth=True,old_ids=np.unique(a['nearest_vertex_indices']),new_ids=np.unique(b['nearest_vertex_indices']))
 t('NEW_nearest_vertex_alias_can_switch',dict(two_source_branches=True),switching)
 def lipschitz():
  P=np.c_[np.linspace(1.,2.,101),np.full(101,.15),np.full(101,.05)];a=vertex_median_semantics(V,P,np.eye(4),np.eye(3));T=np.eye(4);shift=np.array([.1,-.05,.02]);T[:3,3]=shift;b=vertex_median_semantics(V,P,T,np.eye(3));return dict(passed=abs(a['median']-b['median'])<=np.linalg.norm(shift),actual_median_change=abs(a['median']-b['median']),translation_norm_bound=np.linalg.norm(shift),minimum_and_order_statistics_preserve_common_Lipschitz_bound=True,NOT_a_nonregression_certificate_for_nonzero_delta=True)
 t('NEW_median_NEAREST_rank_change_Lipschitz_necessity',dict(arbitrary_translation=True),lipschitz)
 def monotone():
  old=real_quality(.4,.006);new=real_quality(.41,.005);return dict(passed=new>=old,old_quality=old,new_quality=new,mathematical_scope='positive I exp(-m/.02) is monotone in I and antitone in m',actual_ORIGINAL_fullpixel_quality_gate_still_separate=True)
 t('NEW_real_quality_monotonicity_NOT_gate_replacement',dict(source_scale=.02),monotone)
 def unknown():
  P=np.zeros((100,3));r=vertex_median_semantics(V,P,np.eye(4),np.eye(3));small=vertex_median_semantics(V,P[:2],np.eye(4),np.eye(3));return dict(passed=r['zero_central_distance_UNRESOLVED_subgradient']and small['status'].startswith('UNRESOLVED'),no_zero_distance_or_missing_support_promoted=True)
 t('NEW_zero_rank_and_insufficient_source_support_remain_explicit',dict(zero_distance=True),unknown)
 out=dict(schema='v2d.NEW-training-source-semantic-contracts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEWcontracts=len(passed),contracts=passed,no_episode_or_old_controls_renderer_scorer_or_solver_calls=True,new_training_model_NOT_original_gate_change=True,no_trajectory_source_accuracy_or_physical_qualification=True);save('report.json',out);journal(log,dict(kind='AFTER_NEW_training_source_contract_suite',actual_RETURN=out));print(json.dumps(out))

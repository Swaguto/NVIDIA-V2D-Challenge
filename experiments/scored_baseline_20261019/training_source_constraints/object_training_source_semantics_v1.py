"""NEW training-domain counterpart of original silhouette/vertex-median scorer.

It never replaces the original full-pixel/heldout gates. Points/masks/depth are
inferred source evidence, not measured object pose or contact truth.
"""
import cv2,numpy as np
from scipy.spatial import cKDTree

TRAINING=dict(parity=0,maximum_depth_points=4096,minimum_depth_points=100,depth_minimum_m=.1,depth_maximum_m=2.5,vertex_near_plane_m=.01,projection_clip_px=20000,quality_distance_scale_m=.02)

def training_domain(shape):
 yy,xx=np.indices(shape);return (xx+yy)%2==0

def select_training_pixels(depth,valid,mask):
 d=np.asarray(depth);v=np.asarray(valid);m=np.asarray(mask)
 if d.ndim!=2 or v.shape!=d.shape or m.shape!=d.shape:raise ValueError('same original calibrated grid required')
 domain=training_domain(d.shape);eligible=v&m&np.isfinite(d)&(d>TRAINING['depth_minimum_m'])&(d<TRAINING['depth_maximum_m'])&domain;yx=np.argwhere(eligible);count=len(yx)
 indices=np.linspace(0,count-1,TRAINING['maximum_depth_points'],dtype=int)if count>TRAINING['maximum_depth_points']else np.arange(count,dtype=int)
 pixels=yx[indices][:,::-1];return dict(domain=domain,pixels=pixels,indices=indices,eligible_count=count,status='training_support_available'if len(pixels)>=TRAINING['minimum_depth_points']else'UNRESOLVED_insufficient_training_support')

def projected_integer_vertices(camera_vertices,K):
 C=np.asarray(camera_vertices,float);K=np.asarray(K,float)
 if C.ndim!=2 or C.shape[1]!=3 or not np.isfinite(C).all()or np.any(C[:,2]<=TRAINING['vertex_near_plane_m']):return dict(status='UNRESOLVED_original_vertex_near_plane')
 uv=np.column_stack([K[0,0]*C[:,0]/C[:,2]+K[0,2],K[1,1]*C[:,1]/C[:,2]+K[1,2]]);integer=np.clip(uv,-TRAINING['projection_clip_px'],TRAINING['projection_clip_px']).astype(np.int32);return dict(status='projected',actual_uv_float=uv,actual_uv_integer=integer,quantizer='ORIGINAL clip then truncate toward zero, NOT round-to-nearest')

def raster_training_counts(camera_vertices,faces,K,mask_training,domain):
 p=projected_integer_vertices(camera_vertices,K)
 if p['status']!='projected':return p
 predicted=np.zeros(np.asarray(mask_training).shape,np.uint8)
 for tri in p['actual_uv_integer'][np.asarray(faces,int)]:cv2.fillConvexPoly(predicted,tri,255)
 P=predicted>127;M=np.asarray(mask_training,bool);D=np.asarray(domain,bool);I=int((P&M&D).sum());U=int(((P|M)&D).sum());return dict(status='training_raster_counted',intersection=I,union=U,iou=float(I/max(1,U)),predicted_training_count=int((P&D).sum()),source_training_count=int((M&D).sum()),predicted_training_mask=P&D,projection_return=p)

def exact_training_iou_nonregression(current,baseline):
 if min(current['union'],baseline['union'])<=0:return dict(status='UNRESOLVED_empty_union')
 determinant=int(current['intersection'])*int(baseline['union'])-int(baseline['intersection'])*int(current['union']);return dict(status='exact_integer_ratio_comparison',determinant=determinant,nonregressing=determinant>=0,scope='training construction restriction ONLY; original source tolerance/gates unchanged')

def point_cloud_from_pixels(pixels,depth,K):
 p=np.asarray(pixels);z=np.asarray(depth[p[:,1],p[:,0]],float);return np.column_stack([(p[:,0]-K[0,2])*z/K[0,0],(p[:,1]-K[1,2])*z/K[1,1],z])

def vertex_median_semantics(vertices,points_camera,camera_from_object,camera_from_world_linear):
 V=np.asarray(vertices,float);P=np.asarray(points_camera,float);T=np.asarray(camera_from_object,float);RC=np.asarray(camera_from_world_linear,float)
 if len(P)<TRAINING['minimum_depth_points']:return dict(status='UNRESOLVED_insufficient_training_points')
 local=np.einsum('ni,ij->nj',P-T[:3,3],T[:3,:3],optimize=False);tree=cKDTree(V);dist,ids=tree.query(local,k=2,workers=1);first=dist[:,0];nn=ids[:,0];order=np.argsort(first,kind='stable');n=len(first);ranks=[(n-1)//2,n//2];selected=order[ranks];weights=np.array([.5,.5]);B=np.einsum('ji,jk->ik',T[:3,:3],RC,optimize=False);gradient=np.zeros(3);zero=False
 for i,w in zip(selected,weights,strict=True):
  if first[i]==0:zero=True;continue
  gradient-=w*((local[i]-V[nn[i]])/first[i])@B
 median=float((first[selected[0]]+first[selected[1]])/2);return dict(status='training_mesh_vertex_median',median=median,point_local=local,nearest_vertex_indices=nn,nearest_vertex_distances=first,second_vertex_distances=dist[:,1],central_point_indices=selected,central_distances=first[selected],central_weights=weights,world_translation_local_gradient=gradient,point_world_translation_linear=-B,zero_central_distance_UNRESOLVED_subgradient=zero,central_rank_tie=bool(first[selected[0]]==first[selected[1]]),nearest_vertex_tie_count=int(np.count_nonzero(dist[:,0]==dist[:,1])),nearest_vertex_correspondences_NOT_measurements_or_contacts=True,NN_and_rank_changes_NOT_fixed_as_mandatory=True)

def real_quality(iou,median):return float(iou*np.exp(-median/TRAINING['quality_distance_scale_m']))

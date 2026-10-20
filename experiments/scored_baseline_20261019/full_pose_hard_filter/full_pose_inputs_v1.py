"""Immutable factor/geometry loader for NEW full-pose construction only."""
from pathlib import Path
import json,hashlib,numpy as np
from scipy import sparse
S=Path(__file__).resolve().parents[1];C=S/'compute';F=S/'e037-object-full-pose-factors-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def load():
 from full_pose_construction_v1 import metric_radius
 d={};paths=[]
 p=F/'NEW-source-rotation-factors-RETURN.npz';paths.append(p)
 with np.load(p)as z:d.update(world=z['original_world_from_object'],camera=z['original_camera_from_object'],Rcw=z['camera_from_world_translation_linear'],times=z['original_timestamps_s'],indices=z['original_frame_indices'],source_rotation_J=z['camera_rotation_J_at_zero'],V=z['visual_vertices'],faces=z['faces'],K=z['K'])
 p=F/'NEW-native-body-rotation-factors-RETURN.npz';paths.append(p)
 with np.load(p)as z:
  d.update(body=z['cached_body_poses'],stage=z['stage_world_translation_linear']);d['vertices']={int(k.split('_')[0][1:]):z[k]for k in z.files if k.endswith('_vertices_body')};d['normals']={int(k.split('_')[0][1:]):z[k]for k in z.files if k.endswith('_normals_body')}
 p=F/'NEW-ALL-cross-pair-incidence-RETURN.npz';paths.append(p)
 with np.load(p)as z:d['pairs']=z['frame_fixed_object']
 p=F/'NEW-static-MJB-geometry-metadata-RETURN.json';paths.append(p);meta=read(p);d['object_geoms']=[r['geom_id']for r in meta['geoms']if r['body']=='wooden_spoon_object'];d['fixed_geoms']=[r['geom_id']for r in meta['geoms']if r['body']!='wooden_spoon_object'];d['radius']=metric_radius([d['vertices'][g]for g in d['object_geoms']])
 p=S/'e037-object-dynamic-runtime-v1/dynamic-source-data.npz';paths.append(p)
 with np.load(p)as z:d['C0']=z['C0']
 d['min_C0_z']=d['C0'][:,:,2].min(axis=1);d['min_depth_vertex']=d['C0'][:,:,2].argmin(axis=1)
 p=S/'e037-training-source-constraints-v1/all151-NEW-training-source-statistics.json';paths.append(p);d['stats']=read(p);d['translation_median_gradient']=np.array([r['local_median_world_gradient']for r in d['stats']])
 p=F/'NEW-training-median-rotation-SEARCH-gradient-RETURN.npz';paths.append(p)
 with np.load(p)as z:d['rotation_median_gradient']=z['rotation_SEARCH_gradient_at_F']
 for key,path in [('linear_M',S/'e037-object-dynamic-runtime-v1/dynamic-motion-linear-map.npz'),('angular_M',F/'NEW-original-angular-clock-Jacobian-RETURN.npz')]:d[key]=sparse.load_npz(path);paths.append(path)
 p=S/'e037-object-dynamic-runtime-v1/dynamic-motion-constants.npy';paths.append(p);d['linear_base']=np.load(p)
 p=F/'NEW-angular-clock-base-factors-RETURN.npz';paths.append(p)
 with np.load(p)as z:d['angular_base']=np.r_[z['angular_velocity'].ravel(),z['angular_acceleration'].ravel()]
 p=C/'object-authority-checkpoint-v1.json';paths.append(p);d['caps']=read(p)['original_object_motion_caps'];d['motion_caps']=list(d['caps'].values())
 # original dict order is pinned; assert semantic names rather than reorder gates.
 assert list(d['caps'])==['maximum_linear_speed_m_s','maximum_angular_speed_rad_s','maximum_linear_acceleration_m_s2','maximum_angular_acceleration_rad_s2','p95_linear_speed_m_s','p95_angular_speed_rad_s']
 d['provenance']={str(p):sha(p)for p in paths};return d

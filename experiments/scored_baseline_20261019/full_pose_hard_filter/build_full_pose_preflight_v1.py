"""ONE NEW complete907-column all-rotation proposal canonical, no solve."""
from pathlib import Path
import datetime,json,hashlib,os,time
import numpy as np
from scipy import sparse
from full_pose_inputs_v1 import load
from full_pose_geometry_runtime_v1 import geometry_all
from full_pose_construction_v1 import canonical
S=Path(__file__).resolve().parents[1];C=S/'compute';O=S/'e037-full-pose-preflight-v1';start=time.monotonic()
def save(n,d):
 with (O/n).open('w')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def arrays(n,d):
 with (O/n).open('wb')as f:np.savez_compressed(f,**d);f.flush();os.fsync(f.fileno())
 save(n+'.metadata.json',dict(path=str(O/n),sha256=hashlib.sha256((O/n).read_bytes()).hexdigest()))
d=load();save('NEW-complete-construction-declaration-BEFORE.json',dict(provenance=d['provenance'],origin='ORIGINAL F only; no old rejected vector/intermediate/fullpixel/heldout failure inputs',new_dimensions=907,all151_6pose_coordinates=True,rotation_order='R_F Exp(theta)',rotation_norm_bound_rad=float(np.pi/12),translation_norm_bound_m=d['radius']/8,metric_radius_m=d['radius'],norm_bounds_CONSTRUCTION_ONLY_NOT_original_authority=True,one_QP_one_raw_trial_ONLY=True,no_line_search_retry_or_continuation=True,QP_max_iter=100,QP_seconds=60.,threads=1,default_QP_tolerances=True,objective='rho/r + .5*1e-3*||[delta/r,theta]||^2',actual_trial_requires_all151_training_source_and6motion_and_strict_proxy_descent=True,original_fullpixel_heldout_POST_only=True,zero_F_source_uses_cache=True,initial_geometry_is_NEW_rotational_support_J_construction_NOT_old_native_audit=True,all66591pairs_no_broadphase_omissions=True,full_face_union_is_algorithmic_proxy_NOT_native_distance_or_grasp=True,no_episode_pose_or_solver_called=True))
g=geometry_all(d,np.zeros((151,6)));arrays('NEW-all-pair-rotating-support-linearization-RETURN.npz',{k:v for k,v in g.items()if isinstance(v,np.ndarray)});save('NEW-initial-proxy-RETURN.json',dict(proxy_phase_m=g['proxy_phase_m'],axis_instances=len(g['all_axis_gaps']),pair_instances=len(g['gaps']),exact_branch_tie_pairs=int(np.count_nonzero(g['branch_tie_counts']>1)),support_tie_pairs=int(np.count_nonzero(np.max(g['support_tie_counts'],axis=1)>1)),no_native_position_gap_or_physics_calls=True));del g['all_axis_gaps']
m=canonical(d,g)
for name in ['A','P']:sparse.save_npz(O/f'NEW-canonical-{name}-RETURN.npz',m[name])
arrays('NEW-canonical-vectors-RETURN.npz',dict(b=m['b'],q=m['q']))
metadata={k:v for k,v in m.items()if k not in ['A','P','b','q']};metadata['rows']=m['A'].shape[0];metadata['nonzeros']=m['A'].nnz;save('NEW-canonical-metadata-RETURN.json',metadata)
save('report.json',dict(schema='v2d.NEW-full-pose-preflight.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=m['A'].shape[0],columns=m['columns'],nonnegative=m['nonnegative'],SOC4=len(m['cones']),canonical_nonzeros=m['A'].nnz,all_pair_instances=len(d['pairs']),rotation_coordinates=453,initial_proxy_phase_m=g['proxy_phase_m'],all_pairs_retained=True,construction_bounds=m['config'],no_episode_pose_source_NN_raster_native_FK_solver_physics_GPU=True,old_contracts_maps_authority_reused_BY_HASH=True,wall_seconds=time.monotonic()-start));print(json.dumps(dict(rows=m['A'].shape[0],columns=m['columns'],SOC4=len(m['cones']),wall=time.monotonic()-start)))

"""Read immutable F and saved proposal inputs; no old producer or metric calls."""
from pathlib import Path
import hashlib,json
import numpy as np,trimesh
from scipy import sparse

S=Path(__file__).resolve().parents[1];D=S/'e037-object-dynamic-runtime-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def load_inputs():
    frozen=S/'e037-thumb-ray-independent-validation-v1/frozen';manifest=read(frozen/'manifest.json')
    assert sha(frozen/'manifest.json')=='f438adf3da846ec1c76c31c7b6d6c12b46bea76004aa7745cfada504378da606'
    targets=frozen/'targets.npz';assert sha(targets)==manifest['files']['targets.npz']
    stats_path=S/'e037-training-source-constraints-v1/all151-NEW-training-source-statistics.json'
    stats=read(stats_path);authority=read(D/'runtime-config-and-original-authority.json');caps=authority['original_object_caps']
    with np.load(D/'dynamic-source-data.npz')as z:
        C0=z['C0'];Rcw=z['camera_from_world_linear'];T=z['original_camera_from_object'];K=z['K'];faces=z['faces'];times=z['timestamps_s'];world=z['original_world_from_object'];indices=z['frame_indices']
    with np.load(D/'dynamic-axis-group-bindings.npz')as z:pair_frames=z['pair_frames'];ptr=z['ptr'];pair_geoms=z['pair_geoms']
    initial=S/'e037-object-dynamic-evaluation-v1/evaluation-000-NEW-union-support-RETURN.npz'
    with np.load(initial)as z:active=z['active_axis_indices'];gap=z['gap_proxy_m'];delta=z['world_translation']
    assert np.array_equal(delta,np.zeros((151,3)))
    axes=np.load(D/'ALL-retained-axis-world-coefficients.npy',mmap_mode='r')
    data=dict(motion_M=sparse.load_npz(D/'dynamic-motion-linear-map.npz').tocsc(),
      motion_base=np.load(D/'dynamic-motion-constants.npy'),
      motion_caps=np.r_[np.full(150,caps['maximum_linear_speed_m_s']),np.full(149,caps['maximum_linear_acceleration_m_s2'])],
      pair_frames=pair_frames,baseline_median=np.array([r['training_mesh_vertex_median_m']for r in stats]),
      Rcw=Rcw,minimum_C0_z=C0[:,:,2].min(axis=1))
    state=dict(world_delta=delta.copy(),active_normals=np.asarray(axes[active]).copy(),
      gap_proxy=gap.copy(),median=data['baseline_median'].copy(),
      median_gradient=np.array([r['local_median_world_gradient']for r in stats]),active_axis_indices=active.copy())
    mesh_path=frozen/manifest['objects'][0]['mesh'];V=np.asarray(trimesh.load(mesh_path,force='mesh',process=False).vertices,float)
    paths=[frozen/'manifest.json',targets,mesh_path,stats_path,D/'runtime-config-and-original-authority.json',D/'dynamic-source-data.npz',D/'dynamic-axis-group-bindings.npz',initial,D/'ALL-retained-axis-world-coefficients.npy',D/'ALL-retained-axis-saved-gaps.npy',D/'dynamic-motion-linear-map.npz',D/'dynamic-motion-constants.npy']
    provenance={str(p):sha(p)for p in paths}
    return dict(data=data,state=state,source=dict(V=V,faces=faces,K=K,C0=C0,T=T,Rcw=Rcw,stats=stats),
      geometry=dict(axes=axes,gaps0=np.load(D/'ALL-retained-axis-saved-gaps.npy',mmap_mode='r'),ptr=ptr,pair_frames=pair_frames,pair_geoms=pair_geoms),
      world_poses=world,timestamps=times,frame_indices=indices,caps=caps,frozen=frozen,provenance=provenance)

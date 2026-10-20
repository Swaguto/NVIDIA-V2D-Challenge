"""NEW synthetic full-runtime derivatives and independent immutable data bindings."""
from pathlib import Path
import datetime,hashlib,json,os,time
import numpy as np
from scipy import sparse
from object_dynamic_runtime_v1 import DynamicRuntime
from object_dynamic_visibility_v1 import CONFIG
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];O=S/'e037-object-dynamic-runtime-binding-v1';D=S/'e037-object-dynamic-runtime-v1';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=O/n;p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,v)
 emit(dict(kind='BEFORE_NEW_synthetic_runtime_directional_derivative_and_data_binding',runtime_source_sha256=sha(S/'compute/object_dynamic_runtime_v1.py'),NEW10contracts_reused_BY_HASH=sha(S/'e037-object-dynamic-contract-v1/report.json'),no_actual_episode_objective_or_old_queries_evaluated=True))
 P=O/'SYNTHETIC-only';P.mkdir();F=151;K=np.array([[100.,0,0],[0,100.,0],[0,0,1.]]);tri=np.array([[-1,-1,2],[1,-1,2],[0,1,2]],float);world=np.tile(np.eye(4),(F,1,1));np.save(P/'ALL-retained-axis-world-coefficients.npy',np.tile([[1.,0,0],[-1.,0,0]],(F,1)));np.save(P/'ALL-retained-axis-saved-gaps.npy',np.full(2*F,-.005));np.savez(P/'dynamic-axis-group-bindings.npz',ptr=np.arange(0,2*F+1,2),pair_frames=np.arange(F),pair_geoms=np.tile([[0,1]],(F,1)));np.savez(P/'dynamic-source-data.npz',C0=np.tile(tri,(F,1,1)),faces=np.array([[0,1,2]]),edges=np.array([[0,1],[0,2],[1,2]]),K=K,pixels=np.tile([[0.,0.],[90.,13.]],(F,1,1)),observations=np.tile([[2.05,2.1]],(F,1)),baseline_faces=np.tile([[0,-1]],(F,1)),baseline_depth=np.tile([[2.,np.inf]],(F,1)),baseline_depth_derivative=np.tile([[[0,0,1],[0,0,0]]],(F,1,1)),camera_from_world_linear=np.tile(np.eye(3),(F,1,1)),timestamps_s=np.arange(F,dtype=float),frame_indices=np.arange(F),original_world_from_object=world,original_camera_from_object=world)
 rr=[];cc=[];vv=[]
 for i in range(150):
  for k in range(3):
   for f,a in [(i,-1.),(i+1,1.)]:rr.append(3*i+k);cc.append(3*f+k);vv.append(a)
 for i in range(149):
  for k in range(3):
   for f,a in [(i,1.),(i+1,-2.),(i+2,1.)]:rr.append(450+3*i+k);cc.append(3*f+k);vv.append(a)
 Dsyn=sparse.csr_matrix((vv,(rr,cc)),shape=(897,453));sparse.save_npz(P/'dynamic-motion-linear-map.npz',Dsyn);m=np.zeros((299,3));m[:150,0]=.6+.001*np.arange(150);m[150:,1]=.2;np.save(P/'dynamic-motion-constants.npy',m.ravel());save('SYNTHETIC-only/runtime-config-and-original-authority.json',dict(config=CONFIG,original_object_caps=dict(maximum_linear_speed_m_s=.65,maximum_linear_acceleration_m_s2=.1,p95_linear_speed_m_s=.7)))
 def snpz(name,**kw):
  p=P/name;np.savez_compressed(p,**kw);p.chmod(0o600)
  with p.open('rb')as f:os.fsync(f.fileno())
  return str(p)
 runtime=DynamicRuntime(P,emit,snpz);x=np.tile([.1,.05,.02],151);runtime.start=time.monotonic();base=runtime.evaluate(x);J=base['J'];cache=runtime.evaluate(x);assert cache is base;errors=[];h=2e-6
 for a,phase in enumerate([.3,1.1]):
  v=np.sin(np.arange(453)*.37+phase)*.5;emit(dict(kind='BEFORE_NEW_synthetic_runtime_directional_check',direction_id=a,x=x,direction=v,epsilon=h,no_episode_data=True));plus=runtime.evaluate(x+h*v);minus=runtime.evaluate(x-h*v);fd=(plus['r']-minus['r'])/(2*h);expected=J@v;err=float(abs(fd-expected).max());scale=max(1.,float(abs(expected).max()));result=dict(direction=a,maximum_absolute_error=err,maximum_relative_to_global_scale=err/scale,independent_FD=fd,analytic_direction=expected,diagnostic_NOT_original_candidate_tolerance=True);save(f'NEW-synthetic-direction-{a}-RETURN.json',result);emit(dict(kind='AFTER_NEW_synthetic_runtime_directional_check_RETURN',actual_RETURN={k:v for k,v in result.items()if k not in ['independent_FD','analytic_direction']}));assert err/scale<2e-6;errors.append(err/scale)
 # The actual dataset is only bound/inspected, never evaluated at an episode pose.
 meta=read(D/'runtime-config-and-original-authority.json');assert meta['config']==CONFIG;datafiles=['ALL-retained-axis-world-coefficients.npy','ALL-retained-axis-saved-gaps.npy','dynamic-axis-group-bindings.npz','dynamic-source-data.npz','dynamic-motion-linear-map.npz','dynamic-motion-constants.npy'];axis=np.load(D/datafiles[0],mmap_mode='r');g0=np.load(D/datafiles[1],mmap_mode='r');maxnorm=0.;finite=True
 for i in range(0,len(axis),50000):
  values=np.asarray(axis[i:i+50000]);finite=finite and np.isfinite(values).all();maxnorm=max(maxnorm,float(abs(np.linalg.norm(values,axis=1)-1).max()))
 assert finite and maxnorm<1e-12
 with np.load(D/'dynamic-axis-group-bindings.npz')as z:ptr=z['ptr'];frames=z['pair_frames'];assert ptr[0]==0 and ptr[-1]==len(axis)==len(g0)==4832184 and len(ptr)==17521 and len(frames)==17520 and(np.diff(ptr)>0).all()
 actualD=sparse.load_npz(D/'dynamic-motion-linear-map.npz');old=sparse.load_npz(S/'e037-object-visibility-support-model-v3/canonical-A.npz').tocsr();idx=[26576+4*k+j for k in range(299)for j in [1,2,3]];assert (actualD+old[idx]).nnz==0
 with np.load(S/'e037-object-visibility-support-model-v3/canonical-vectors.npz')as z:assert np.array_equal(np.load(D/'dynamic-motion-constants.npy'),z['b'][idx])
 with np.load(D/'dynamic-source-data.npz')as z:
  assert z['C0'].shape==(151,6022,3)and z['edges'].shape==(18000,2)and z['pixels'].shape==(151,24,2)and (z['pixels'].sum(2)%2==0).all();assert np.count_nonzero(z['baseline_faces']<0)==1662 and np.isinf(z['baseline_depth'][z['baseline_faces']<0]).all();assert np.isfinite(z['observations']).all()and(z['observations']>0).all()
 hashes={n:sha(D/n)for n in datafiles};save('actual-runtime-data-bindings.json',dict(data_sha256=hashes,maximum_axis_world_norm_error=maxnorm,all_axes_finite=True,all399_motion_components_aliases_bound=True,actual_dataset_not_evaluated=True))
 out=dict(schema='v2d.NEW-dynamic-object-runtime-independent-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEW_full_runtime_directional_controls=2,NEW_fun_jac_cache_control=1,maximum_synthetic_directional_relative_errors=errors,old10contracts_reused_BY_HASH=True,actual_axis_directions=4832184,actual_pairs=17520,actual_axis_norm_error=maxnorm,motion_scalar_aliases=897,actual_even_training_samples=3624,old1662missing_rays_retained=True,no_actual_episode_function_FK_ray_solver_or_native_position_calls=True,original_authority_config_and_inputs_UNCHANGED=True,all_actual_runtime_bindings_pass=True);save('report.json',out);emit(dict(kind='AFTER_NEW_runtime_independent_binding',actual_RETURN=out));print(json.dumps(out))

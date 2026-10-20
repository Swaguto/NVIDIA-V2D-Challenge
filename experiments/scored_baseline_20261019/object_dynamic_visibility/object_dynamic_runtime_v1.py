"""NEW all-clock dynamic-face/union-support nonlinear construction runtime.

Only optimizer residuals are generated. Actual original source/native/motion
qualification is separate. No failed vector is ever accepted here.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy import sparse
from object_dynamic_visibility_v1 import CONFIG,robust_signed,projected_edge_distance,p95_value_gradient
from object_visibility_support_v1 import ray_cells

class MethodBudgetExceeded(RuntimeError):pass

class DynamicRuntime:
 def __init__(self,path,emit,save_npz):
  self.path=Path(path);self.emit=emit;self.save_npz=save_npz;self.meta=json.loads((self.path/'runtime-config-and-original-authority.json').read_text());assert self.meta['config']==CONFIG;self.caps=self.meta['original_object_caps'];self.axes=np.load(self.path/'ALL-retained-axis-world-coefficients.npy',mmap_mode='r');self.g0=np.load(self.path/'ALL-retained-axis-saved-gaps.npy',mmap_mode='r')
  with np.load(self.path/'dynamic-axis-group-bindings.npz')as z:self.ptr=z['ptr'];self.pairframes=z['pair_frames'];self.pairgeoms=z['pair_geoms']
  with np.load(self.path/'dynamic-source-data.npz')as z:
   for name in ['C0','faces','edges','K','pixels','observations','baseline_faces','baseline_depth','baseline_depth_derivative','camera_from_world_linear','timestamps_s','frame_indices','original_world_from_object','original_camera_from_object']:setattr(self,name,z[name].copy())
  self.D=sparse.load_npz(self.path/'dynamic-motion-linear-map.npz').tocsr();self.motion0=np.load(self.path/'dynamic-motion-constants.npy');self.frame_count=len(self.C0);self.rays_per_frame=self.pixels.shape[1];self.source_count=self.frame_count*self.rays_per_frame;self.n=self.frame_count*3;self.pair_count=len(self.pairframes);self.axis_frames=np.repeat(self.pairframes,np.diff(self.ptr));self.last=None;self.calls=0;self.start=None;self.callback_count=0
  self.emit(dict(kind='AFTER_NEW_dynamic_runtime_constructor_RETURN',columns=self.n,pairs=self.pair_count,face_directions=len(self.g0),source_count=self.source_count,old_FK_baseline_queries_NOT_replayed=True,actual_authority=self.caps))
 def evaluate(self,x):
  x=np.asarray(x,float)
  if self.last is not None and np.array_equal(x,self.last['x']):self.emit(dict(kind='AFTER_dynamic_fun_jac_cache_REUSED',evaluation=self.last['id'],same_x_exact=True));return self.last
  if self.start is None:self.start=time.monotonic()
  if time.monotonic()-self.start>CONFIG['method_seconds']:raise MethodBudgetExceeded('NEWmethod170second bound; no failed-vector promotion')
  i=self.calls;self.calls+=1;delta=CONFIG['construction_halfwidth_m']*x.reshape(self.frame_count,3);self.emit(dict(kind='BEFORE_NEW_dynamic_actual_model_proposal',evaluation=i,normalized_x=x,world_translation=delta,initial_baseline_RETURNS_REUSED_BY_HASH=bool(np.all(x==0)),NOT_source_reference_or_observation=True))
  if x.shape!=(self.n,)or not np.isfinite(x).all():raise ValueError('UNRESOLVED malformed NEWproposal')
  # Geometry functional permits every retained face-direction branch.
  av=self.g0+np.einsum('ni,ni->n',self.axes,delta[self.axis_frames],optimize=False);active=np.array([int(a)+int(np.argmax(av[a:b]))for a,b in zip(self.ptr[:-1],self.ptr[1:],strict=True)],np.int64);gaps=av[active];violation=np.maximum(-.001-gaps,0);native_r=CONFIG['native_weight']*violation/CONFIG['native_scale_m'];native_gradient=np.asarray(self.axes[active]).copy();native_gradient[violation==0]=0.;native_gradient*=-CONFIG['native_weight']*CONFIG['construction_halfwidth_m']/CONFIG['native_scale_m'];nr=np.repeat(np.arange(self.pair_count),3);nc=(3*self.pairframes[:,None]+np.arange(3)).ravel();Jn=sparse.csr_matrix((native_gradient.ravel(),(nr,nc)),shape=(self.pair_count,self.n));native_file=self.save_npz(f'evaluation-{i:03d}-NEW-union-support-RETURN.npz',active_axis_indices=active,gap_proxy_m=gaps,positive_violations_m=violation,world_translation=delta,normalized_x=x);self.emit(dict(kind='AFTER_NEW_dynamic_union_support_RETURN',evaluation=i,data_file=native_file,all_axes_retained=True,native_gap_OR_measured_force_NOT_claimed=True,violating_pair_count=int(np.count_nonzero(violation))))
  # Current source faces can change. Infinite missing depths are preserved.
  ds=[];cs=[];depths=np.full((self.frame_count,self.rays_per_frame),np.inf);faceids=np.full((self.frame_count,self.rays_per_frame),-1,np.int32);coverage=np.zeros_like(depths);jrows=[];jcols=[];jvalues=[];initial=bool(np.all(x==0));source_switches=0
  for f in range(self.frame_count):
   RC=self.camera_from_world_linear[f];C=self.C0[f]+np.einsum('ij,j->i',RC,delta[f],optimize=False)
   if np.any(C[:,2]<=.01):self.emit(dict(kind='AFTER_NEW_source_function_UNRESOLVED',evaluation=i,frame=f,reason='source vertices cross ORIGINAL near plane; no clipping'));raise ValueError('UNRESOLVED source near plane')
   pix=self.pixels[f];rays=np.column_stack([(pix[:,0]-self.K[0,2])/self.K[0,0],(pix[:,1]-self.K[1,2])/self.K[1,1],np.ones(len(pix))]);cells=None if initial else ray_cells(C[self.faces],rays);frame_records=[]
   for k in range(self.rays_per_frame):
    row=f*self.rays_per_frame+k;fid=int(self.baseline_faces[f,k])if initial else -1;z=float(self.baseline_depth[f,k])if initial else float('inf');zd=self.baseline_depth_derivative[f,k]if initial else np.zeros(3)
    if not initial:
     u,v,zz=cells['u0'][k],cells['v0'][k],cells['z0'][k];hit=cells['nonzero'][k]&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(zz>0);ids=np.flatnonzero(hit)
     if len(ids):fid=int(ids[np.argmin(zz[ids])]);z=float(zz[fid]);zd=cells['zd'][k,fid]
    faceids[f,k]=fid;depths[f,k]=z;source_switches+=int(fid!=self.baseline_faces[f,k]);missing=fid<0
    if missing:
     dep,_=robust_signed(CONFIG['missing_optimization_equivalent_m']/CONFIG['depth_scale_m']);Jd=np.zeros(3);edge=projected_edge_distance(C,self.edges,pix[k],self.K,RC);self.emit(dict(kind='AFTER_NEW_projected_union_edge_RETURN',evaluation=i,frame=f,ray=k,actual_RETURN=edge));assert edge['status']=='NEW_closest_projected_edge_distance';distance=edge['distance_px'];cov,dcov=robust_signed(distance/CONFIG['coverage_scale_px']);Jc=dcov*np.asarray(edge['world_translation_gradient'])*CONFIG['construction_halfwidth_m']/CONFIG['coverage_scale_px'];coverage[f,k]=distance
    else:
     actual_residual=(z-float(self.observations[f,k]))/CONFIG['depth_scale_m'];dep,ddep=robust_signed(actual_residual);Jd=ddep*np.einsum('i,ij->j',zd,RC,optimize=False)*CONFIG['construction_halfwidth_m']/CONFIG['depth_scale_m'];cov=0.;Jc=np.zeros(3);edge=None
    ds.append(dep);cs.append(cov)
    for j in range(3):
     if Jd[j]:jrows.append(row);jcols.append(3*f+j);jvalues.append(Jd[j])
     if Jc[j]:jrows.append(self.source_count+row);jcols.append(3*f+j);jvalues.append(Jc[j])
    frame_records.append(dict(ray_index=k,actual_face=fid,actual_depth_m=z,actual_model_hit=not missing,first_surface_reused_BY_HASH=initial,actual_missing_depth_NOT_filled=missing,optimizer_ONLY_depth_residual=dep,optimizer_ONLY_coverage_residual=cov,depth_jacobian_normalized_variable=Jd,coverage_jacobian_normalized_variable=Jc,source_truth_OR_original_source_gate_NOT_claimed=True))
   self.emit(dict(kind='AFTER_NEW_actual_dynamic_source_frame_RETURN',evaluation=i,frame=f,actual_RETURN=frame_records))
  Js=sparse.csr_matrix((jvalues,(jrows,jcols)),shape=(2*self.source_count,self.n));source_file=self.save_npz(f'evaluation-{i:03d}-NEW-source-RETURN.npz',actual_front_faces=faceids,actual_depth_m=depths,coverage_distance_px=coverage,source_depth_residual=np.asarray(ds),source_coverage_residual=np.asarray(cs),world_translation=delta)
  # All ORIGINAL maximum norms and actual p95, rather than the old inner envelope.
  packed=(self.motion0+self.D@delta.ravel()).reshape(-1,3);norms=np.linalg.norm(packed,axis=1);caps=np.r_[np.full(150,self.caps['maximum_linear_speed_m_s']),np.full(149,self.caps['maximum_linear_acceleration_m_s2'])];positive=np.maximum(norms-caps,0);motion_r=CONFIG['motion_weight']*positive/caps;factor=np.divide(CONFIG['motion_weight'],norms*caps,out=np.zeros_like(norms),where=(positive>0)&(norms>0));grads=factor[:,None]*packed;G=sparse.csr_matrix((grads.ravel(),(np.repeat(np.arange(299),3),np.arange(897))),shape=(299,897));Jm=CONFIG['construction_halfwidth_m']*(G@self.D)
  quantile=p95_value_gradient(norms[:150]);self.emit(dict(kind='AFTER_NEW_actual_dynamic_p95_RETURN',evaluation=i,actual_RETURN=quantile,original_cap=self.caps['p95_linear_speed_m_s']));qcap=self.caps['p95_linear_speed_m_s'];qexcess=max(quantile['value']-qcap,0.);qr=CONFIG['motion_weight']*qexcess/qcap;qgrad=np.zeros((1,897))
  if qexcess>0:
   for k,w in enumerate(quantile['gradient']):
    if w and norms[k]>0:qgrad[0,3*k:3*k+3]=CONFIG['motion_weight']*w*packed[k]/(qcap*norms[k])
  Jq=CONFIG['construction_halfwidth_m']*sparse.csr_matrix(qgrad)@self.D;r=np.r_[ds,cs,native_r,motion_r,qr];J=sparse.vstack([Js,Jn,Jm,Jq],format='csr');stats=dict(evaluation=i,function_seconds=time.monotonic()-self.start,objective=float(.5*np.dot(r,r)),source_model_hits=int(np.count_nonzero(faceids>=0)),source_model_misses=int(np.count_nonzero(faceids<0)),front_face_switches=source_switches,native_proxy_violating_pairs=int(np.count_nonzero(violation)),max_native_proxy_excess_m=float(violation.max(initial=0)),motion_norm_positive_rows=int(np.count_nonzero(positive)),actual_model_p95=quantile['value'],max_normalized_translation=float(abs(x).max(initial=0)),residual_rows=len(r),Jacobian_shape=J.shape,Jacobian_nonzeros=J.nnz,all_ORIGINAL_actual_source_native_motion_gates_STILL_PENDING=True)
  self.last=dict(id=i,x=x.copy(),r=r,J=J,stats=stats);data=self.save_npz(f'evaluation-{i:03d}-actual-method-residual-RETURN.npz',normalized_x=x,residual=r,motion_vectors=packed,motion_norms=norms);self.emit(dict(kind='AFTER_NEW_actual_dynamic_model_evaluation_RETURN',actual_RETURN=stats,residual_file=data,source_file=source_file));return self.last
 def fun(self,x):return self.evaluate(x)['r']
 def jac(self,x):return self.evaluate(x)['J']
 def callback(self,intermediate_result):
  self.callback_count+=1;r=intermediate_result;ret={k:(str(v)if callable(v)else v)for k,v in r.items()if k not in ['jac']};self.emit(dict(kind='AFTER_NEW_actual_optimizer_callback_RETURN',callback=self.callback_count,actual_RETURN=ret,no_intermediate_promoted=True))
  if time.monotonic()-self.start>CONFIG['method_seconds']:raise StopIteration

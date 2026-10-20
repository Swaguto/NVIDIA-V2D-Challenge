"""Independent NEW full-pose factor binding, not old FK/source/native replay."""
from pathlib import Path
import datetime,json,hashlib,os,math
import numpy as np,mujoco
from scipy import sparse
S=Path(__file__).resolve().parents[1];C=S/'compute';I=S/'e037-object-full-pose-factors-v1';O=S/'e037-object-full-pose-factor-binding-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def cv(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return{str(k):cv(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return[cv(v)for v in x]
 return x

def save(n,x):
 with (O/n).open('w')as f:json.dump(cv(x),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 (O/n).chmod(0o600)
def arrays(n,**x):
 with (O/n).open('wb')as f:np.savez_compressed(f,**x);f.flush();os.fsync(f.fileno())
 (O/n).chmod(0o600)
def anti(v):
 x,y,z=map(float,v);return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
# Seal read-only inputs before numerical identities.
inputs=[p for p in I.iterdir()if p.suffix in ['.npz','.json']]+[C/'object-authority-checkpoint-v1.json',C/'object-full-pose-factors-CPU-plan-v1.json']
save('NEW-binding-declaration-BEFORE.json',dict(inputs_sha256={str(p):sha(p)for p in inputs},scope='Only NEW rotation/source/full-pair/angular/column maps. Saved old source/NN/raster/FK/native results and baseline certificates not recomputed.',no_episode_nonzero_point_or_actual_gate_query=True))
decl=read(I/'NEW-declaration-BEFORE.json');semantic=read(I/'NEW-full-pose-factor-semantics-RETURN.json');static=read(I/'NEW-static-MJB-geometry-metadata-RETURN.json')
for name,digest in decl['provenance'].items():assert sha(name)==digest
with np.load(I/'NEW-source-rotation-factors-RETURN.npz')as z:
 V=z['visual_vertices'];R=z['original_camera_from_object'][:,:3,:3];J=z['camera_rotation_J_at_zero'];world=z['original_world_from_object'];times=z['original_timestamps_s'];indices=z['original_frame_indices']
# Component cross-product expansion independently binds every NEW coefficient.
expected=np.empty_like(J)
for a,(b,c)in enumerate([(1,2),(2,0),(0,1)]):expected[:,:,:,a]=V[None,:,b,None]*R[:,None,:,c]-V[None,:,c,None]*R[:,None,:,b]
error=J-expected;source_error=float(np.max(abs(error)));arrays('NEW-all-source-rotation-coefficient-residuals-RETURN.npz',residual=error);save('NEW-source-rotation-binding-RETURN.json',dict(coefficients=J.size,maximum_error=source_error,diagnostic_bound=2e-14,passed=source_error<2e-14,NOT_candidate_source_tolerance_or_camera_accuracy=True));assert source_error<2e-14
# Native body derivatives use SAVED compiled BODY vertices. No geom placement
# twice, no all-mesh_vert, no FK or native gap/position calls.
with np.load(I/'NEW-native-body-rotation-factors-RETURN.npz')as z:
 B=z['cached_body_poses'];native_errors={};native_count=0
 for g in [r['geom_id']for r in static['geoms']if r['body']=='wooden_spoon_object']:
  W=z[f'g{g}_vertices_body'];A=z[f'g{g}_rotation_J_at_zero'];E=np.empty_like(A);RR=B[:,g,:3,:3]
  for a,(b,c)in enumerate([(1,2),(2,0),(0,1)]):E[:,:,:,a]=W[None,:,b,None]*RR[:,None,:,c]-W[None,:,c,None]*RR[:,None,:,b]
  native_errors[f'g{g}']=A-E;native_count+=A.size
native_error=max(float(np.max(abs(a)))for a in native_errors.values());arrays('NEW-all-native-rotation-coefficient-residuals-RETURN.npz',**native_errors);save('NEW-native-rotation-binding-RETURN.json',dict(coefficients=native_count,maximum_error=native_error,diagnostic_bound=2e-14,passed=native_error<2e-14,no_new_native_measurement=True));assert native_error<2e-14
# Cartesian product independently binds all 66,591 cases with no broadphase.
fixed=[r['geom_id']for r in static['geoms']if r['body']!='wooden_spoon_object'];objects=[r['geom_id']for r in static['geoms']if r['body']=='wooden_spoon_object'];P=np.array([(f,a,b)for f in range(151)for a in fixed for b in objects],np.int32)
with np.load(I/'NEW-ALL-cross-pair-incidence-RETURN.npz')as z:actual=z['frame_fixed_object'];mask=z['collision_mask_eligible']
byid={r['geom_id']:r for r in static['geoms']};masks=np.array([bool((byid[a]['contype']&byid[b]['conaffinity'])or(byid[b]['contype']&byid[a]['conaffinity']))for _,a,b in P])
cols=[k for row in decl['variables']for name in ['world_translation_columns','object_rotation_columns']for k in row[name]]
save('NEW-full-coverage-column-binding-RETURN.json',dict(cross_pair_instances=len(P),all_pair_rows_equal=np.array_equal(P,actual),unique_pairs=len(set(map(tuple,actual))),mask_predicates_equal=np.array_equal(masks,mask),columns=cols,all906_columns_once=sorted(cols)==list(range(906)),all151clocks_UNASSIGNED=True,all66691_or_other_count_typo_NOT_used=True,broadphase_omissions=0));assert np.array_equal(P,actual)and np.array_equal(masks,mask)and sorted(cols)==list(range(906))
# Native exclusion signatures are NEW static metadata, recorded without any
# collision or body-state evaluation. Full cross coverage contains excluded
# cases as well; future search must declare its restriction, native gates rule.
mp=S/'e037-motion-constrained-wrist-native-v1/native-model.mjb';model=mujoco.MjModel.from_binary_path(str(mp));sigs=np.asarray(model.exclude_signature,dtype=np.int64);exclude_pairs=[(int(s)>>16,int(s)&65535)for s in sigs];object_body=int(next(r['body_id']for r in static['geoms']if r['body']=='wooden_spoon_object'));touch=[p for p in exclude_pairs if object_body in p]
save('NEW-static-native-exclusions-RETURN.json',dict(model_sha256=sha(mp),raw_exclude_signatures=sigs,decoded_body_pairs=exclude_pairs,object_body_id=object_body,exclusions_touching_object=touch,explicit_pairs=int(model.npair),all_cross_pairs_RETAINED_even_if_excluded=True,no_MjData_FK_native_position_or_distance=True))
# Independently derive left-log differential via trace/antisymmetric entries,
# rather than the producer's SciPy rotation composition/right-J inverse.
RR=world[:,:3,:3];dt=np.diff(times);mid=(dt[:-1]+dt[1:])/2;N=len(dt);JV=np.zeros((3*N,3*len(times)));phis=[]
for i in range(N):
 A=RR[i+1]@RR[i].T;vee=np.array([A[2,1]-A[1,2],A[0,2]-A[2,0],A[1,0]-A[0,1]])/2;sn=float(np.linalg.norm(vee));cs=float((np.trace(A)-1)/2);angle=math.atan2(sn,cs)
 p=vee if sn<1e-15 else vee*(angle/sn);phis.append(p);q=float(p@p);H=anti(p)
 if q<1e-8:coef=1/12+q/720+q*q/30240
 else:
  a=math.sqrt(q);coef=(1-.5*a/math.tan(.5*a))/q
 left_inverse=np.eye(3)-.5*H+coef*(H@H);block=left_inverse@RR[i+1]/dt[i];JV[3*i:3*i+3,3*i:3*i+3]=-block;JV[3*i:3*i+3,3*(i+1):3*(i+1)+3]=block
JA=np.vstack([(JV[3*(i+1):3*(i+2)]-JV[3*i:3*(i+1)])/mid[i]for i in range(N-1)]);E=np.vstack([JV,JA]);A=sparse.load_npz(I/'NEW-original-angular-clock-Jacobian-RETURN.npz').toarray();delta=A-E
with np.load(I/'NEW-angular-clock-base-factors-RETURN.npz')as z:pdiff=z['principal_relative_rotvec']-np.array(phis)
angular_error=float(np.max(abs(delta)));phi_error=float(np.max(abs(pdiff)));arrays('NEW-all-angular-coefficient-residuals-RETURN.npz',residual=delta,principal_rotvec_residual=pdiff);save('NEW-angular-original-clock-binding-RETURN.json',dict(new_coefficient_nonzeros=int(np.count_nonzero(A)),all_dense_cells=A.size,maximum_error=angular_error,principal_rotvec_error=phi_error,diagnostic_coefficient_bound=2e-10,diagnostic_rotvec_bound=2e-14,original_motion_comparison_UNCHANGED=1e-9,original_actual_p95_still_required=True,passed=angular_error<2e-10 and phi_error<2e-14));assert angular_error<2e-10 and phi_error<2e-14
# NEW source rotation search gradients, all old NN/rank/distance values reused
# by exact path/SHA. This does not freeze future correspondences or source truth.
stats=read(S/'e037-training-source-constraints-v1/all151-NEW-training-source-statistics.json');expected=[]
for f,st in enumerate(stats):
 path=Path(st['median_data']['path']);assert sha(path)==st['median_data']['sha256']
 with np.load(path)as z:
  values=[[],[],[]]
  for idx,w in zip(z['central_point_indices'],z['central_weights'],strict=True):
   n=z['nearest_vertex_indices'][idx];distance=z['nearest_vertex_distances'][idx]
   if distance==0:continue
   y=z['point_local'][idx];u=(y-V[n])/distance
   for j,(b,c)in enumerate([(1,2),(2,0),(0,1)]):values[j].append(float(w)*(float(u[b])*float(y[c])-float(u[c])*float(y[b])))
  expected.append([math.fsum(v)for v in values])
with np.load(I/'NEW-training-median-rotation-SEARCH-gradient-RETURN.npz')as z:difference=z['rotation_SEARCH_gradient_at_F']-np.array(expected)
grad_error=float(np.max(abs(difference)));arrays('NEW-training-rotation-SEARCH-gradient-residuals-RETURN.npz',residual=difference);save('NEW-training-search-scope-binding-RETURN.json',dict(frames=151,coefficients=453,maximum_error=grad_error,diagnostic_bound=2e-14,passed=grad_error<2e-14,actual_NN_rank_rounded_median_quality_fulltriangles_mandatory=True,no_IoU_derivative_or_fixed_source_truth=True));assert grad_error<2e-14
# Existing caps/source tolerances are copies; do not rerun old authority gates.
authority=read(C/'object-authority-checkpoint-v1.json');copied=semantic['all_original_caps_COPIED']==authority['original_object_motion_caps'];save('NEW-authority-and-actual-acceptance-binding-RETURN.json',dict(six_original_caps_copied=copied,original_object_motion_comparison=semantic['object_motion_comparison'],original_fullpixel_source_POST_only_comparison=semantic['original_fullpixel_source_POST_only_comparison'],training_operators_only_NEW_nonzero_inputs_and_zero_cache=True,no_fullpixel_heldout_fitting_selection=True,pi_branch_unavailable_or_nonfinite_returns_UNRESOLVED=True,no_solver_or_complete_canonical_model_selected=True));assert copied and semantic['object_motion_comparison']==1e-9 and semantic['original_fullpixel_source_POST_only_comparison']==1e-10
save('report.json',dict(schema='v2d.NEW-full-pose-independent-factor-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEW_source_rotation_coefficients=J.size,NEW_native_rotation_coefficients=native_count,NEW_angular_sparse_coefficients=int(np.count_nonzero(A)),NEW_full_cross_pairs=len(P),NEW_columns=906,NEW_training_rotation_predictor_coefficients=453,errors=dict(source=source_error,native=native_error,angular=angular_error,principal_rotvec=phi_error,rotation_search_gradient=grad_error),actual_original_source_or_motion_gate_replay=False,no_episode_pose_source_NN_raster_FK_native_solver_physics_GPU=True,original_gates_F_separate_components_source_accuracy_missing136_heldout31of32_UNCHANGED=True));print(json.dumps(dict(success=True,source_coefficients=J.size,native_coefficients=native_count,angular_error=angular_error,cross_pairs=len(P))))

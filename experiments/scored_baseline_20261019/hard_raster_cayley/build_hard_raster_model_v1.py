"""ONE NEW streamed all-vertex integer-cell canonical. No solve/source replay."""
from pathlib import Path
import datetime,json,hashlib,os,time
import numpy as np
from scipy import sparse
from hard_raster_cayley_v1 import curvature_scale,frame_rows
S=Path(__file__).resolve().parents[1];C=S/'compute';F=S/'e037-object-full-pose-factors-v1';B=S/'e037-full-pose-preflight-v1';O=S/'e037-hard-raster-model-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def save(n,d):
 with (O/n).open('w')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def arrays(n,d):
 with (O/n).open('wb')as f:np.savez_compressed(f,**d);f.flush();os.fsync(f.fileno())
 return dict(path=str(O/n),sha256=sha(O/n))
with np.load(F/'NEW-source-rotation-factors-RETURN.npz')as z:V=z['visual_vertices'];T=z['original_camera_from_object'];L=z['camera_from_world_translation_linear'];K=z['K'];times=z['original_timestamps_s'];indices=z['original_frame_indices']
stats=read(S/'e037-training-source-constraints-v1/all151-NEW-training-source-statistics.json');base_meta=read(B/'NEW-canonical-metadata-RETURN.json');r=base_meta['radius'];bound=curvature_scale(T[:,:3,:3],T[:,:3,3],V,K,r);s=bound['scale'];assert bound['exact_scale_square_bound'];N=151;nv=len(V);hard_per=5*nv+1;hard_count=N*hard_per;oldnn=base_meta['nonnegative'];newnn=oldnn+N+hard_count;oldA=sparse.load_npz(B/'NEW-canonical-A-RETURN.npz');oldP=sparse.load_npz(B/'NEW-canonical-P-RETURN.npz');oldrows=oldA.shape[0];n=7*N+1;totalrows=oldrows+N+hard_count+5*N;source_start=oldnn+N;psi_start=newnn+(oldrows-oldnn)
with np.load(B/'NEW-canonical-vectors-RETURN.npz')as z:oldb=z['b'];oldq=z['q']
paths=[F/'NEW-source-rotation-factors-RETURN.npz',B/'NEW-canonical-A-RETURN.npz',B/'NEW-canonical-P-RETURN.npz',B/'NEW-canonical-vectors-RETURN.npz',B/'NEW-canonical-metadata-RETURN.json',S/'e037-training-source-constraints-v1/all151-NEW-training-source-statistics.json',C/'free-rank-raster-checkpoint-v1.json',C/'full-pose-construction-checkpoint-v1.json']
save('NEW-hard-raster-construction-BEFORE.json',dict(provenance={str(p):sha(p)for p in paths},variables='all151 frames eta/r, u=s*t, sigma>=u.u; finalrho/r',Cayley_rotation='Q=((1-t.t)I+2ttT+2[t]x)/(1+t.t)',translation='delta=eta/(1+t.t)',all3_rotation_components_FREE=True,all_rows_include_ALLvisual_vertices_not_new_source_pixels=True,radius=r,curvature_scale=bound,strict_faces_closed_ONLY_for_proposal_actual_cells_and_strict_depth_hard_REQUIRED=True,lambda_proof_conditional_saved_binary_inputs_NOT_global_numeric_source_truth=True,no_initial_F_exact_numeric_cell_feasibility_assumed=True,old_C0_alias_nominal_defects_NOT_repaired=True,all_NN_rank_and_rounded_median_quality_actual_gates_REQUIRED=True,original_fullpixel_heldout_POST_only=True,no_old_model_or_source_producer_replay=True,all_original_physical_parameters_and_gates_UNCHANGED=True,planned_rows=totalrows,planned_columns=n,planned_source_rows=hard_count,one_future_solve_only_AFTER_independent_complete_matrix_binding=True))
# Direct CSC allocation, no giant Python per-row lists or COO copies.
mapcol=np.array([7*(j//6)+(j%6)for j in range(906)]+[1057]);factor=np.array([1. if j%6<3 else 2/s for j in range(906)]+[1.]);counts=np.zeros(n,np.int64)
for j in range(907):counts[mapcol[j]]+=oldA.indptr[j+1]-oldA.indptr[j]
for f in range(N):counts[7*f:7*f+7]+=hard_per;counts[7*f+3:7*f+6]+=1;counts[7*f+6]+=3
indptr=np.r_[0,np.cumsum(counts)].astype(np.int32);data=np.empty(int(indptr[-1]),np.float64);rowidx=np.empty(len(data),np.int32);cursor=indptr[:-1].astype(np.int64).copy();rhs=np.empty(totalrows,np.float64)
def put(col,rows,values):
 pos=cursor[col];m=len(rows);rowidx[pos:pos+m]=rows;data[pos:pos+m]=values;cursor[col]+=m
for j in range(907):
 a,b=int(oldA.indptr[j]),int(oldA.indptr[j+1]);rows=oldA.indices[a:b].copy();rows[rows>=oldnn]+=N+hard_count;put(int(mapcol[j]),rows,oldA.data[a:b]*factor[j])
rhs[:oldnn]=oldb[:oldnn];rhs[newnn:newnn+oldrows-oldnn]=oldb[oldnn:]
nominal_negative=0;inactive_total=0;strict_total=0;frame_meta=[]
for f in range(N):
 target_path=Path(stats[f]['raster']['path']);assert sha(target_path)==stats[f]['raster']['sha256']
 with np.load(target_path)as z:target=z['uv_integer']
 out=frame_rows(T[f,:3,:3],T[f,:3,3],L[f],V,K,target,r,s);coef=np.vstack([out['coefficients'],np.r_[-L[f,2],np.zeros(3),1.] ]);bb=np.r_[out['rhs'],(T[f,2,3]-.05)/r];strict=np.r_[out['strict_open'],True];active=np.r_[out['active'],True];kinds=np.r_[out['row_kind'],6];ret=arrays(f'frame-{f:03d}-NEW-hard-cell-coefficients-RETURN.npz',dict(coefficients=coef,rhs=bb,strict_open=strict,active=active,row_kind=kinds,target_integer_uv=target));frame_meta.append(dict(frame=f,source_raster_return=str(target_path),source_raster_sha256=sha(target_path),new_RETURN=ret,nominal_negative_constants=int(np.count_nonzero(bb<0)),inactive_rows=int(np.count_nonzero(~active)),strict_open_faces=int(strict.sum())))
 # Constructor RETURN is durable before any checks/next frame.
 save(f'frame-{f:03d}-NEW-hard-cell-constructor-RETURN.json',frame_meta[-1]);assert np.isfinite(coef).all()and np.isfinite(bb).all()
 rows=np.arange(source_start+f*hard_per,source_start+(f+1)*hard_per,dtype=np.int32);rhs[rows]=bb
 for j in range(7):put(7*f+j,rows,coef[:,j])
 row=oldnn+f;put(7*f+6,np.array([row],np.int32),np.array([1.]));rhs[row]=(s*np.pi/24)**2
 row=psi_start+5*f;put(7*f+6,np.array([row,row+4],np.int32),np.array([-1.,-1.]));rhs[row]=1.;rhs[row+4]=-1.
 for j in range(3):put(7*f+3+j,np.array([row+1+j],np.int32),np.array([-2.]));rhs[row+1+j]=0.
 nominal_negative+=frame_meta[-1]['nominal_negative_constants'];inactive_total+=frame_meta[-1]['inactive_rows'];strict_total+=frame_meta[-1]['strict_open_faces']
assert np.array_equal(cursor,indptr[1:]);A=sparse.csc_matrix((data,rowidx,indptr),shape=(totalrows,n));A.sort_indices();diag=np.zeros(n);old_diag=oldP.diagonal()
for j in range(907):diag[mapcol[j]]=old_diag[j]*factor[j]**2
P=sparse.diags(diag,format='csc');q=np.zeros(n);q[-1]=1.;sparse.save_npz(O/'NEW-canonical-A-RETURN.npz',A);sparse.save_npz(O/'NEW-canonical-P-RETURN.npz',P);arrays('NEW-canonical-vectors-RETURN.npz',dict(b=rhs,q=q));save('NEW-all151-frame-bindings-RETURN.json',frame_meta)
metadata=dict(rows=totalrows,columns=n,nonnegative=newnn,old_nonnegative=oldnn,hard_source_start=source_start,hard_source_rows_per_frame=hard_per,hard_source_rows=hard_count,psi_SOC_start=psi_start,old_SOC4=900,NEW_SOC5=151,old_cones_BY_HASH=base_meta['cones'],radius=r,scale=s,curvature_scale=bound,columns_per_frame=7,rho_column=n-1,nominal_negative_source_constants=nominal_negative,nominal_negative_status='RETAINED; not initialF numeric feasibility or original gate failure',inactive_saturation_rows_retained=inactive_total,strict_open_proposal_faces=strict_total,canonical_nnz=A.nnz,canonical_storage_bytes=A.data.nbytes+A.indices.nbytes+A.indptr.nbytes,solver_QP_max_iterations=100,solver_seconds=60.,solver_threads=1,default_tolerances=True,no_warmstart_linefraction_restart=True,resource_guard_seconds=240,resource_guard_RSS=2147483648,minimum_free_disk=1610612736)
save('NEW-canonical-metadata-RETURN.json',metadata);save('report.json',dict(schema='v2d.NEW-hard-raster-model.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=totalrows,columns=n,nonnegative=newnn,SOC4=900,SOC5=151,hard_source_rows=hard_count,canonical_nnz=A.nnz,canonical_storage_bytes=metadata['canonical_storage_bytes'],nominal_negative_source_constants=nominal_negative,strict_open_source_faces=strict_total,all151frames_6022visual_vertices_immutable=True,all_integer_cell_targets_reused_BY_HASH=True,old_completed_source_renderer_NN_model_solver_NOT_replayed=True,no_episode_pose_or_solver_evaluated=True,wall_seconds=time.monotonic()-start));print(json.dumps(dict(rows=totalrows,columns=n,nnz=A.nnz,nominal_negative=nominal_negative,wall=time.monotonic()-start)))

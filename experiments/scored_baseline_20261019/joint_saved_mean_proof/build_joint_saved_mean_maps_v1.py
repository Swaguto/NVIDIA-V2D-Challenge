"""NEW complete proof-incidence maps on SAVED first-query returns only."""
from joint_saved_mean_bound_v1 import *
from joint_lossless_returns_v1 import seal_return
D=S/'e037-joint-mean-proof-maps-v1';assert json.loads((S/'e037-joint-mean-proof-contracts-v1/report.json').read_text())['passed']
ff=frame_files();df=dynamic_file();inputs=[df,df.with_suffix('.npz'),*ff,*[p.with_suffix('.npz') for p in ff],C/'joint_saved_mean_bound_v1.py',S/'e037-joint-mean-proof-contracts-v1/report.json']
save(D/'NEW-ALL-saved-proof-input-bindings-RETURN.json',{str(p):sha(p) for p in inputs})
v=typed(df);J=v['J'];E=v['absolute_tie_envelope'];nnz=J.nnz;col=np.repeat(np.arange(J.shape[1],dtype=np.int64),np.diff(J.indptr));rows=J.indices.copy()
kind=np.zeros(nnz,dtype=np.int8);local=np.full(nnz,-1,dtype=np.int32);frame=np.full(nnz,-1,dtype=np.int16);pair=np.full(nnz,-1,dtype=np.int16)
order=np.argsort(rows,kind='stable');rp=np.r_[0,np.cumsum(np.bincount(rows,minlength=J.shape[0]))];counts=np.zeros((151,1653),dtype=np.int16);commands=0;allpairs=0
for f,p in enumerate(ff):
 a=typed(p)
 for k in range(1653):
  row=852+f*1653+k;positions=order[rp[row]:rp[row+1]];counts[f,k]=len(positions);allpairs+=1
  start,end=map(int,a['command_ptr'][k:k+2]);lookup={int(a['columns'][i]):i for i in range(start,end)}
  for pos in positions:
   c=int(col[pos]);frame[pos]=f;pair[pos]=k
   if c==1522:kind[pos]=1
   elif c>=1523:kind[pos]=3;local[pos]=c-(1523+2*(f*1653+k))
   else:kind[pos]=2;local[pos]=lookup[c];commands+=1
seal_return(D,'NEW-complete-proof-maps-RETURN',dict(kind=kind,local=local,frame=frame,pair=pair,columns=col,rows=rows,all_pair_entry_counts=counts,
 entry_order_by_row=order,rowptr=rp,J_data_bytes_sha256=hashlib.sha256(J.data.tobytes()).hexdigest(),E_data_bytes_sha256=hashlib.sha256(E.data.tobytes()).hexdigest()))
assert len(ff)==151 and nnz==933560 and J.shape==(250455,500729) and np.array_equal(J.indices,E.indices) and np.array_equal(J.indptr,E.indptr)
assert allpairs==249603 and np.all((local[kind==3]>=0)&(local[kind==3]<2))
assert np.all((rows<852)==(kind==0))
declaration=dict(schema='v2d.NEW-saved-all-tie-mean-proof-declaration.v1',ALL933560_dynamic_stored_entries=True,ALL249603_pair_scopes_retained=True,ALL4810140_original_rows_RHS_Pq_conditioning_strict_flags_UNCHANGED_BY_HASH=True,
 exact_certificate_bit_cap=8192,outward_binary_significand_bits=96,certificate_wall_seconds=180,outer_guard_seconds=240,RSS_bytes=2*1024**3,disk_reserve_bytes=int(1.5*1024**3),
 mean_operation_bound='m=2*n+16; arbitrary reduction association using <=m RN scalar operations; no overflow; absolute subnormal allowance retained',
 dot_operation_bound='3 exact saved-binary products define ideal proof dots; gamma_(2*d+16) absolute dot allowance; unknown unsaved numeric intermediates are NOT reconstructed',
 composed_bound='B=max_ALL|ideal_A_tie|+max_ALL|ideal_B_tie|+Gamma_means_products_and_final_addition; B independent of observed J',
 actual_check='every exact |SAVED_J-ideal_mean|<=Gamma and |SAVED_J|<=B<=96bit_outward_B',
 source_phase='original saved source/depth/domain coefficients exact abs copies; original phase constant exact abs; no old source operators replayed',
 no_global_NumPy_BLAS_compiler_binary_source_accuracy_identity=True,no_candidate_tolerance_gate_or_old_J_E_repair=True,no_optimizer_episode_method_geometry_source_native_query=True,
 serialization_per_frame='all exact ideal/tie/Gamma/B rational terms durable BEFORE checks; ALL entries routed once; chunks NOT new source partitions',
 no_certificate_retry_tuning_or_parameter_sweep=True,no_new_reference_qualification=True)
save(D/'NEW-fixed-proof-declaration-RETURN.json',declaration)
save(D/'report.json',dict(passed=True,ALL_stored_entries=nnz,ALL_pair_scopes=allpairs,command_entries=commands,kinds={str(k):int(np.count_nonzero(kind==k)) for k in range(4)},no_new_numerical_method_or_episode_query=True))

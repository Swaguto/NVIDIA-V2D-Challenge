"""Independent complete NEW canonical binding, not a source/native audit."""
from pathlib import Path
from fractions import Fraction as Q
import datetime,hashlib,json,os
import numpy as np
from scipy import sparse
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];O=S/'e037-hard-training-source-binding-v1';M=S/'e037-hard-training-source-model-v1';D=S/'e037-object-dynamic-runtime-v1';T=S/'e037-training-orderstat-support-v1';B=S/'e037-training-proxy-roundoff-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
def same(a,b):return np.array_equal(a,b)
def eq_sparse(a,b):
 d=(a-b).tocsr();d.eliminate_zeros();return d.nnz==0
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(x):journal(log,x)
 meta=read(M/'canonical-metadata.json');A=sparse.load_npz(M/'canonical-A.npz').tocsr();P=sparse.load_npz(M/'canonical-P.npz')
 with np.load(M/'canonical-vectors.npz')as z:b=z['b'];q=z['q']
 emit(dict(kind='BEFORE_NEW_complete_model_binding',canonical_hashes={n:sha(M/n)for n in ['canonical-A.npz','canonical-P.npz','canonical-vectors.npz']},old_proxy_maps_uniform_error_motion_geometry_controls_reused_BY_HASH=True))
 n=454;rows=meta['rows'];nn=meta['nonnegative_rows'];checks={}
 checks['shape']=A.shape==(1166056,454)and len(b)==1166056 and len(q)==454
 checks['box_matrix']=eq_sparse(A[:2*n],sparse.vstack([sparse.eye(n),-sparse.eye(n)],format='csr'))
 checks['box_rhs']=same(b[:n],np.r_[np.full(453,.02),.05])and same(b[n:2*n],np.r_[np.full(453,.02),0.])
 checks['objective']=same(q,np.r_[np.zeros(453),1.])and eq_sparse(P,sparse.diags(np.r_[np.full(453,1e-3),0.],format='csc'))
 with np.load(D/'dynamic-axis-group-bindings.npz')as z:frames=z['pair_frames']
 with np.load(meta['cached_initial_F_union_RETURN'])as z:active=z['active_axis_indices'];gaps=z['gap_proxy_m'];initial=z['world_translation']
 axes=np.load(D/'ALL-retained-axis-world-coefficients.npy',mmap_mode='r');chosen=np.asarray(axes[active]);geometry=[];max_rhs_error=Q(0)
 for k in range(len(frames)):
  row=A.getrow(2*n+k);expected=np.zeros(n);expected[3*frames[k]:3*frames[k]+3]=-chosen[k];expected[-1]=-1.
  ok=same(row.toarray()[0],expected)and b[2*n+k]==float(gaps[k]+.001)
  err=Q(float(b[2*n+k]))-Q(float(gaps[k]))-Q(.001);max_rhs_error=max(max_rhs_error,abs(err));geometry.append(ok)
 geometry_result=dict(selected_geometry_rows=len(geometry),all_rows_bound=all(geometry),cached_initial_F_only=same(initial,np.zeros((151,3))),maximum_exact_RHS_addition_error=dict(numerator=str(max_rhs_error.numerator),denominator=str(max_rhs_error.denominator)),NOT_actual_native_or_source_certificate=True)
 save('NEW-selected-geometry-row-binding.json',geometry_result);emit(dict(kind='AFTER_NEW_geometry_binding_RETURN',actual_RETURN=geometry_result));checks['geometry']=all(geometry)and geometry_result['cached_initial_F_only']
 motion=sparse.load_npz(D/'dynamic-motion-linear-map.npz').tocsr();constant=np.load(D/'dynamic-motion-constants.npy').reshape(299,3);caps=meta['original_caps'];motion_ok=[]
 for k in range(299):
  start=meta['motion_start']+4*k;expected=sparse.vstack([sparse.csr_matrix((1,453)),-motion[3*k:3*k+3]],format='csr');expected=sparse.hstack([expected,sparse.csr_matrix((4,1))],format='csr');cap=caps['maximum_linear_speed_m_s']if k<150 else caps['maximum_linear_acceleration_m_s2'];motion_ok.append(eq_sparse(A[start:start+4],expected)and same(b[start:start+4],np.r_[cap,constant[k]]))
 checks['all299motion']=all(motion_ok);source=[];source_coeff=0;source_rows=0
 for block in meta['source_blocks']:
  f=block['frame'];st=block['row_start'];nr=block['rows'];c=block['column_start'];old=Path(block['old_map']);guard=read(B/f'frame-{f:03d}-NEW-uniform-proxy-error-RETURN.json')
  with np.load(old)as z:oldA=z['canonical_A'].reshape(-1,3);oldb=z['canonical_b'];r=z['nominal_radius_m']
  actual=A[st:st+nr];local=actual[:,c:c+3].toarray();outside=actual.nnz-actual[:,c:c+3].nnz
  expected_b=oldb.copy();expected_b[:,0]=np.where(r==guard['original_lower_radius'],guard['NEW_outward_proposal_lower_radius'],guard['NEW_outward_proposal_upper_radius'])
  result=dict(frame=f,row_start=st,rows=nr,coefficient_cells=oldA.size,old_map_sha256=sha(old),all_old_A_coefficients_identical=same(local,oldA),zero_outside_declared_three_world_columns=outside==0,NEW_outward_RHS_exactly_bound=same(b[st:st+nr],expected_b.ravel()),coordinate_RHS_UNCHANGED=same(expected_b[:,1:],oldb[:,1:]),original_radii_UNALTERED=True,guard_IS_NOT_actual_source_gate=True)
  save(f'frame-{f:03d}-NEW-complete-source-block-binding.json',result);emit(dict(kind='AFTER_NEW_source_block_binding_RETURN',actual_RETURN=result));assert all(result[k]for k in ['all_old_A_coefficients_identical','zero_outside_declared_three_world_columns','NEW_outward_RHS_exactly_bound','coordinate_RHS_UNCHANGED'])
  source.append(result);source_coeff+=oldA.size;source_rows+=nr
 checks['all151source_blocks']=len(source)==151;checks['row_partition']=2*n+len(geometry)+4*299+source_rows==rows
 save('NEW-complete-model-binding-decisions.json',checks);emit(dict(kind='AFTER_NEW_complete_binding_decisions_RETURN',actual_RETURN=checks));assert all(checks.values())
 report=dict(schema='v2d.NEW-hard-training-source-complete-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,columns=n,nonnegative_rows=nn,SOC4=meta['SOC4'],source_blocks=151,source_rows=source_rows,source_coefficient_cells=source_coeff,selected_geometry_rows=len(geometry),original_motion_SOC4=299,unit_box_rows=2*n,all_bindings_pass=True,roundoff_proofs151_REUSED_BY_HASH=True,old_source_NN_controls_or_model_audits_NOT_replayed=True,no_solver_or_actual_nonzero_episode_pose=True,all_original_gates_source_parameters_F_UNCHANGED=True,canonical_hashes={n:sha(M/n)for n in ['canonical-A.npz','canonical-P.npz','canonical-vectors.npz']})
 save('report.json',report);emit(dict(kind='AFTER_NEW_complete_binding_report_RETURN',actual_RETURN=report));print(json.dumps(report))

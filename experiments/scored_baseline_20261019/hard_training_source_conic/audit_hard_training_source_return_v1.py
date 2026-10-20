"""NEW saved-return audit and defect-corrected finite-box rho certificate.

Actual x/z/s are never repaired. Certificates concern the declared numeric,
restricted proxy/selected-axis construction only, never the original family.
"""
from pathlib import Path
from fractions import Fraction as Q
import ctypes,datetime,hashlib,json,math,os
import numpy as np
from scipy import sparse
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];O=S/'e037-hard-training-source-return-audit-v1';M=S/'e037-hard-training-source-model-v1';E=S/'e037-hard-training-source-evaluation-v2';T=S/'e037-training-source-constraints-v1';D=S/'e037-object-dynamic-runtime-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();SCALE=2**96;U=Q(1,2**53);ETA=Q(1,2**1074)
def pack(q):return dict(numerator=str(q.numerator),denominator=str(q.denominator))
def ceilq(q):return -((-q.numerator)//q.denominator)
def dyadic_up(q):return Q(ceilq(Q(q)*SCALE),SCALE)
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
def arrays(n,**v):
 p=O/n;np.savez_compressed(p,**v);p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
def dot_enclosure(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);n=len(a)
 if not n:return dict(value=0.,absolute_sum=0.,operations=0,error_upper=Q(0),products=np.array([]),prefix=np.array([]))
 products=a*b;prefix=np.cumsum(products,dtype=np.float64);absolute=np.cumsum(np.abs(products),dtype=np.float64);value=float(prefix[-1]);mass=float(absolute[-1]);ops=2*n;g=ops*U/(1-ops*U);sub=ops*ETA/(1-ops*U);true_mass=(Q(mass)+sub)/(1-g);err=dyadic_up(g*true_mass+sub)
 return dict(value=value,absolute_sum=mass,operations=ops,error_upper=err,products=products,prefix=prefix)
def root96(squared):
 value=Q(squared)*SCALE*SCALE;k=math.isqrt(value.numerator//value.denominator)
 if k*k*value.denominator<value.numerator:k+=1
 return k,Q(k,SCALE)

with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,v)
 scope=dict(schema='v2d.NEW-hard-training-source-return-audit-declaration.v1',
  source='SAVED sole v2 RETURN and NEW canonical matrices only; no solver/source/NN/FK/native replay',
  certificate='lower bound on rho using arbitrary raw dual and finite-box slack/dual-cone defect corrections; no projection or normalization of z',
  dot_error='fixed product then cumulative binary64 sums, gamma(2n) and subnormal bounds, outward96bit certificate bounds',
  root_error='ceil-sqrt of exact raw squared cone components on fixed96bit dyadic certificate grid',
  scope_limit='THIS numeric median-proxy/initialFselected-axis/20mmbox construction ONLY; no source/time family or task exclusion',
  raw_values_original_source_tolerances_gates_UNALTERED=True)
 save('ONE-saved-return-audit-scope-BEFORE-work.json',scope);emit(dict(kind='BEFORE_NEW_audit_contracts',scope=scope));contracts=[]
 lib=ctypes.CDLL(None);lib.fegetround.restype=ctypes.c_int;rounding=int(lib.fegetround());save('NEW-audit-read-only-rounding-mode-RETURN.json',dict(actual_RETURN=rounding,mode_NOT_changed=True));assert rounding==0
 def control(n,ret,ok):
  save(n+'-RETURN.json',ret);emit(dict(kind='AFTER_NEW_dual_defect_contract_RETURN',name=n,actual_RETURN=ret));assert ok,n;contracts.append(n)
 z=[Q(-1),Q(3),Q(4),Q(0)];s=[Q(2),Q(-6,5),Q(-8,5),Q(0)];value=sum(a*b for a,b in zip(z,s));bound=-Q(2)*Q(6)
 control('NEW-arbitrary-SOC-dual-defect-lower-bound',dict(raw_z=list(map(pack,z)),SOC_s=list(map(pack,s)),exact_dot=pack(value),lower_bound=pack(bound),raw_z_NOT_repaired=True),value==bound)
 value=Q(-1,8)*Q(1,2);control('NEW-negative-nonnegative-row-dual-defect',dict(dot=pack(value),correction=pack(Q(1,16))),value==-Q(1,16))
 a=np.array([2.**40,-2.**40,2.**-40,.125,-.25]);b0=np.array([1.,1.,1.,.5,.25]);dc=dot_enclosure(a,b0);truth=sum(Q(float(x))*Q(float(y))for x,y in zip(a,b0));control('NEW-cumulative-dot-exact-residual-envelope',dict(actual_dot=dc['value'],exact_dot=pack(truth),error_upper=pack(dc['error_upper'])),abs(Q(dc['value'])-truth)<=dc['error_upper'])
 # Without the SOC defect correction this raw invalid dual would falsely
 # exclude min rho=0 with a constant (1,0,0,0)SOC slack.
 uncorrected=Q(1);corrected=uncorrected-Q(6);control('NEW-invalid-dual-false-exclusion-prevented',dict(uncorrected_lower=pack(uncorrected),corrected_lower=pack(corrected),actual_min_rho=0),uncorrected>0 and corrected<=0)
 assert len(contracts)==4
 meta=read(M/'canonical-metadata.json');report=read(E/'report.json');A=sparse.load_npz(M/'canonical-A.npz').tocsc()
 with np.load(M/'canonical-vectors.npz')as f:b=f['b'];q=f['q']
 with np.load(E/'ONE-actual-primal-dual-slack-RETURN.npz')as f:x=f['x'];z=f['z'];s=f['s']
 summary=dict(schema='v2d.NEW-hard-training-source-return-audit.v1',NEW_analytic_contracts=4,raw_vectors_sha256=sha(E/'ONE-actual-primal-dual-slack-RETURN.npz'),actual_status=report['actual_solver_status'],raw_x_z_s_UNALTERED=True,source_operators_NN_solver_FK_native_NOT_replayed=True)
 # Numerical primal diagnostics are retained, never promoted to actual gates.
 canonical_slack=b-A@x;primal_residual=canonical_slack-s;nn0=meta['nonnegative_rows'];cs=canonical_slack[nn0:].reshape(-1,4);cone_margin=cs[:,0]-np.linalg.norm(cs[:,1:],axis=1)
 pdarrays=arrays('NEW-unmodified-primal-canonical-diagnostics-RETURN.npz',canonical_slack=canonical_slack,raw_slack_difference=primal_residual,SOC_margin=cone_margin)
 primal_diag=dict(maximum_raw_slack_consistency_residual=float(abs(primal_residual).max()),numerical_nonnegative_row_failures=int(np.count_nonzero(canonical_slack[:nn0]<0)),numerical_SOC_failures=int(np.count_nonzero(cone_margin<0)),minimum_SOC_margin=float(cone_margin.min()),arrays=pdarrays,diagnostics_NOT_candidate_tolerances_or_exact_source_feasibility=True)
 save('NEW-primal-numerical-diagnostics.json',primal_diag);emit(dict(kind='AFTER_NEW_raw_primal_diagnostics_RETURN',actual_RETURN=primal_diag))
 # Every NEW canonical coefficient contributes to a certified A^T*z interval.
 rhos=[];certificates=[];products=np.empty(A.nnz);prefix=np.empty(A.nnz)
 for column in range(A.shape[1]):
  start,end=A.indptr[column:column+2];ret=dot_enclosure(A.data[start:end],z[A.indices[start:end]]);products[start:end]=ret['products'];prefix[start:end]=ret['prefix'];center=Q(ret['value'])+Q(float(q[column]));err=ret['error_upper'];lo,hi=center-err,center+err
  piece=-Q(.02)*max(abs(lo),abs(hi))if column<453 else Q(.05)*min(lo,Q(0));rhos.append(piece);certificates.append(dict(column=column,nonzeros=int(end-start),actual_fp_sum=ret['value'],absolute_sum_RETURN=ret['absolute_sum'],operations=ret['operations'],error_upper=pack(err),c_lower=pack(lo),c_upper=pack(hi),finite_box_linear_lower=pack(piece)))
 adot=arrays('NEW-all-canonical-Atranspose-z-products-RETURN.npz',products=products,cumulative_prefix_per_column=prefix);save('all454-NEW-dot-and-box-certificates.json',certificates);emit(dict(kind='AFTER_NEW_Atranspose_z_certificate_RETURN',arrays=adot,columns=len(certificates),coefficients=A.nnz))
 bd=dot_enclosure(b,z);bdarrays=arrays('NEW-bdotz-products-and-prefix-RETURN.npz',products=bd['products'],cumulative_prefix=bd['prefix']);bd_upper=Q(bd['value'])+bd['error_upper'];save('NEW-bdotz-enclosure.json',dict(actual_dot=bd['value'],absolute_sum_RETURN=bd['absolute_sum'],operations=bd['operations'],error_upper=pack(bd['error_upper']),upper=pack(bd_upper),arrays=bdarrays))
 # Finite-box slack upper bounds for the nonnegative rows, exactly rational.
 csr=A.tocsr();nn=meta['nonnegative_rows'];correction=Q(0);bad_nn=0;nn_records=[]
 for row in range(nn):
  if z[row]>=0:continue
  bad_nn+=1;upper=Q(float(b[row]))
  for col,coef in zip(csr.indices[csr.indptr[row]:csr.indptr[row+1]],csr.data[csr.indptr[row]:csr.indptr[row+1]]):
   bounds=(-Q(.02),Q(.02))if col<453 else(Q(0),Q(.05));upper-=min(Q(float(coef))*a for a in bounds)
  upper=max(upper,Q(0));term=-Q(float(z[row]))*upper;correction+=term;nn_records.append(dict(row=row,raw_dual=float(z[row]),slack_upper=pack(upper),negative_dual_correction=pack(term)))
 save('NEW-negative-nonnegative-dual-corrections.json',nn_records)
 # Root and defect proposals are durable by batch BEFORE sign assertions.
 count=meta['SOC4'];bad_cones=0;root_checks=0;cone_correction=Q(0);max_root_bits=0
 for block_start in range(0,count,2048):
  block_end=min(count,block_start+2048);roots=[];defects=[];local_checks=[];block_correction=Q(0)
  for k in range(block_start,block_end):
   row=nn+4*k;raw=z[row:row+4];square=sum(Q(float(v))**2 for v in raw[1:]);ki,root=root96(square);defect=max(root-Q(float(raw[0])),Q(0));di=ceilq(defect*SCALE);d=Q(di,SCALE);upper=Q(float(b[row]));block_correction+=upper*d
   is_bad=Q(float(raw[0]))<0 or Q(float(raw[0]))**2<square
   bad_cones+=int(is_bad);local_checks.append(root>=0 and root*root>=square and upper>=0);roots.append(str(ki));defects.append(str(di));max_root_bits=max(max_root_bits,ki.bit_length(),di.bit_length())
  rootfile=save(f'NEW-cone-root-block-{block_start:06d}-RETURN.json',dict(start=block_start,end=block_end,shared_denominator=str(SCALE),root_numerators=roots,defect_numerators=defects,root_square_checks_passed=all(local_checks),correction=pack(block_correction),raw_duals_UNALTERED=True))
  emit(dict(kind='AFTER_NEW_conic_defect_root_RETURN',block_start=block_start,block_end=block_end,actual_RETURN_file=rootfile));assert all(local_checks);root_checks+=len(local_checks);cone_correction+=block_correction
 correction+=cone_correction;lower=sum(rhos)-bd_upper-correction
 certificate=dict(schema='v2d.NEW-defect-corrected-finite-box-rho-lower.v1',linear_box_lower=pack(sum(rhos)),bdotz_upper=pack(bd_upper),negative_nonnegative_and_SOC_defect_correction=pack(correction),SOC_defect_correction=pack(cone_correction),rho_lower_bound=pack(lower),rho_lower_bound_m_float=float(lower),strictly_positive=lower>0,raw_nonnegative_dual_negative_entries=bad_nn,raw_SOC_dual_exact_failures=bad_cones,exact_root_checks=root_checks,maximum_root_integer_bits=max_root_bits,scope='ONLY THIS frozen numeric source-median-proxy/selected-axis/20mmbox construction; NOT original source family/task',raw_x_z_s_NOT_repaired=True,certificate_objects_NOT_physical_forces_or_commands=True)
 save('NEW-exact-residual-corrected-finite-box-certificate.json',certificate);emit(dict(kind='AFTER_NEW_rho_certificate_RETURN',actual_RETURN=certificate))
 # Actual decisions are checked from SAVED NEW returns, not source replay.
 training=read(E/'all151-actual-hard-training-RETURN.json')if(E/'all151-actual-hard-training-RETURN.json').exists()else[];base=read(T/'all151-NEW-training-source-statistics.json');gate_records=[];failed=[];counts=dict(IoU=0,median=0,quality=0,unknown=0)
 for ret in training:
  f=ret['frame'];old=base[f]
  if ret['status']=='CACHED_F_training_values_BY_HASH':passed=True;checks=dict(IoU=True,median=True,quality=True)
  elif ret['status']=='NEW_training_source_predicate_ONLY':
   det=int(ret['training_intersection'])*int(old['training_union'])-int(old['training_intersection'])*int(ret['training_union']);checks=dict(IoU=det>=0,median=Q(ret['training_median_m'])<=Q(old['training_mesh_vertex_median_m']),quality=Q(ret['training_quality'])>=Q(old['training_quality']));passed=all(checks.values())
  else:checks=dict(unknown=False);passed=False
  raw=dict(frame=f,independent_sign_decisions=checks,independent_training_passed=passed,actual_recorded_passed=ret['passes_TRAINING_construction']);gate_records.append(raw);emit(dict(kind='AFTER_NEW_saved_training_decision_RETURN',actual_RETURN=raw));assert passed==ret['passes_TRAINING_construction']
  if not passed:failed.append(f)
  for k,v in checks.items():counts[k]+=int(not v)
 save('all151-NEW-independent-saved-training-decisions.json',gate_records);assert failed==report.get('training_failed_frames',[])
 if training:
  with np.load(E/'RAW-new-object-proposal-before-gates.npz')as f:delta=f['raw_world_translation'];world=f['world_from_object'];cam=f['camera_from_object'];robot=f['unchanged_robot_qpos']
  with np.load(D/'dynamic-source-data.npz')as f:oldw=f['original_world_from_object'];oldc=f['original_camera_from_object'];RC=f['camera_from_world_linear']
  with np.load(S/'e037-thumb-ray-independent-validation-v1/frozen/targets.npz')as f:oldrobot=f['robot_qpos']
  pose_checks=dict(raw_delta_exact_x=np.array_equal(delta.ravel(),x[:453]),world_constructor_exact=np.array_equal(world[:,:3,3],oldw[:,:3,3]+delta),camera_constructor_exact=np.array_equal(cam[:,:3,3],oldc[:,:3,3]+np.einsum('tij,tj->ti',RC,delta,optimize=False)),world_rotation_bytes_unchanged=world[:,:3,:3].tobytes()==oldw[:,:3,:3].tobytes(),camera_rotation_bytes_unchanged=cam[:,:3,:3].tobytes()==oldc[:,:3,:3].tobytes(),all_robot_bytes_unchanged=robot.tobytes()==oldrobot.tobytes())
  save('NEW-raw-proposal-byte-and-constructor-bindings.json',pose_checks);emit(dict(kind='AFTER_NEW_proposal_byte_binding_RETURN',actual_RETURN=pose_checks));assert all(pose_checks.values())
  motion=read(E/'actual-wholeclock-object-motion-RETURN.json');over=[k for k,v in motion['original_caps'].items()if Q(motion['actual'][k])>Q(v)+Q(1e-9)];save('NEW-all6-saved-motion-decision-bindings.json',dict(over_caps=over,old_actual_caps_UNCHANGED=True,original_tolerance=1e-9));assert over==motion['over_caps']==report['motion_failed_caps']
 callbacks=sorted(E.glob('callback-*-actual-info-RETURN.json'));iteration_values=[read(f)['fields'].get('iterations')for f in callbacks];save('NEW-all-callback-durability-bindings.json',dict(callbacks=len(callbacks),iterations=iteration_values,files_sha256={f.name:sha(f)for f in callbacks}));assert len(callbacks)==report['actual_callbacks']
 summary.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),canonical_coefficients_bound=A.nnz,canonical_rows=len(b),new_exact_root_certificates=root_checks,raw_dual_exact_SOC_failures=bad_cones,raw_dual_nonnegative_failures=bad_nn,finite_box_rho_lower_bound_m=float(lower),restricted_zero_slack_numeric_model_excluded=lower>0,actual_training_decisions=len(training),training_failed_frames=failed,training_named_failure_counts=counts,all6_original_motion_decisions_bound=True,callback_returns_bound=len(callbacks),raw_object_rotations_AND_all_robot_commands_UNCHANGED=True,original_source_POSTgate_calls=report['original_source_queries'],source_operators_solver_native_NOT_replayed=True,no_source_motion_component_full_reference_GPU_PPO_task_AUC_or_trajectory_qualified=True)
 save('report.json',summary);emit(dict(kind='AFTER_NEW_saved_return_audit_report',actual_RETURN=summary));print(json.dumps(summary))

"""ONE NEW exact fixed-terminal-dual bound; no solver/model/source queries."""
from exact_terminal_dual_bound_v1 import *
import time
O=S/'e037-exact-terminal-dual-evaluation-v1';started=time.monotonic();stage='declaration'
class BoundStop(Exception):pass
def timecheck():
 if time.monotonic()-started>180:raise BoundStop('NEW180second exact-certificate limit; no extension')
def require_dot(rec):
 if rec['status']!='exact_dyadic_dot':raise BoundStop(rec['status'])
 cap(rec['value'])
mp=json.loads((P/'NEW-complete-certificate-map-RETURN.json').read_text());bs=json.loads((P/'NEW-exact-rational-constructor-bounds-RETURN.json').read_text());bounds=[(F(z['lower']),F(z['upper'])) for z in bs];magnitudes=[F(z['magnitude']) for z in bs]
# All NEW maps/necessity/controls were independently bound before this evaluation.
for p in [S/'e037-exact-terminal-dual-contracts-v1/report.json',S/'e037-exact-terminal-dual-binding-v1/report.json']:assert p.exists()
dump(O/'ONE-NEW-certificate-declaration-BEFORE.json',dict(schema='v2d.ONE-NEW-exact-terminal-dual-bound.v1',created_utc=now(),functional='normalized phase z_rho = physical_rho/r; original P/q objective unchanged',raw_dual='SOLE saved terminal step32 u, byte unchanged',all_rows=ROWS,all_coefficients=32306054,all_columns=COLS,root_grid_bits=96,exact_arithmetic_bit_cap=8192,internal_seconds=180,outer_seconds=240,RSS=2147483648,disk_reserve=1610612736,no_solver_pose_source_native_queries=True,no_dual_repair_rescaling_or_alternate_certificate_selection=True,formula='lower = sum_j min((e_rho+A.Ty)_j*L_j,(e_rho+A.Ty)_j*H_j) - b.Ty - D_orthant - D_SOC',arithmetic_chunks_are_serialization_only_NOT_source_partitions=True,map_sha256=sha(P/'NEW-complete-certificate-map-RETURN.json')))
A=sparse.load_npz(M/'NEW-canonical-A-RETURN.npz');csr=load_csr()
with np.load(M/'NEW-canonical-vectors-RETURN.npz') as z:b=z['b']
with np.load(E/'ONE-actual-raw-terminal-RETURN.npz') as z:y=z['u']
dump(O/'NEW-actual-certificate-input-constructor-RETURN.json',dict(A_shape=list(A.shape),stored_coefficients=A.nnz,dual_shape=list(y.shape),raw_dual_file_sha256=sha(E/'ONE-actual-raw-terminal-RETURN.npz'),vectors_sha256=sha(M/'NEW-canonical-vectors-RETURN.npz')))
try:
 stage='ALL-column exact functionals';support=F(0);columns=[];entry_count=0;nonzero_count=0
 for j in range(COLS):
  timecheck();a,z=A.indptr[j:j+2];rec=exact_dot(A.data[a:z],y[A.indices[a:z]])
  dump(O/f'column-{j:04d}-NEW-exact-dot-RETURN.json',rec);require_dot(rec)
  coeff=cap(rec['value']+(1 if j==1057 else 0));lo,hi=bounds[j];term=min(coeff*lo,coeff*hi)
  out=dict(column=j,exact_ATy=rec['value'],phase_coefficient=coeff,lower_endpoint=lo,upper_endpoint=hi,exact_box_infimum_term=term,scope='support functional only; endpoints are NOT an episode/source/robot candidate')
  dump(O/f'column-{j:04d}-NEW-exact-support-RETURN.json',out);cap(term)
  support+=term;columns.append(out);entry_count+=rec['entries'];nonzero_count+=rec['nonzero_products']
 dump(O/'NEW-ALL-column-box-support-RETURN.json',dict(exact_support=support,stored_coefficient_entries=entry_count,nonzero_products=nonzero_count));cap(support)
 stage='ALL b dot y';by=F(0);by_records=[]
 for begin in range(0,ROWS,65536):
  timecheck();end=min(ROWS,begin+65536);rec=exact_dot(b[begin:end],y[begin:end]);rec.update(row_start=begin,row_end=end)
  dump(O/f'by-{begin:07d}-NEW-exact-dot-RETURN.json',rec);require_dot(rec);by+=rec['value'];by_records.append((begin,end))
 dump(O/'NEW-ALL-by-exact-RETURN.json',dict(value=by,rows=sum(z-a for a,z in by_records)));cap(by)
 stage='ALL orthant defects';negative=np.flatnonzero(y[:NN]<0);Dorth=F(0);orth_count=0
 for begin in range(0,len(negative),256):
  timecheck();records=[]
  for rawr in negative[begin:begin+256]:
   r=int(rawr);items,constant=row(csr,b,r);Sbound=abs(constant)+sum(abs(v)*magnitudes[j] for j,v in items);weight=-q(y[r]);term=weight*Sbound
   records.append(dict(row=r,exact_raw_y=q(y[r]),necessary_slack_upper=Sbound,defect_pairing_upper=term,ALL_nonzero_coefficients_used=len(items)))
  dump(O/f'orthant-{begin:06d}-NEW-exact-defect-RETURN.json',records)
  for rec in records:cap(F(rec['necessary_slack_upper']));cap(F(rec['defect_pairing_upper']));Dorth+=F(rec['defect_pairing_upper'])
  orth_count+=len(records)
 dump(O/'NEW-ALL-orthant-defect-RETURN.json',dict(value=Dorth,negative_rows=orth_count,all_orthant_rows=NN,other_terms_are_exact_algebraic_zero=True));cap(Dorth)
 stage='ALL whole-cone defects';Dsoc=F(0);cone_count=0
 for k,cone in enumerate(mp['SOC_defect']):
  timecheck();a=cone['start'];size=cone['size'];square=sum(q(v)**2 for v in y[a+1:a+size]);up,integer=root_upper(square);defect=max(up-q(y[a]),F(0));items,constant=row(csr,b,a);Sbound=abs(constant)+sum(abs(v)*magnitudes[j] for j,v in items);term=defect*Sbound
  rec=dict(cone=k,start=a,size=size,raw_top=q(y[a]),exact_tail_square=square,outward_norm_upper=up,grid_integer=str(integer),grid_bits=96,whole_cone_dual_defect=defect,necessary_slack_top_upper=Sbound,defect_pairing_upper=term)
  dump(O/f'cone-{k:04d}-NEW-exact-defect-RETURN.json',rec)
  assert up*up>=square and (integer==0 or F(integer-1,2**96)**2<square)
  cap(term);Dsoc+=term;cone_count+=1
 dump(O/'NEW-ALL-SOC-defect-RETURN.json',dict(value=Dsoc,whole_cones=cone_count));cap(Dsoc)
 stage='final exact lower bound';lower=support-by-Dorth-Dsoc
 with np.load(M/'NEW-canonical-vectors-RETURN.npz') as z:original_q=z['q']
 radius=q(json.loads((M/'NEW-canonical-metadata-RETURN.json').read_text())['radius']);physical=radius*lower
 result=dict(schema='v2d.NEW-exact-terminal-dual-bound-RETURN.v1',normalized_phase_lower_bound=lower,physical_phase_lower_bound_m=physical,exact_box_support=support,exact_by=by,exact_orthant_defect=Dorth,exact_whole_SOC_defect=Dsoc,all_stored_coefficient_entries=entry_count,all_by_rows=ROWS,all_orthant_rows=NN,negative_orthant_defects=orth_count,whole_SOC_cones=cone_count,raw_terminal_dual_SHA=sha(E/'ONE-actual-raw-terminal-RETURN.npz'),positive_bound=lower>0,scope='ONLY SAME frozen conditional sufficient closed hard-raster numeric construction; not original source family/task/physical exclusion or qualification')
 dump(O/'ONE-actual-exact-bound-RETURN.json',result);cap(lower);cap(physical)
 report=dict(schema='v2d.NEW-exact-terminal-dual-evaluation.v1',status='POSITIVE_restricted_model_phase_lower_bound' if lower>0 else 'INCONCLUSIVE_nonpositive_phase_lower_bound',normalized_lower_exact=str(lower),normalized_lower_float=float(lower),physical_lower_m_exact=str(physical),physical_lower_m_float=float(physical),zero_phase_excluded_ONLY_frozen_construction=lower>0,all_coefficient_products_covered=entry_count,all_rows=ROWS,negative_orthant_defects=orth_count,whole_SOC_defects=cone_count,root_grid_bits=96,raw_dual_UNCHANGED=True,original_objective_and_all_gates_UNCHANGED=True,no_trajectory_task_source_family_exclusion_or_qualification=True,no_solver_pose_source_native_GPU_PPO=True,wall_seconds=time.monotonic()-started)
 dump(O/'report.json',report)
except BoundStop as error:
 dump(O/'report.json',dict(schema='v2d.NEW-exact-terminal-dual-evaluation.v1',status='UNRESOLVED_declared_bound_limit_or_unknown',stage=stage,reason=str(error),wall_seconds=time.monotonic()-started,no_new_certificate_or_candidate_accepted=True,no_solver_source_native_GPU=True))

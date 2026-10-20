"""NEW exact subsystem functionals from saved dual/row incidence, no solver."""
from pathlib import Path
from fractions import Fraction as Q
import datetime,hashlib,json,os
import numpy as np
from scipy import sparse
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];O=S/'e037-object-certificate-subsystems-v1';M=S/'e037-object-visibility-support-model-v3';E=S/'e037-object-visibility-support-evaluation-v1';D=S/'e037-object-conic-return-audit-v1';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rat(v):
 if isinstance(v,Q):return dict(numerator=v.numerator,denominator=v.denominator)
 if isinstance(v,list):return [rat(x)for x in v]
 if isinstance(v,dict):return {str(k):rat(x)for k,x in v.items()}
 return v
def parse(v):return Q(v['numerator'],v['denominator'])
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(rat(v)),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,rat(v))
 # NEW scope controls: infeasibility of a joint system need not hold separately.
 emit(dict(kind='BEFORE_NEW_subsystem_scope_contract',rows=[[1,Q(1,2)],[-1,Q(-3,4)]],domain=[-1,1],multipliers=[1,1]));full=Q(1,2)-Q(3,4);separate=[Q(1,2)+1,Q(-3,4)+1];emit(dict(kind='AFTER_NEW_subsystem_scope_contract_RETURN',joint_upper=full,separate_uppers=separate,each_single_row_has_a_feasible_point=True));assert full<0 and all(v>0 for v in separate)
 emit(dict(kind='BEFORE_NEW_whole_SOC_block_selection_contract',actual_dual=[1,1,0,0]));whole=[Q(1),Q(1),Q(0),Q(0)];partial=[Q(0),Q(1),Q(0),Q(0)];valid=lambda v:v[0]>=0 and v[0]*v[0]>=sum(x*x for x in v[1:]);emit(dict(kind='AFTER_NEW_whole_SOC_block_selection_contract_RETURN',whole_valid=valid(whole),partial_valid=valid(partial),zero_whole_block_valid=valid([Q(0)]*4)));assert valid(whole)and not valid(partial)and valid([Q(0)]*4)
 emit(dict(kind='BEFORE_NEW_group_affine_functional_additivity_contract',left_offset=Q(1,3),right_offset=Q(-2,5),left_coefficient=[Q(1),Q(-1,2)],right_coefficient=[Q(-1),Q(3,2)]));left=[Q(1),Q(-1,2)];right=[Q(-1),Q(3,2)];point=[Q(1,7),Q(2,9)];a=Q(1,3)-sum(v*x for v,x in zip(left,point));b=Q(-2,5)-sum(v*x for v,x in zip(right,point));combo=Q(1,3)-Q(2,5)-sum((v+w)*x for v,w,x in zip(left,right,point));emit(dict(kind='AFTER_NEW_group_affine_functional_additivity_contract_RETURN',separate_sum=a+b,combined=combo,zero_margin_NOT_exclusion=True));assert a+b==combo
 prior=read(D/'ONE-exact-raw-dual-residual-finite-box-certificate.json');priorreport=read(D/'report.json');assert priorreport['raw_nonnegative_dual_violations']==priorreport['raw_SOC4_dual_violations']==0;actual=read(E/'ONE-actual-SOCP-RETURN.json');z=[Q(float(x))for x in actual['z']];A=sparse.load_npz(M/'canonical-A.npz').tocsr()
 with np.load(M/'canonical-vectors.npz')as v:rhs=v['b']
 origins=read(M/'canonical-row-origins.json');meta=read(M/'canonical-shapes-and-config.json');N=meta['columns'];nr=meta['nonnegative_rows'];assert N==453 and len(origins['nonnegative'])==nr==26576
 spec=dict(schema='v2d.NEW-six-fixed-subsystem-functional-scope.v1',model_sha256=sha(M/'report.json'),row_origins_sha256=sha(M/'canonical-row-origins.json'),actual_dual_RETURN_sha256=sha(E/'ONE-actual-SOCP-RETURN.json'),prior_exact_cone_dual_validity_reused_BY_HASH=sha(D/'report.json'),scopes=[['B','G'],['B','V'],['B','M'],['B','G','V'],['B','G','M'],['B','V','M']],group_meaning=dict(B='declared20mmalgorithmicbox rows',G='NEWselected-axis geometric support construction',V='local training source-face cells and strict center/vertex depth construction',M='WHOLE299original-object-motionSOC4blocks with sufficientp95envelope'),no_actual_dual_values_changed=True,separate_functionals_NOT_repaired_raw_duals=True,no_solver_pose_or_family_exclusion_proposal=True);save('ONE-declared-subsystem-scope.json',spec);emit(dict(kind='BEFORE_ONE_NEW_six_subsystem_functional_construction',proposal=spec))
 groups={k:dict(rows=[],offset=Q(0),coefficients=[Q(0)]*N,scalar_coefficients=0)for k in ['B','G','V','M']};rowgroups=[]
 for i in range(A.shape[0]):
  if i>=nr:g='M'
  else:
   kind=origins['nonnegative'][i]['kind']
   if kind.startswith('algorithm_world_translation_box_'):g='B'
   elif kind=='NEW_sufficient_fixed_axis_support_plane':g='G'
   elif kind.startswith('ONE_training_face_cell_')or kind.startswith('strict_depth_'):g='V'
   else:raise ValueError('unknown row origin '+kind)
  rowgroups.append(g);d=groups[g];d['rows'].append(i);d['offset']+=z[i]*Q(float(rhs[i]));row=A.getrow(i)
  for j,v in zip(row.indices,row.data,strict=True):d['coefficients'][int(j)]+=z[i]*Q(float(v));d['scalar_coefficients']+=1
  emit(dict(kind='AFTER_NEW_exact_row_to_subsystem_binding',row=i,group=g,actual_raw_dual=z[i],origin=origins['nonnegative'][i]if i<nr else dict(SOCblock=(i-nr)//4,coordinate=(i-nr)%4),whole_SOC_selection_mandatory=True))
 assert sum(g['offset']for g in groups.values())==parse(prior['offset_z_dot_b']);assert all(sum(g['coefficients'][j]for g in groups.values())==parse(prior['all453_stationarity_residual_coefficients'][j])for j in range(N));save('NEW-exact-group-contribution-RETURNS.json',groups);save('all27772-row-group-bindings.json',rowgroups)
 # Six predeclared one-shot functionals. Positive upper bounds are inconclusive.
 outcomes=[];h=Q(1,50)
 for labels in spec['scopes']:
  name=''.join(labels);emit(dict(kind='BEFORE_NEW_subsystem_exact_finite_box_certificate',scope=name,groups=labels,NOT_repaired_actual_solver_dual=True));rows=[i for k in labels for i in groups[k]['rows']];offset=sum(groups[k]['offset']for k in labels);c=[sum(groups[k]['coefficients'][j]for k in labels)for j in range(N)];upper=offset+h*sum(abs(v)for v in c);r=dict(scope=name,groups=labels,selected_whole_rows=rows,offset=offset,all453_coefficients=c,exact_finite_box_upper=upper,strict_subsystem_exclusion=upper<0,inconclusive_if_nonnegative=True,cone_validity_reused_BY_HASH=True,scope_text='ONLY selected SAVED numeric construction rows under declared20mm algorithmic box. NOT original object/source/native/task infeasibility, measured contacts, physical forces or causal importance.');save(f'{name}-exact-subsystem-certificate.json',r);emit(dict(kind='AFTER_NEW_subsystem_exact_finite_box_certificate_RETURN',actual_RETURN=r));outcomes.append(dict(scope=name,selected_rows=len(rows),upper_approx=float(upper),strict_subsystem_exclusion=upper<0))
 summary=dict(schema='v2d.NEW-object-certificate-subsystems.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEWanalytic_scope_contracts=3,rows_bound=len(rowgroups),group_rows={k:len(v['rows'])for k,v in groups.items()},group_scalar_coefficients={k:v['scalar_coefficients']for k,v in groups.items()},exact453combined_group_identities_pass=True,scope_outcomes=outcomes,no_dual_magnitude_cause_or_force_inference=True,all_actual_model_and_solver_vectors_UNCHANGED=True,no_old_solver_audit_or_producer_repeated=True,no_native_source_point_or_trajectory_proposed=True,no_global_source_native_family_or_task_exclusion=True);save('report.json',summary);emit(dict(kind='AFTER_ONE_NEW_subsystem_scope_analysis',actual_RETURN=summary));print(json.dumps(summary))

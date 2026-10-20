"""NEW raw-return finite-box conic certificate and exact opposite-row diagnosis."""
from pathlib import Path
from fractions import Fraction as Q
import datetime,hashlib,itertools,json,math,os
import numpy as np
from scipy import sparse
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];O=S/'e037-object-conic-return-audit-v1';M=S/'e037-object-visibility-support-model-v3';E=S/'e037-object-visibility-support-evaluation-v1';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rat(x):
 if isinstance(x,Q):return dict(numerator=x.numerator,denominator=x.denominator)
 if isinstance(x,list):return [rat(v)for v in x]
 if isinstance(x,dict):return {str(k):rat(v)for k,v in x.items()}
 return x
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(rat(v)),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
def sqrt_upper(x):
 x=Q(x);assert x>=0;scale=2**96;n=x.numerator*scale*scale;d=x.denominator;r=math.isqrt(n//d)
 if r*r*d<n:r+=1
 return Q(r,scale)
with (O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,rat(v))
 # New analytical controls. Cube corners here are algebraic proof evaluations.
 h=Q(1,50);coef=[Q(-1,3),Q(1,7),Q(1,11)];offset=Q(-2);sup=offset+h*sum(abs(x)for x in coef);corners=[offset-sum(sign*h*a for sign,a in zip(v,coef,strict=True))for v in itertools.product([-1,1],repeat=3)];emit(dict(kind='AFTER_NEW_finite_box_support_contract_RETURN',exact_upper=sup,all8corner_returns=corners));assert max(corners)==sup
 lam=Q(-1,100);upper=Q(3);svals=[Q(0),Q(1),Q(3)];lower=lam*upper;emit(dict(kind='AFTER_NEW_signed_nonnegative_dual_residual_contract_RETURN',lower=lower,actual_returns=[lam*s for s in svals]));assert all(lam*s>=lower for s in svals)
 norm=sqrt_upper(Q(14));assert norm*norm>=14 and (norm-Q(1,2**96))**2<14;t=Q(3);T=Q(2);soclower=min(Q(0),t-norm)*T;emit(dict(kind='AFTER_NEW_SOC4_dual_residual_contract_RETURN',norm_upper=norm,unit_box_time_upper=T,raw_dual_time=t,lower=soclower,mathematical_identity='t*a+v.w >= (t-||v||)*a >= min(0,t-uppernorm(v))*T for 0<=||w||<=a<=T',raw_dual_NOT_projected_or_repaired=True));assert soclower<0
 a=Q(1,3);b1=Q(-2);b2=Q(1);emit(dict(kind='AFTER_NEW_exact_opposite_halfspace_contract_RETURN',a=a,b1=b1,b2=b2,strict_sum=b1+b2,zero_sum_does_NOT_exclude=True));assert b1+b2<0 and not Q(0)<0
 result=read(E/'ONE-actual-SOCP-RETURN.json');raw=read(E/'report.json');meta=read(M/'canonical-shapes-and-config.json');A=sparse.load_npz(M/'canonical-A.npz').tocsr()
 with np.load(M/'canonical-vectors.npz')as v:b=v['b']
 z=[Q(float(x))for x in result['z']];assert len(z)==A.shape[0] and len(z)==27772;origins=read(M/'canonical-row-origins.json');nr=meta['nonnegative_rows'];N=meta['columns'];emit(dict(kind='BEFORE_NEW_raw_dual_exact_finite_box_audit',actual_solver_status=result['status'],actual_RETURN_sha256=sha(E/'ONE-actual-SOCP-RETURN.json'),raw_model_sha256=sha(M/'report.json'),declared_algorithm_box_halfwidth=h,all_raw_x_z_s_UNALTERED=True))
 c=[Q(0)]*N;offset=Q(0);lower=Q(0);badnn=[];badSOC=[];coeffs=0
 def rowterms(i):
  row=A.getrow(i);return [(int(j),Q(float(a)))for j,a in zip(row.indices,row.data,strict=True)]
 for i,zi in enumerate(z):
  bi=Q(float(b[i]));offset+=zi*bi
  for j,ai in rowterms(i):c[j]+=zi*ai;coeffs+=1
 for i in range(nr):
  if z[i]>=0:continue
  maximum_slack=Q(float(b[i]))+h*sum(abs(a)for _,a in rowterms(i));term=z[i]*max(Q(0),maximum_slack);lower+=term;record=dict(row=i,actual_raw_dual=z[i],finite_box_slack_upper=maximum_slack,lower_contribution=term);badnn.append(record);emit(dict(kind='AFTER_NEW_nonnegative_dual_residual_correction_RETURN',actual_RETURN=record))
 for k in range(meta['SOC4']):
  i=nr+4*k;t=z[i];r2=sum(x*x for x in z[i+1:i+4]);r=sqrt_upper(r2);defect=min(Q(0),t-r);maximum_time=Q(float(b[i]))+h*sum(abs(a)for _,a in rowterms(i));term=defect*max(Q(0),maximum_time);lower+=term;record=dict(block=k,canonical_start=i,actual_raw_dual_time=t,actual_raw_dual_spatial=z[i+1:i+4],norm_squared=r2,rational96bit_norm_upper=r,finite_box_slack_time_upper=maximum_time,lower_contribution=term,raw_dual_cone_pass=t>=0 and t*t>=r2)
  emit(dict(kind='AFTER_NEW_SOC4_residual_certificate_RETURN',actual_RETURN=record))
  if not record['raw_dual_cone_pass']:badSOC.append(record)
 upper=offset+h*sum(abs(x)for x in c);gap=lower-upper;certificate=dict(schema='v2d.NEW-raw-conic-finitebox-infeasibility-certificate.v1',raw_solver_status=result['status'],offset_z_dot_b=offset,all453_stationarity_residual_coefficients=c,finite_box_expression_upper=upper,residual_corrected_cone_expression_lower=lower,strict_contradiction_margin=gap,strict_infeasibility_of_THIS_saved_canonical_and_declared20mm_box=gap>0,exact_coefficients=coeffs,raw_nonnegative_dual_violations=len(badnn),raw_SOC_dual_violations=len(badSOC),all_raw_values_UNCHANGED=True,scope='THIS one fixed-axis/local-training-visibility/SAVED-binary-canonical construction under declared20mm box only. NOT all object translations, original source-time family, free rotations, task or required wrist change. Source-cell maps remain locally approximate; source truth not proved.');save('ONE-exact-raw-dual-residual-finite-box-certificate.json',certificate);emit(dict(kind='AFTER_NEW_exact_finite_box_certificate_RETURN',actual_RETURN=certificate))
 # One exact map lookup of opposite NEW support rows, not any pose/anchor sweep.
 bank={};opposites=[];seen=set()
 for i,o in enumerate(origins['nonnegative']):
  if o['kind']!='NEW_sufficient_fixed_axis_support_plane':continue
  f=o['frame'];terms=dict(rowterms(i));key=tuple(terms.get(3*f+j,Q(0))for j in range(3));op=tuple(-v for v in key)
  if (f,op)in bank:
   j=bank[(f,op)];pair=tuple(sorted((i,j)))
   if pair not in seen:
    seen.add(pair);sum_upper=Q(float(b[i]))+Q(float(b[j]));r=dict(frame=f,rows=list(pair),exact_opposite_coefficients=list(key),upper_sum=sum_upper,strict_TWO_ROW_contradiction=sum_upper<0,scope='THIS pair of selected construction halfspaces ONLY; different support choices/free original native problem NOT excluded');opposites.append(r);emit(dict(kind='AFTER_NEW_exact_opposite_support_row_certificate_RETURN',actual_RETURN=r))
  if (f,key)not in bank or b[i]<b[bank[(f,key)]]:bank[(f,key)]=i
 save('all-NEW-exact-opposite-support-row-certificates.json',opposites);negative=[r for r in opposites if r['strict_TWO_ROW_contradiction']];save('strict-opposite-row-contradictions.json',negative)
 summary=dict(schema='v2d.NEW-object-conic-raw-return-independent-audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEW_analytic_finite_box_SOC_sign_contracts=4,actual_episode_status=result['status'],actual_episode_iterations=result['iterations'],actual_episode_seconds=result['solve_time'],all_raw_primal_dual_slack_UNALTERED=True,exact_raw_dual_model_coefficients=coeffs,raw_nonnegative_dual_violations=len(badnn),raw_SOC4_dual_violations=len(badSOC),residual_corrected_finite_box_strict_margin_approx=float(gap),strict_constructed_canonical_box_exclusion=gap>0,opposite_support_row_pairs=len(opposites),strict_two_row_support_conflicts=len(negative),support_conflict_frames=sorted({r['frame']for r in negative}),source_native_original_family_or_task_exclusion=False,no_required_wrist_axis_or_physical_force_claim=True,no_new_solver_or_source_native_measurement_or_proposal=True,no_trajectory_reference_task_or_score_qualified=True)
 save('report.json',summary);emit(dict(kind='AFTER_NEW_raw_return_independent_audit',actual_RETURN=summary));print(json.dumps(summary))

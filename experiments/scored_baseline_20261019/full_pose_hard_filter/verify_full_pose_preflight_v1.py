"""Independent complete NEW canonical/selected-support binding; no producer replay."""
from pathlib import Path
import datetime,json,hashlib,os,math
import numpy as np
from scipy import sparse
from full_pose_inputs_v1 import load
S=Path(__file__).resolve().parents[1];C=S/'compute';I=S/'e037-full-pose-preflight-v1';O=S/'e037-full-pose-preflight-binding-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
 with (O/n).open('w')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def arrays(n,**d):
 with (O/n).open('wb')as f:np.savez_compressed(f,**d);f.flush();os.fsync(f.fileno())
d=load();m=json.loads((I/'NEW-canonical-metadata-RETURN.json').read_text());save('NEW-independent-binding-BEFORE.json',dict(inputs={str(p):sha(p)for p in I.iterdir()if p.is_file()and p.name!='worker.log'},all_original_inputs=d['provenance'],no_model_source_geometry_runtime_solver_replay=True))
with np.load(I/'NEW-all-pair-rotating-support-linearization-RETURN.npz')as z:g={k:z[k]for k in z.files}
finite={k:bool(np.isfinite(v).all())for k,v in g.items()};save('NEW-all-returned-support-finiteness-RETURN.json',dict(all_returned_arrays_finite=finite,all_axis_values=int(len(g['all_axis_gaps'])),warnings_cause_UNPROVEN=True));assert all(finite.values())
# Bind only NEW selected support branches/rotational coefficients. Full axis
# maxima/ties bind SAVED returned arrays; no old native/axis producer replay.
gerrors=[];jerrors=[];maxerrors=[];tieerrors=[];normal_errors=[]
for k,(f,a,b)in enumerate(d['pairs']):
 f,a,b=map(int,(f,a,b));TF=d['body'][f,a];TO=d['body'][f,b];H=d['vertices'][a]@TF[:3,:3].T+TF[:3,3];V=d['vertices'][b];obj=V@TO[:3,:3].T+TO[:3,3];n=g['normals'][k];iv=int(g['object_support_vertex'][k]);ih=int(g['fixed_support_vertex'][k]);family=int(g['families'][k]);normal_errors.append(abs(float(np.linalg.norm(n))-1.));J=np.empty(6)
 if family==0:
  pO=np.array([math.fsum(float(x)*float(y)for x,y in zip(v,n))for v in obj]);pH=np.array([math.fsum(float(x)*float(y)for x,y in zip(v,n))for v in H]);expected=float(pO.min()-pH.max());J[:3]=np.array([math.fsum(float(n[z])*float(d['stage'][z,j])for z in range(3))for j in range(3)]);J[3:]=np.cross(V[iv],TO[:3,:3].T@n)
 else:
  pO=np.array([math.fsum(float(x)*float(y)for x,y in zip(v,n))for v in obj]);pH=np.array([math.fsum(float(x)*float(y)for x,y in zip(v,n))for v in H]);expected=float(pH.min()-pO.max());J[:3]=-np.array([math.fsum(float(n[z])*float(d['stage'][z,j])for z in range(3))for j in range(3)]);objnorm=np.vstack([d['normals'][b],-d['normals'][b]]);idx=int(g['branches'][k])-2*len(d['normals'][a]);n0=objnorm[idx];J[3:]=np.cross(n0,TO[:3,:3].T@(H[ih]-TO[:3,3]))/np.linalg.norm(TO[:3,:3]@n0)
 gerrors.append(g['gaps'][k]-expected);jerrors.append(g['J'][k]-J);vals=g['all_axis_gaps'][g['axis_ptr'][k]:g['axis_ptr'][k+1]];maxerrors.append(g['gaps'][k]-vals.max());tieerrors.append(int(g['branch_tie_counts'][k])-int(np.count_nonzero(vals==vals.max())))
ge=float(np.max(abs(gerrors)));je=float(np.max(abs(jerrors)));me=float(np.max(abs(maxerrors)));arrays('NEW-all-selected-support-rotation-binding-residuals-RETURN.npz',gap_residual=np.array(gerrors),J_residual=np.array(jerrors),saved_max_residual=np.array(maxerrors),tie_count_difference=np.array(tieerrors),normal_unit_error=np.array(normal_errors));save('NEW-selected-support-independent-binding-RETURN.json',dict(pair_rows=len(gerrors),all_axis_values=len(g['all_axis_gaps']),max_gap_error=ge,max_rotation_translation_J_error=je,max_saved_axis_max_error=me,tie_count_errors=int(np.count_nonzero(tieerrors)),local_diagnostic_bound=2e-12,NOT_native_gate_or_candidate_tolerance=True));assert ge<2e-12 and je<2e-12 and me==0 and not any(tieerrors)
# Build independent row origins/coefficients from the NEW saved linearization
# and immutable original motion maps. Do not call the canonical constructor.
r=d['radius'];N=151;n=907;rows=[];cols=[];vals=[];rhs=[];types=[]
def emit(entries,b,kind):
 i=len(rhs);rhs.append(float(b));types.append(kind)
 for j,v in entries:
  if v:rows.append(i);cols.append(j);vals.append(float(v))
emit([(906,-1.)],0.,'rho_lower');emit([(906,1.)],1.,'rho_upper')
for k,(f,_,_)in enumerate(d['pairs']):
 f=int(f);J=g['J'][k];emit([(6*f+j,-float(J[j])*(r if j<3 else 1)/r)for j in range(6)]+[(906,-1.)],(float(g['gaps'][k])+.001)/r,'rotating_support_search')
for f in range(N):
 emit([(6*f+j,float(d['translation_median_gradient'][f,j]))for j in range(3)]+[(6*f+3+j,float(d['rotation_median_gradient'][f,j])/r)for j in range(3)],0.,'median_SEARCH');v=int(d['min_depth_vertex'][f]);emit([(6*f+j,-float(d['Rcw'][f,2,j]))for j in range(3)]+[(6*f+3+j,-float(d['source_rotation_J'][f,v,2,j])/r)for j in range(3)],(float(d['min_C0_z'][f])-.01)/r,'depth_SEARCH')
nonnegative=len(rhs);cone_starts=[]
for angular in [False,True]:
 A=d['angular_M'if angular else'linear_M'].tocsr();base=d['angular_base'if angular else'linear_base']
 for k in range(299):
  cap=d['motion_caps'][(1 if k<150 else 3)if angular else(0 if k<150 else 2)];kind='angular_motion_SEARCH'if angular else'linear_motion';cone_starts.append(len(rhs));emit([],1.,kind)
  for a in range(3):
   z=A.getrow(3*k+a);items=[]
   for col,value in zip(z.indices,z.data):
    f,j=divmod(int(col),3);items.append((6*f+j+(3 if angular else 0),-float(value)*(1 if angular else r)/cap))
   emit(items,float(base[3*k+a])/cap,kind)
for f in range(N):
 for angular,bound in [(False,1/8),(True,float(np.pi/12))]:
  kind='rotation_norm_CONSTRUCTION'if angular else'translation_norm_CONSTRUCTION';cone_starts.append(len(rhs));emit([],bound,kind)
  for j in range(3):emit([(6*f+j+(3 if angular else 0),-1.)],0.,kind)
E=sparse.coo_matrix((vals,(rows,cols)),shape=(len(rhs),n)).tocsc();actual=sparse.load_npz(I/'NEW-canonical-A-RETURN.npz');diff=actual-E;error=float(abs(diff.data).max(initial=0));sparse.save_npz(O/'NEW-all-canonical-coefficient-residuals-RETURN.npz',diff)
with np.load(I/'NEW-canonical-vectors-RETURN.npz')as z:be=z['b']-np.array(rhs);q=z['q']
P=sparse.load_npz(I/'NEW-canonical-P-RETURN.npz');expected_diag=np.r_[np.full(906,1e-3),0.];qe=q-np.r_[np.zeros(906),1.];pe=P-sparse.diags(expected_diag);arrays('NEW-all-canonical-RHS-objective-residuals-RETURN.npz',b_error=be,q_error=qe,P_diagonal_error=P.diagonal()-expected_diag)
berror=float(np.max(abs(be)));passed=error==0 and berror==0 and not np.any(qe)and pe.nnz==0 and nonnegative==m['nonnegative']and len(cone_starts)==900 and types==m['row_kinds']and cone_starts==[c['start']for c in m['cones']]
save('NEW-complete-canonical-binding-RETURN.json',dict(rows=actual.shape[0],columns=actual.shape[1],coefficients=actual.nnz,nonnegative=nonnegative,SOC4=len(cone_starts),coefficient_error=error,RHS_error=berror,all_row_kinds_and_whole_cone_scales_bound=True,all_actual_entries_finite=bool(np.isfinite(actual.data).all()and np.isfinite(rhs).all()),objective_matches=not np.any(qe)and pe.nnz==0,passed=passed));assert passed and np.isfinite(actual.data).all()and np.isfinite(rhs).all()
log=(I/'worker.log').read_text();warnings=[line for line in log.splitlines()if 'RuntimeWarning:'in line];save('NEW-retained-warning-scope.json',dict(actual_warning_lines=warnings,cause_UNPROVEN=True,independent_actual_return_finiteness_and_coefficients_checked=True,not_reconstructed_old_warning_or_claim_of_future_reliability=True))
save('report.json',dict(schema='v2d.NEW-full-pose-preflight-independent-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all66591_selected_support_and_rotation_rows_bound=True,all907_columns_and70495_rows_bound=True,canonical_coefficients=actual.nnz,SOC4=900,errors=dict(geometry=ge,rotation_J=je,canonical=error,RHS=berror),preflight_RuntimeWarnings_retained=len(warnings),warning_cause_UNPROVEN=True,all_returned_arrays_finite=True,no_solver_or_episode_pose_source_native_queries=True));print(json.dumps(dict(success=True,coefficients=actual.nnz,geometry_error=ge,J_error=je,warnings_retained=len(warnings))))

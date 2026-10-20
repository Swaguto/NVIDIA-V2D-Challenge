"""Independent complete NEW canonical/selected-support binding; no producer replay."""
from pathlib import Path
import datetime,json,hashlib,os,math
import numpy as np
from scipy import sparse
from full_pose_inputs_v1 import load
S=Path(__file__).resolve().parents[1];C=S/'compute';I=S/'e037-full-pose-preflight-v1';O=S/'e037-full-pose-preflight-binding-v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
 with (O/n).open('w')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def arrays(n,**d):
 with (O/n).open('wb')as f:np.savez_compressed(f,**d);f.flush();os.fsync(f.fileno())
d=load();m=json.loads((I/'NEW-canonical-metadata-RETURN.json').read_text());save('NEW-independent-binding-BEFORE.json',dict(inputs={str(p):sha(p)for p in I.iterdir()if p.is_file()and p.name!='worker.log'},all_original_inputs=d['provenance'],no_model_source_geometry_runtime_solver_replay=True))
with np.load(I/'NEW-all-pair-rotating-support-linearization-RETURN.npz')as z:g={k:z[k]for k in z.files}
parent=S/'e037-full-pose-preflight-binding-v1'
finite_receipt=parent/'NEW-all-returned-support-finiteness-RETURN.json'
save('NEW-failed-reader-prefix-provenance.json',dict(completed_finiteness_receipt=str(finite_receipt),sha256=sha(finite_receipt),old_unsaved_loop_residuals_UNAVAILABLE_not_reconstructed=True,NEW_distinct_world_infinitesimal_coefficient_proof=True))
# Distinct proof: world infinitesimal polynomial coefficients, grouped over all
# clocks, with every group RETURN durable before a check or subsequent group.
maxgap=0.;maxJ=0.;maxmax=0.;tiebad=0;groups=0
for a in d['fixed_geoms']:
 for b in d['object_geoms']:
  ids=np.flatnonzero((d['pairs'][:,1]==a)&(d['pairs'][:,2]==b));frames=d['pairs'][ids,0].astype(int);TF=d['body'][frames,a];TO=d['body'][frames,b];R=TO[:,:3,:3];t=TO[:,:3,3];H=np.einsum('fij,vj->fvi',TF[:,:3,:3],d['vertices'][a],optimize=False)+TF[:,:3,3][:,None,:];obj=np.einsum('fij,vj->fvi',R,d['vertices'][b],optimize=False)+t[:,None,:];n=g['normals'][ids];iv=g['object_support_vertex'][ids];ih=g['fixed_support_vertex'][ids];osel=obj[np.arange(len(ids)),iv];hsel=H[np.arange(len(ids)),ih];pO=np.einsum('fvi,fi->fv',obj,n,optimize=False);pH=np.einsum('fvi,fi->fv',H,n,optimize=False);family=g['families'][ids];expected_gap=np.where(family==0,pO.min(axis=1)-pH.max(axis=1),pH.min(axis=1)-pO.max(axis=1));E=np.zeros((len(ids),6));E[:,:3]=np.einsum('fi,ij->fj',n,d['stage'],optimize=False)*np.where(family[:,None]==0,1.,-1.)
  for j in range(3):
   axis=R[:,:,j];dO=np.cross(axis,osel-t);dn=np.cross(axis,n);fixedder=np.einsum('fi,fi->f',n,dO,optimize=False);movingder=np.einsum('fi,fi->f',hsel-osel,dn,optimize=False)-fixedder;E[:,3+j]=np.where(family==0,fixedder,movingder)
  gres=g['gaps'][ids]-expected_gap;jres=g['J'][ids]-E;maxres=[];tr=[]
  for k in ids:
   values=g['all_axis_gaps'][g['axis_ptr'][k]:g['axis_ptr'][k+1]];maxres.append(float(g['gaps'][k]-values.max()));tr.append(int(g['branch_tie_counts'][k])-int(np.count_nonzero(values==values.max())))
  arrays(f'NEW-world-coefficient-group-{a:02d}-{b:02d}-RETURN.npz',pair_indices=ids,gap_residual=gres,J_residual=jres,saved_axis_max_residual=np.array(maxres),saved_tie_count_difference=np.array(tr))
  e1=float(np.max(abs(gres)));e2=float(np.max(abs(jres)));e3=max(map(abs,maxres));bad=int(np.count_nonzero(tr));save(f'NEW-world-coefficient-group-{a:02d}-{b:02d}-decision.json',dict(pair_rows=len(ids),gap_error=e1,coefficient_error=e2,axismax_error=e3,tie_errors=bad,local_diagnostic_bound=2e-12,NOT_candidate_tolerance=True));assert e1<2e-12 and e2<2e-12 and e3==0 and bad==0
  maxgap=max(maxgap,e1);maxJ=max(maxJ,e2);maxmax=max(maxmax,e3);tiebad+=bad;groups+=1
ge=maxgap;je=maxJ;me=maxmax
save('NEW-world-infinitesimal-support-binding-RETURN.json',dict(NEW_groups=groups,all66591_pairs_bound=True,maximum_gap_error=ge,maximum_coefficient_error=je,max_returned_axismax_error=me,tie_count_errors=tiebad,old_V1_unsaved_values_UNAVAILABLE_not_reconstructed=True))
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
save('report.json',dict(schema='v2d.NEW-full-pose-preflight-independent-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all66591_selected_support_and_rotation_rows_bound=True,all907_columns_and70495_rows_bound=True,canonical_coefficients=actual.nnz,SOC4=900,errors=dict(geometry=ge,rotation_J=je,canonical=error,RHS=berror),preflight_RuntimeWarnings_retained=len(warnings),warning_cause_UNPROVEN=True,all_returned_arrays_finite=True,no_solver_or_episode_pose_source_native_queries=True,V1_unsaved_support_residuals_UNAVAILABLE_not_reconstructed=True,NEW_world_infinitesimal_grouped_proof=True));print(json.dumps(dict(success=True,coefficients=actual.nnz,geometry_error=ge,J_error=je,warnings_retained=len(warnings))))

"""Independent NEW SAVED split-return identities; no method/solve/operator replay.

Do not run A@x, A.T@v, cone projector or normal solver. Check actual archived
row-state equations, linear residual RETURNs, exact dyadic cone signs, recorded
projection branches and terminal identity. Numerical cone defects stay defects.
"""
from whole_row_conic_split_v1 import *
from fractions import Fraction as F
O=S/'e037-whole-row-conic-return-audit-v1';E=S/'e037-whole-row-conic-evaluation-v1'
meta=json.loads((I/'NEW-structured-layout-RETURN.json').read_text());cones=meta['cones'];nn=4613807
with np.load(M/'NEW-canonical-vectors-RETURN.npz') as z:b=z['b'];q=z['q']
def read_array(path,key):
 with np.load(path) as z:return z[key]
def exact_soc_bad(v):
 bad=0;minimum=None
 for a,k,_ in cones:
  top=F.from_float(float(v[a]));det=top*top-sum(F.from_float(float(t))**2 for t in v[a+1:a+k])
  if top<0 or det<0:bad+=1
  if minimum is None or det<minimum:minimum=det
 return bad,str(minimum)
old_s=read_array(E/'initial-NEW-projection-RETURN.npz','s');old_u=read_array(E/'initial-NEW-dual-RETURN.npz','u');records=[];maxrat=0.
steps=sorted(E.glob('step-*-NEW-step-summary-RETURN.json'))
for summary_path in steps:
 k=int(summary_path.name.split('-')[1]);p=E/f'step-{k:03d}'
 term=read_array(str(p)+'-NEW-adjoint-input-RETURN.npz','term');at=read_array(str(p)+'-NEW-adjoint-RETURN.npz','ATterm');rhs=read_array(str(p)+'-NEW-normal-rhs-BEFORE.npz','rhs');x=read_array(str(p)+'-NEW-actual-linear-solve-RETURN.npz','x')
 Ax=read_array(str(p)+'-NEW-ALL-row-Ax-RETURN.npz','Ax');w=read_array(str(p)+'-NEW-ALL-cone-input-RETURN.npz','w');s=read_array(str(p)+'-NEW-ALL-cone-projection-RETURN.npz','s');u=read_array(str(p)+'-NEW-ALL-row-dual-RETURN.npz','u')
 # Different association from producer updates; compare real algebraic identities
 # with conditional, fixed per-operand RN envelopes. No value is corrected.
 residuals=[term+old_s+old_u-b,w+Ax+old_u-b,u-old_u-Ax-s+b]
 scales=[abs(term)+abs(old_s)+abs(old_u)+abs(b),abs(w)+abs(Ax)+abs(old_u)+abs(b),abs(u)+abs(old_u)+abs(Ax)+abs(s)+abs(b)]
 maxima=[]
 for residual,scale in zip(residuals,scales):
  bound=32*np.finfo(float).eps*np.maximum(scale,np.finfo(float).tiny);ratio=float(np.max(abs(residual)/bound));maxima.append(ratio)
 rhs_error=float(np.max(abs(at-rhs-q)));rhs_bound=float(8*np.finfo(float).eps*max(1.,np.max(abs(at)+abs(rhs)+abs(q))))
 pm=json.loads(Path(str(p)+'-NEW-ALL-cone-meta-RETURN.json').read_text());normerror=F(0);branch_bad=0
 assert pm['status']=='projected_ALL_product_coordinates' and len(pm['branches'])==1051
 for branch,(a,size,kind) in zip(pm['branches'],cones):
  assert branch['start']==a and branch['size']==size
  top=F.from_float(float(w[a]));squares=sum(F.from_float(float(t))**2 for t in w[a+1:a+size]);returned_norm=F.from_float(float(branch['norm']));diff=abs(returned_norm**2-squares)
  if squares: normerror=max(normerror,diff/squares)
  if branch['region']=='inside_or_boundary':branch_bad+=not(top>=0 and top*top>=squares)
  elif branch['region']=='polar_or_boundary':branch_bad+=not(top<=0 and top*top>=squares)
  else:branch_bad+=not(top*top<squares)
 s_bad,s_det=exact_soc_bad(s);u_bad,u_det=exact_soc_bad(u)
 lin=json.loads(Path(str(p)+'-NEW-linear-decision-RETURN.json').read_text());rawlin=read_array(str(p)+'-NEW-linear-residual-RETURN.npz','residual');linscale=read_array(str(p)+'-NEW-linear-residual-RETURN.npz','scale');linratio=float(np.max(abs(rawlin)/np.maximum(1.,linscale)))
 summary=json.loads(summary_path.read_text());primal=float(np.max(abs(u-old_u)))
 record=dict(step=k,ALL_row_state_equation_entries=3*len(b),state_identity_max_bound_ratios=maxima,rhs_error=rhs_error,rhs_bound=rhs_bound,ALL1051_saved_norm_exact_squared_relative_error=str(normerror),exact_branch_condition_failures=branch_bad,raw_exact_SOC4_SOC5_slack_violations=s_bad,raw_exact_SOC4_SOC5_dual_violations=u_bad,raw_minimum_slack_SOC_square=s_det,raw_minimum_dual_SOC_square=u_det,raw_nonnegative_slack_failures=int(np.count_nonzero(s[:nn]<0)),raw_nonnegative_dual_failures=int(np.count_nonzero(u[:nn]<0)),linear_return_guard_matches=linratio==lin['relative_error'],saved_primal_residual_matches=primal==summary['max_primal_residual'],raw_x_finite=bool(np.isfinite(x).all()),projection_values_UNALTERED=True)
 dump(O/f'step-{k:03d}-NEW-independent-state-RETURN.json',record)
 assert all(z<=1 for z in maxima) and rhs_error<=rhs_bound and record['linear_return_guard_matches'] and record['saved_primal_residual_matches'] and record['raw_x_finite']
 assert normerror<=F(1,2**45) # conditional tiny-vector norm diagnostic ONLY
 records.append(record);maxrat=max(maxrat,max(maxima));old_s=s;old_u=u
method=json.loads((E/'report.json').read_text());terminal=None
if (E/'ONE-actual-raw-terminal-RETURN.npz').exists():
 with np.load(E/'ONE-actual-raw-terminal-RETURN.npz') as z:tx=z['x'];ts=z['s'];tu=z['u']
 assert len(records)==32 and np.array_equal(tx,x) and np.array_equal(ts,s) and np.array_equal(tu,u)
 terminal=dict(raw_final_step32_identity=True,not_best_or_intermediate_selection=True)
else:assert method.get('no_terminal_selected')
dump(O/'report.json',dict(schema='v2d.NEW-whole-row-conic-SAVED-return-audit.v1',complete_steps=len(records),ALL_saved_row_state_equations=len(records)*3*len(b),ALL1051norms_and_branch_conditions_each_step_checked=True,slack_and_dual_exact_rational_cone_failures_RETAINED=True,total_raw_slack_SOC_defects=sum(z['raw_exact_SOC4_SOC5_slack_violations'] for z in records),total_raw_dual_SOC_defects=sum(z['raw_exact_SOC4_SOC5_dual_violations'] for z in records),total_exact_branch_condition_failures=sum(z['exact_branch_condition_failures'] for z in records),max_state_identity_bound_ratio=maxrat,terminal=terminal,actual_source_queries=method['actual_source_queries'],old_method_solver_projection_Ax_AT_normal_source_NN_raster_native_NOT_replayed=True,local_arithmetic_diagnostics_NOT_candidate_tolerances=True,no_physical_qualification=True))

"""NEW bounded sequential local camera-SOCP method. No physical qualification.

The local affine camera cones are STEP MODELS, not global outer exclusions.
No solver status, common slack or merit tolerance accepts a trajectory.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,sys,time
import numpy as np
from scipy import sparse
from coupled_angle_source_v1 import Chain,camera_source_factor,unrat

CONFIG=dict(major_iterations=40,method_seconds=120.,max_step_rad=.1,damping=1e-3,backtracks=12,armijo=1e-4,exact_check_trigger=1e-7,subproblem_iterations=100,subproblem_seconds=10.,threads=1,linear_scale_rad=.05,depth_scale_m=.01,halfangle_scale=.05)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def oldrat(r):return F(int(r['numerator_hex'],16),int(r['denominator_hex'],16))
def sparse_record(A):
 A=A.tocsc();return dict(shape=list(A.shape),data=A.data,indices=A.indices,indptr=A.indptr)
def returned(solution):
 result={}
 for key in dir(solution):
  if key.startswith('_'):continue
  value=getattr(solution,key)
  if callable(value):continue
  result[key]=str(value)if key=='status'else value
 return result

class SavedAngleModel:
 def __init__(self,S,emit):
  self.S=Path(S);self.P=self.S/'e037-coupled-angle-model-v1';self.binding=read(self.P/'fullclock-coupled-angle-model-binding.json');self.variables=read(self.P/'continuous-angle-variables.json');self.columns=self.binding['columns'];self.n=len(self.variables);self.rows=read(self.P/'all-exact-linear-angle-inequalities.json');self.specs=read(self.P/'all-direct-source-factor-proposals.json');self.domains=read(self.P/'retained-halfangle-domains-and-unknown-source-evidence.json')['retained_domain_factors'];self.basepath=self.P/'unqualified-model-fixed-components-and-initial-F.npz'
  for n,k in [('continuous-angle-variables.json','variables_sha256'),('all-exact-linear-angle-inequalities.json','linear_rows_sha256'),('all-direct-source-factor-proposals.json','direct_factors_sha256'),('retained-halfangle-domains-and-unknown-source-evidence.json','retained_domain_sha256'),(self.basepath.name,'model_arrays_sha256')]:
   if sha(self.P/n)!=self.binding[k]:raise ValueError('saved-model hash mismatch')
  with np.load(self.basepath)as z:self.base=z['model_base_qpos'].copy();self.x0=z['initial_F_variables'].copy()
  FROZEN=self.S/'e037-thumb-ray-independent-validation-v1/frozen';manifest=read(FROZEN/'manifest.json');self.names=manifest['joint_names']['right'];self.chain=Chain((FROZEN/'robot/urdfs/g1/dex3_right.urdf').read_text(),self.names)
  ri=[];ci=[];av=[];self.exact_upper=[];self.exact_terms=[];packing=[]
  emit(dict(kind='BEFORE_NEW_saved_angle_runtime_pack',saved_binding_sha256=sha(self.P/'fullclock-coupled-angle-model-binding.json'),no_model_or_derivative_producer_replayed=True))
  for i,row in enumerate(self.rows):
   terms=[]
   for t in row['active_terms']:
    a=unrat(t['coefficient']);j=self.columns[t['variable']];ri.append(i);ci.append(j);av.append(float(a));terms.append((j,a))
   b=unrat(row['folded_upper']);bf=float(b);self.exact_upper.append(b);self.exact_terms.append(terms);packing.append(dict(row=i,original_row_id=row['id'],exact_upper=row['folded_upper'],actual_float_upper=bf,exact_conversion_error=dict(numerator=str((F(bf)-b).numerator),denominator=str((F(bf)-b).denominator)),coefficients_are_integer_and_exact=all(F(float(a))==a for _,a in terms)))
  self.A=sparse.csr_matrix((av,(ri,ci)),shape=(len(self.rows),self.n));self.b=np.array([float(v)for v in self.exact_upper]);self.linear_scale=CONFIG['linear_scale_rad']*np.maximum(1.,np.asarray(abs(self.A).sum(axis=1)).ravel());self.initial=[];self.source_scale=[]
  for spec in self.specs:
   p=self.P/f'factor-{spec["id"]:03d}-actual-constructor-return.json';record=read(p);source=record['initial_F_source_RETURN'];cols=[v['column']for v in record['active_columns']];joints=[v['finger_index']for v in record['active_columns']];C=np.asarray(source['camera_point']);scale=float(C[2]*max(1.,source['constructed_radius_px']))
   if not math.isfinite(scale)or C[2]<=.01 or scale<=0:raise ValueError('positive unchanged initial source conditioning required')
   self.initial.append(dict(record=record,source_return=source,columns=cols,joints=joints,initial_constructor_sha256=sha(p)));self.source_scale.append(scale)
  self.source_scale=np.array(self.source_scale);self.half_indices=[self.columns[d['angle_variable']]for d in self.domains];self.half_lo=np.array([float(oldrat(d['half_angle_bounds_COPIED'][0]))for d in self.domains]);self.half_hi=np.array([float(oldrat(d['half_angle_bounds_COPIED'][1]))for d in self.domains]);emit(dict(kind='AFTER_NEW_saved_angle_runtime_pack',returned=dict(rows=len(self.rows),columns=self.n,source_factors=len(self.specs),packed_linear_rows=packing,positive_static_linear_scales=self.linear_scale,positive_static_source_scales=self.source_scale,halfangle_columns=self.half_indices,halfangle_bounds_COPIED=self.domains,initial_sources_reused_BY_HASH_NOT_recomputed=True,conditioning_NOT_original_gate_or_confidence=True)))
 def snapshot(self,x,emit,label,initial=False):
  x=np.asarray(x,dtype=float)
  if x.shape!=(self.n,)or not np.isfinite(x).all():raise ValueError('finite complete proposed angle vector required')
  if np.any(abs(x[self.half_indices])>=3):raise ValueError('UNRESOLVED outside retained finite half-angle branch')
  emit(dict(kind='BEFORE_NEW_actual_unlinearized_model_return',label=label,proposed_angles=x,initial_constructor_reuse_BY_HASH=initial));q=self.base.copy()
  for v in self.variables:q[v['frame'],1,v['command_column']]=x[v['column']]
  source=[]
  for spec,cached in zip(self.specs,self.initial,strict=True):
   if initial:
    if not np.array_equal(x,self.x0):raise ValueError('only unchanged F may reuse constructor returns')
    ret=cached['source_return'];emit(dict(kind='AFTER_initial_source_REUSED_BY_HASH',label=label,factor_id=spec['id'],constructor_sha256=cached['initial_constructor_sha256']))
   else:
    f=spec['frame'];emit(dict(kind='BEFORE_NEW_trial_URDF_source_RETURN',label=label,factor_id=spec['id'],root_pose=q[f,1,:7],finger_angles=q[f,1,7:]));point=self.chain.point_jacobian(q[f,1,:7],q[f,1,7:],spec['link']);ret=camera_source_factor(point['point'],point['jacobian'],spec['camera'],spec['stage'],spec['K'],spec['source_pixel'],spec['ORIGINAL_cap_px']);emit(dict(kind='AFTER_NEW_trial_URDF_source_RETURN',label=label,factor_id=spec['id'],actual_chain_RETURN=point,actual_source_RETURN=ret))
   C=np.asarray(ret['camera_point']);J=np.asarray(ret['camera_jacobian'])[:,cached['joints']];B=np.array([[spec['K'][0][0],0.,spec['K'][0][2]-spec['source_pixel'][0]],[0.,spec['K'][1][1],spec['K'][1][2]-spec['source_pixel'][1]]]);source.append(dict(id=spec['id'],columns=cached['columns'],C=C,J=J,B=B,radius=ret['constructed_radius_px'],direct_image_error_px=ret['direct_image_error_px'],actual_source_RETURN=ret))
  u=np.tan(x[self.half_indices]/2);du=.5*(1+u*u);snapshot=dict(x=x.copy(),source=source,linear_violation=(self.A@x-self.b)/self.linear_scale,half_values=u,half_derivatives=du)
  merit=actual_merit(self,snapshot);snapshot['merit']=merit;emit(dict(kind='AFTER_NEW_actual_unlinearized_model_return',label=label,returned=dict(proposed_angles=x,merit=merit,all_original_linear_residuals=snapshot['linear_violation'],retained_half_values=u,retained_half_derivatives=du,source_factor_ids=[s['id']for s in source])));return snapshot
 def exact_acceptance(self,snapshot,emit,label):
  emit(dict(kind='BEFORE_NEW_actual_terminal_model_check',label=label,angles=snapshot['x'],threshold_does_NOT_define_acceptance=True));x=list(map(F,snapshot['x']));bad=[];sourcebad=[];halfbad=[]
  for i,(terms,b)in enumerate(zip(self.exact_terms,self.exact_upper,strict=True)):
   value=sum((a*x[j]for j,a in terms),F(0))-b
   if value>0:bad.append(dict(row_id=i,numerator=str(value.numerator),denominator=str(value.denominator)))
  for s,spec in zip(snapshot['source'],self.specs,strict=True):
   error=s['direct_image_error_px'];C=s['C'];passed=error is not None and C[2]>.01 and error<=spec['ORIGINAL_cap_px']+1e-6
   if not passed:sourcebad.append(dict(factor_id=s['id'],camera_point=C,direct_error_px=error,original_cap_px=spec['ORIGINAL_cap_px'],original_image_tolerance_px=1e-6))
  for i,(u,d)in enumerate(zip(snapshot['half_values'],self.domains,strict=True)):
   if not oldrat(d['half_angle_bounds_COPIED'][0])<=F(float(u))<=oldrat(d['half_angle_bounds_COPIED'][1]):halfbad.append(dict(domain=i,actual_half_angle=u,original_domain_COPIED=d))
  result=dict(all_original_model_checks_passed=not(bad or sourcebad or halfbad),exact_linear_failures=bad,actual_original_image_strict_depth_failures=sourcebad,retained_domain_failures=halfbad,no_NEW_candidate_tolerance=True,wholeclock_independent_URDF_root_source_motion_native_grasp_still_REQUIRED=True);emit(dict(kind='AFTER_NEW_actual_terminal_model_check',label=label,returned=result));return result

def actual_merit(system,snapshot):
 values=[0.,float(np.max(snapshot['linear_violation'],initial=0.))]
 for i,s in enumerate(snapshot['source']):
  C=s['C'];values.append(float((np.linalg.norm(s['B']@C)-s['radius']*C[2])/system.source_scale[i]));values.append(float((.01-C[2])/CONFIG['depth_scale_m']))
 if len(snapshot['half_values']):values.extend([float(np.max((system.half_lo-snapshot['half_values'])/CONFIG['halfangle_scale'])),float(np.max((snapshot['half_values']-system.half_hi)/CONFIG['halfangle_scale']))])
 if not np.isfinite(values).all():raise ValueError('UNRESOLVED nonfinite actual merit')
 return max(values)

def step_problem(system,snapshot,emit,label):
 n=system.n;x=snapshot['x'];emit(dict(kind='BEFORE_NEW_camera_SOCP_canonical_step',label=label,actual_angles=x,configuration=CONFIG));nonneg=sparse.hstack([system.A,sparse.csr_matrix(-system.linear_scale[:,None])],format='csr');rhs=list(system.b-system.A@x);extra=[];extra_b=[];origins=[dict(kind='original_linear_prefix',row_ids=list(range(system.A.shape[0])))]
 def add(values,b,origin):extra.append(values);extra_b.append(float(b));origins.append(origin)
 for i,j in enumerate(system.half_indices):
  for side,sgn,b in[('lower',-1,snapshot['half_values'][i]-system.half_lo[i]),('upper',1,system.half_hi[i]-snapshot['half_values'][i])]:add({j:sgn*snapshot['half_derivatives'][i],n:-CONFIG['halfangle_scale']},b,dict(kind='retained_half_angle_local_'+side,domain=i))
 for s in snapshot['source']:add({**{j:-s['J'][2,k]for k,j in enumerate(s['columns'])},n:-CONFIG['depth_scale_m']},s['C'][2]-.01,dict(kind='separate_depth_local_nonstrict_step_ONLY',factor_id=s['id'],actual_terminal_depth_remains_STRICT=True))
 add({n:-1.},0.,dict(kind='common_elastic_rho_nonnegative'))
 for j in range(n):
  for sign in[-1.,1.]:add({j:sign},CONFIG['max_step_rad'],dict(kind='fixed_algorithm_trust_step',column=j,sign=sign))
 er=[];ec=[];ev=[]
 for i,row in enumerate(extra):
  for j,a in row.items():
   if a:er.append(i);ec.append(j);ev.append(a)
 E=sparse.csr_matrix((ev,(er,ec)),shape=(len(extra),n+1));A=sparse.vstack([nonneg,E],format='csr');b=np.r_[rhs,extra_b];nonnegative_rows=A.shape[0];soc=[];socr=[];socc=[];socv=[];socb=[]
 for i,s in enumerate(snapshot['source']):
  C=s['C'];J=s['J'];V=np.vstack([s['radius']*J[2],s['B']@J]);v=np.r_[s['radius']*C[2],s['B']@C]
  for a in range(3):
   for k,j in enumerate(s['columns']):
    if V[a,k]:socr.append(3*i+a);socc.append(j);socv.append(-V[a,k])
  socr.append(3*i);socc.append(n);socv.append(-system.source_scale[i]);socb.extend(v);soc.append(dict(factor_id=s['id'],canonical_start_row=nonnegative_rows+3*i,dimension=3,radius=s['radius'],positive_static_scale=system.source_scale[i],columns=s['columns'],actual_C=C,actual_J=J,B=s['B']))
 if soc:A=sparse.vstack([A,sparse.csr_matrix((socv,(socr,socc)),shape=(3*len(soc),n+1))],format='csc');b=np.r_[b,socb]
 else:A=A.tocsc()
 P=sparse.diags(np.r_[np.full(n,CONFIG['damping']),0.],format='csc');q=np.r_[np.zeros(n),1.];result=dict(P=P,q=q,A=A,b=b,nonnegative_rows=nonnegative_rows,SOC_blocks=soc,linear_origins=origins)
 emit(dict(kind='AFTER_NEW_camera_SOCP_canonical_step',label=label,returned=dict(P=sparse_record(P),q=q,A=sparse_record(A),b=b,nonnegative_rows=nonnegative_rows,SOC_blocks=soc,linear_origins=origins,local_model_NOT_global_exclusion=True)));return result

def solve_step(problem,emit,label,seconds,private_site):
 sys.path.insert(0,str(private_site));import clarabel
 settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=CONFIG['subproblem_iterations'];settings.time_limit=min(CONFIG['subproblem_seconds'],seconds);settings.max_threads=CONFIG['threads'];cones=[clarabel.NonnegativeConeT(problem['nonnegative_rows'])]+[clarabel.SecondOrderConeT(3)for _ in problem['SOC_blocks']];emit(dict(kind='BEFORE_NEW_SOCP_constructor',label=label,settings=dict(max_iter=settings.max_iter,time_limit=settings.time_limit,max_threads=settings.max_threads,tol_gap_abs=settings.tol_gap_abs,tol_gap_rel=settings.tol_gap_rel,tol_feas=settings.tol_feas),cones=[dict(kind='Nonnegative',dim=problem['nonnegative_rows'])]+[dict(kind='SecondOrder',dim=3)for _ in problem['SOC_blocks']]))
 solver=clarabel.DefaultSolver(problem['P'],problem['q'],problem['A'],problem['b'],cones,settings);emit(dict(kind='AFTER_NEW_SOCP_constructor_RETURN',label=label,returned_type=type(solver).__name__,opaque_constructor_repr=repr(solver)));emit(dict(kind='BEFORE_NEW_SOCP_solve',label=label));solution=solver.solve();result=returned(solution);emit(dict(kind='AFTER_NEW_SOCP_solve_RETURN',label=label,actual_RETURN=result,raw_x_z_s_UNCHANGED=True));return result

def construct(system,emit,private_site):
 start=time.monotonic();emit(dict(kind='BEFORE_ONE_NEW_sequential_camera_construction',configuration=CONFIG,initial_F_angles=system.x0,all_prior_failed_poses_UNUSED=True));x=system.x0.copy();view=system.snapshot(x,emit,'initial-F-BY-HASH',initial=True);calls=0;checks=0;trials=0
 def end(status,iteration,extra=None):
  r=dict(status=status,major_iterations=iteration,subproblem_calls=calls,trial_evaluations=trials,exact_acceptance_checks=checks,actual_seconds=time.monotonic()-start,terminal_angles=x.copy(),terminal_merit=view['merit'],proposal_returned=status=='PROPOSED_actual_original_model_point',all_unsuccessful_terminal_and_intermediate_points_REJECTED=status!='PROPOSED_actual_original_model_point',extra=extra);emit(dict(kind='AFTER_ONE_NEW_sequential_camera_construction_RETURN',actual_RETURN=r));return r
 for iteration in range(CONFIG['major_iterations']):
  if time.monotonic()-start>=CONFIG['method_seconds']:return end('UNRESOLVED_method_time_cap',iteration)
  if view['merit']<=CONFIG['exact_check_trigger']:
   acceptance=system.exact_acceptance(view,emit,f'check-{iteration}');checks+=1
   if acceptance['all_original_model_checks_passed']:return end('PROPOSED_actual_original_model_point',iteration,acceptance)
  problem=step_problem(system,view,emit,f'major-{iteration}');remaining=CONFIG['method_seconds']-(time.monotonic()-start)
  if remaining<=0:return end('UNRESOLVED_method_time_cap',iteration)
  result=solve_step(problem,emit,f'major-{iteration}',remaining,private_site);calls+=1
  if result['status']not in['Solved','AlmostSolved']:return end('UNRESOLVED_subproblem_status',iteration+1,dict(actual_status=result['status'],no_original_family_exclusion=True))
  actual=np.asarray(result['x'],dtype=float)
  if actual.shape!=(system.n+1,)or not np.isfinite(actual).all():return end('UNRESOLVED_nonfinite_step_RETURN',iteration+1)
  step=actual[:-1];rho=float(actual[-1]);predicted=view['merit']-rho;emit(dict(kind='AFTER_NEW_raw_predicted_reduction',iteration=iteration,actual_step=step,actual_rho=rho,predicted_reduction=predicted,raw_RETURN_NOT_clipped_or_repaired=True))
  if not predicted>0:return end('UNRESOLVED_no_predicted_descent',iteration+1)
  accepted=False
  for backtrack in range(CONFIG['backtracks']):
   alpha=2.**(-backtrack);trial=x+alpha*step;emit(dict(kind='BEFORE_NEW_unlinearized_Armijo_trial',iteration=iteration,backtrack=backtrack,alpha=alpha,proposed_angles=trial,raw_step=step))
   if float(abs(alpha*step).max(initial=0.))>CONFIG['max_step_rad']:emit(dict(kind='AFTER_NEW_unlinearized_Armijo_trial',iteration=iteration,backtrack=backtrack,accepted=False,reason='algorithm_trust_bound; rawstep_UNCHANGED'));continue
   try:trialview=system.snapshot(trial,emit,f'trial-{iteration}-{backtrack}')
   except (ValueError,FloatingPointError)as ex:emit(dict(kind='AFTER_NEW_unlinearized_Armijo_trial',iteration=iteration,backtrack=backtrack,accepted=False,reason=repr(ex)));continue
   trials+=1;bound=view['merit']-CONFIG['armijo']*alpha*predicted;accepted=trialview['merit']<=bound;emit(dict(kind='AFTER_NEW_unlinearized_Armijo_trial',iteration=iteration,backtrack=backtrack,actual_merit=trialview['merit'],original_merit=view['merit'],Armijo_bound=bound,accepted=accepted))
   if accepted:x=trial;view=trialview;break
  if not accepted:return end('UNRESOLVED_no_Armijo_step',iteration+1)
 if view['merit']<=CONFIG['exact_check_trigger']:
  acceptance=system.exact_acceptance(view,emit,'terminal-cap-check');checks+=1
  if acceptance['all_original_model_checks_passed']:return end('PROPOSED_actual_original_model_point',CONFIG['major_iterations'],acceptance)
 return end('UNRESOLVED_iteration_cap',CONFIG['major_iterations'])

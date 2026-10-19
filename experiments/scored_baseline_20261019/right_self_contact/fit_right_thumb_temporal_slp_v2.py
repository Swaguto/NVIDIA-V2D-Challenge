"""NEW single whole-clock sequential-convex source/self retarget, original motion caps."""
from right_self_context_v3 import *
import datetime,xml.etree.ElementTree as ET
import scipy.sparse as sp
from scipy.optimize import linprog
start=time.monotonic();O=S/'e037-right-thumb-temporal-slp-v2';N=len(q);nv=N*3;axes=np.array([4,5,6]);base=q[:,1,7:].copy();before=[evaluate(i,base[i]) for i in range(N)];origin=aligned[:,1,7:];lo=np.array([bounds(i)[0][axes] for i in range(N)])-base[:,axes];hi=np.array([bounds(i)[1][axes] for i in range(N)])-base[:,axes];image_frames=[i for i,b in enumerate(before) if b['source_mask'][1]];missing=[i for i,b in enumerate(before) if not b['source_mask'].all()];lo[missing]=0;hi[missing]=0
# No source data is synthesized for absent frames; all right commands at those frames are fixed.
names=manifest['joint_names']['right'];limits_xml={j.attrib['name']:j.find('limit').attrib for j in ET.parse(F/'robot/urdfs/g1/dex3_right.urdf').getroot().findall('joint') if j.attrib['type']!='fixed'};velocity=np.array([float(limits_xml[n]['velocity']) for n in names])[axes]
D1=sp.kron(sp.diags([-np.ones(N-1),np.ones(N-1)],[0,1],shape=(N-1,N)),sp.eye(3),format='csc');D2=D1[:-3] # replaced by explicit clock operator below
D2=sp.kron(sp.diags([np.ones(N-2),-2*np.ones(N-2),np.ones(N-2)],[0,1,2],shape=(N-2,N)),sp.eye(3),format='csc');I=sp.eye(nv,format='csc');old_delta=(base[:,axes]-origin[:,axes]).ravel();baseflat=base[:,axes].ravel();dt=np.diff(ts)
H=I+.2*(D1.T@D1)+.05*(D2.T@D2)
hard=sp.vstack([I,D1,D2,D1],format='csc');hardlo=np.r_[lo.ravel(),-.05-D1@old_delta,-.05-D2@old_delta,-(dt[:,None]*velocity).ravel()-D1@baseflat];hardhi=np.r_[hi.ravel(),.05-D1@old_delta,.05-D2@old_delta,(dt[:,None]*velocity).ravel()-D1@baseflat]
penalty=1e4;history=[];d=np.zeros(nv);trust=.04;last_qp_status=None;converged=False
# Native/pixel differential controls at the NEW right bindings, no old left controls repeated.
fd=[]
for frame in [4,110,119]:
 b=before[frame];h=1e-5
 for a in axes:
  x=np.eye(7)[a]*h;actual=(evaluate(frame,base[frame]+x)['pixel_residual'][1]-evaluate(frame,base[frame]-x)['pixel_residual'][1])/(2*h)
  fd.append(float(np.max(abs(actual-b['pixel_jac'][1,:,a]))))
assert max(fd)<1e-5

def values(d):
 angles=base.copy();angles[:,axes]+=d.reshape(N,3);return [evaluate(i,angles[i]) for i in range(N)]
def assess(d,vals):
 collision=sum(np.maximum(0.,-v['gaps']-.0005).sum()/.01 for v in vals)
 source=sum(max(0.,vals[i]['errors'][1]-before[i]['errors'][1])/max(before[i]['errors'][1],1.) for i in image_frames)
 maxsrc=max([float(vals[i]['errors'][1]-before[i]['errors'][1]) for i in image_frames],default=0.)
 maxpen=max(max(0.,float(-v['gaps'].min())) for v in vals)
 merit=float(abs(d).sum())+penalty*(collision+source)
 return dict(merit=merit,collision_scaled=collision,source_scaled=source,max_source_regression_px=maxsrc,max_self_penetration_m=maxpen)
vals=before;status='iteration cap';dirs=np.c_[np.cos(np.arange(64)*2*np.pi/64),np.sin(np.arange(64)*2*np.pi/64)]
for iteration in range(70):
 if time.monotonic()-start>205:status='internal time cap';break
 current=assess(d,vals);rows=[];bl=[];bu=[];kinds=[]
 for i,v in enumerate(vals):
  for k in np.flatnonzero(v['gaps']<0):
   row=np.zeros(nv);row[i*3:(i+1)*3]=v['gap_jac'][k,axes]/.01;rows.append(sp.csr_matrix(row));bl.append(float(row@d-(v['gaps'][k]+.0005)/.01));bu.append(np.inf);kinds.append('gap')
 # Source image error uses an explicit 2-D polygon linearization, then exact Euclidean checks.
 for i in image_frames:
  scale=max(before[i]['errors'][1],1.);pj=vals[i]['pixel_jac'][1][:,axes];res=vals[i]['pixel_residual'][1]
  for normal in dirs:
   row=np.zeros(nv);row[i*3:(i+1)*3]=np.einsum('i,ij->j',normal,pj,optimize=False)/scale;rows.append(sp.csr_matrix(row));bl.append(-np.inf);bu.append(float(row@d+(before[i]['errors'][1]-normal@res)/scale));kinds.append('image')
 m=len(rows);G=sp.vstack(rows,format='csc');ngap=sum(k=='gap' for k in kinds);ns=ngap+len(image_frames)
 # One nonnegative slack per native pair or image FRAME; image directions share a slack.
 sr=[];sc=[];sv=[]
 for rowid,k in enumerate(kinds):
  sr.append(rowid);sc.append(rowid if k=='gap' else ngap+(rowid-ngap)//len(dirs));sv.append(-1.)
 slack=sp.csc_matrix((sv,(sr,sc)),shape=(m,ns))
 sign=np.array([-1. if k=='gap' else 1. for k in kinds]);Gc=sp.diags(sign)@G
 soft=sp.hstack([Gc,slack,sp.csc_matrix((m,nv))],format='csc');softb=np.array([-bl[i] if k=='gap' else bu[i] for i,k in enumerate(kinds)])
 lower=hardlo.copy();upper=hardhi.copy();lower[:nv]=np.maximum(lower[:nv],d-trust);upper[:nv]=np.minimum(upper[:nv],d+trust)
 zeroh=sp.csc_matrix((hard.shape[0],ns+nv));hardfull=sp.hstack([hard,zeroh],format='csc')
 absplus=sp.hstack([I,sp.csc_matrix((nv,ns)),-I],format='csc');absminus=sp.hstack([-I,sp.csc_matrix((nv,ns)),-I],format='csc')
 A=sp.vstack([hardfull,-hardfull,soft,absplus,absminus],format='csc');b=np.r_[upper,-lower,softb,np.zeros(2*nv)]
 linear=np.r_[np.zeros(nv),np.full(ns,penalty),np.ones(nv)]
 solution=linprog(linear,A_ub=A,b_ub=b,bounds=[(None,None)]*nv+[(0,None)]*(ns+nv),method='highs',options=dict(primal_feasibility_tolerance=1e-8,dual_feasibility_tolerance=1e-8,time_limit=30.))
 last_qp_status=str(solution.message)
 row=dict(iteration=iteration,LP_success=bool(solution.success),LP_status=last_qp_status,LP_iterations=int(solution.nit),trust_rad=trust,before=current)
 if not solution.success:history.append(row);status='LP numerical failure';break
 proposal=solution.x[:nv];direction=proposal-d;accepted=False
 for fraction in [1.,.5,.25,.125,.0625]:
  trial=d+fraction*direction;vs=values(trial);merit=assess(trial,vs)
  if merit['merit']<current['merit']-1e-8:
   accepted=True;d=trial;vals=vs;row.update(fraction=fraction,step_max_rad=float(abs(fraction*direction).max()),after=merit);break
 if not accepted:
  row.update(accepted=False,proposal_step_max_rad=float(abs(direction).max()));history.append(row)
  if abs(direction).max()<2e-6 and current['max_self_penetration_m']<=.0005+1e-8 and current['max_source_regression_px']<=1e-6:converged=True;status='small-step convergence with original gates';break
  trust*=.5
  if trust<1e-5:status='no acceptable bounded merit step';break
 else:
  history.append(row)
  if row['fraction']==1.:trust=min(.06,trust*1.25)
  if row['step_max_rad']<2e-6 and row['after']['max_self_penetration_m']<=.0005+1e-8 and row['after']['max_source_regression_px']<=1e-6:converged=True;status='small-step convergence with original gates';break
 if iteration%5==0:print(json.dumps(row),flush=True)
final=assess(d,vals);commands=q.copy();commands[:,1,7+axes]+=d.reshape(N,3);np.testing.assert_array_equal(commands[:,0],q[:,0]);np.testing.assert_array_equal(commands[:,:,:7],q[:,:,:7]);np.testing.assert_array_equal(commands[:,1,7:11],q[:,1,7:11]);np.testing.assert_array_equal(commands[missing,1],q[missing,1]);hardvalue=hard@d;hardpass=bool((hardvalue>=hardlo-1e-7).all() and (hardvalue<=hardhi+1e-7).all());sourcepass=bool(all((v['errors'][b['source_mask']]<=b['errors'][b['source_mask']]+1e-6).all() for v,b in zip(vals,before,strict=True)));passed=bool(converged and hardpass and sourcepass and final['max_self_penetration_m']<=.001)
np.savez_compressed(O/'right-thumb-reference-commands.npz',robot_qpos=commands,original_robot_qpos=q,frame_indices=np.arange(N),timestamps_s=ts)
report=dict(schema='v2d.right-thumb-whole-clock-source-self-SLP.v2',prior_QP_failed_before_any_new_candidate=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),context_sha256=sha(S/'compute/right_self_context_v3.py'),input_manifest_sha256=sha(F/'manifest.json'),method='One whole-clock sequential linearization with shared image-frame slack, exact nonlinear merit checks, hard original component/adjacent/second/URDF speed bounds; prior QP numerical failure retained; no restarts or best-intermediate promotion',history=history,return_status=status,numerical_convergence=converged,final=final,all_original_motion_caps_passed=hardpass,available_source_image_nonregression=sourcepass,pixel_Jacobian_max_FD_error=max(fd),missing_source_frames_frozen=missing,available_right_source_image_points=len(image_frames)*4,verified_left_roots_objects_nonthumb_source_human_flags_exact_unchanged=True,derived_robot_angles_not_measured=True,right_source_self_motion_component_passed=passed,commands_sha256=sha(O/'right-thumb-reference-commands.npz'),whole_reference_or_physical_task_qualified=False,source_rows=[dict(frame=i,source_direct=bool(flags['hand_directly_observed'][i,1]),source_metric=bool(flags['hand_metric_valid'][i,1]),source_available=b['source_mask'].tolist(),before_after_px=[[float(x) if np.isfinite(x) else None for x in w['errors']] for w in [b,v]],self_penetration_m=max(0.,float(-v['gaps'].min()))) for i,(b,v) in enumerate(zip(before,vals,strict=True))],no_native_compile_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['return_status','numerical_convergence','final','all_original_motion_caps_passed','available_source_image_nonregression','right_source_self_motion_component_passed','wall_seconds']},indent=2))

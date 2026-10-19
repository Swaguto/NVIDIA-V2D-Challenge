"""NEW root differential controls at the returned rejected trajectory, no solve repeat."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-source-root-derivative-controls-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
from source_root_differential_v1 import *
from scipy.spatial.transform import Rotation
prior=S/'e037-source-root-finger-trajectory-v1/report.json';p=json.loads(prior.read_text());assert not p['solver_success'];commands=S/'e037-source-root-finger-trajectory-v1/reference-commands.npz';assert sha(commands)==p['commands_sha256']
with np.load(commands) as z:q=z['robot_qpos'].copy()
authority=json.loads((S/'e037-wrist-source-authority-v1/report.json').read_text());B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037';rows=[];np.testing.assert_array_equal(left_jacobian([0,0,0]),np.eye(3))
for frame in [104,109,111,113]:
 c.q=q.copy();base=q[frame,1];alignedR=Rotation.from_quat(c.aligned[frame,1,3:7]).as_matrix();r=Rotation.from_matrix(Rotation.from_quat(base[3:7]).as_matrix()@alignedR.T).as_rotvec();v=c.evaluate(frame,base[7:]);points=c.data.xpos[c.tipids].copy();camera=c.cameras[frame];A=camera[:3,:3].T@c.stage.T;offset=camera[:3,:3].T@camera[:3,3];J=pixel_root_jacobian(points,base[:3],r,(A,offset),c.K);rec=json.loads((B/f'source/hands/right/{frame:06d}.json').read_text());sourceR=c.stage@camera[:3,:3]@Rotation.from_rotvec(rec['mano']['global_orient']).as_matrix()@np.diag([-1.,-1.,1.]);rotation=Rotation.from_rotvec(r).as_matrix()@alignedR;ori=Rotation.from_matrix(sourceR@rotation.T).as_rotvec();orientation_gradient=-2*np.einsum('i,ij->j',ori,left_jacobian(r),optimize=False);pixel_errors=[];orient_errors=[];norm_errors=[];gap_changes=[]
 for h in [2e-6,1e-7]:
  numeric=[];on=[];en=[];gn=[]
  for axis in range(3):
   uvpair=[];orpair=[];errpair=[];gappair=[]
   for sign in [-1,1]:
    x=r.copy();x[axis]+=sign*h;rr=Rotation.from_rotvec(x).as_matrix()@alignedR;c.q[frame,1,3:7]=Rotation.from_matrix(rr).as_quat();value=c.evaluate(frame,base[7:]);uvpair.append(value['pixel_residual']);orpair.append(float(Rotation.from_matrix(sourceR@rr.T).magnitude()**2));errpair.append(value['errors']);gappair.append(value['gaps'])
   numeric.append((uvpair[1]-uvpair[0])/(2*h));on.append((orpair[1]-orpair[0])/(2*h));en.append((errpair[1]-errpair[0])/(2*h));gn.append(float(np.max(abs((gappair[1]-gappair[0])/(2*h)))))
  numeric=np.stack(numeric,axis=2);pixel_errors.append(float(np.max(abs(numeric-J))));orient_errors.append(float(np.max(abs(np.array(on)-orientation_gradient))));analytic_error=np.einsum('ni,nij->nj',v['pixel_residual']/np.maximum(v['errors'],1e-12)[:,None],J,optimize=False);norm_errors.append(float(np.max(abs(np.stack(en,axis=1)-analytic_error))));gap_changes.append(max(gn))
 rows.append(dict(frame=frame,pixel_derivative_max_error_px_per_rad=pixel_errors,orientation_squared_derivative_max_error_rad=orient_errors,norm_error_root_derivative_truncation_px_per_rad=norm_errors,rigid_root_self_gap_derivative_max_m_per_rad=gap_changes,smallest_original_image_error_px=float(v['errors'][1:].min())))
assert max(max(r['pixel_derivative_max_error_px_per_rad']) for r in rows)<5e-5;assert max(max(r['orientation_squared_derivative_max_error_rad']) for r in rows)<1e-8
report=dict(schema='v2d.source-root-derivative-controls.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),helper_sha256=sha(S/'compute/source_root_differential_v1.py'),context_sha256=sha(S/'compute/source_orientation_context_v1.py'),rejected_trajectory_report_sha256=sha(prior),rejected_commands_sha256=sha(commands),rows=rows,explicit_SO3_pixel_and_orientation_derivatives_verified=True,source_norm_constraints_can_be_equivalently_squared_without_changing_image_disk=True,prior_SLSQP_iteration_cap_and_BLAS_runtime_warnings_retained=True,root_norm_derivative_error_not_proven_sole_failure_cause=True,no_source_pose_fit_optimizer_or_old_native_audit_repeated=True,no_candidate_promoted=True,new_GPU_jobs=0,new_PPO_steps=0,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

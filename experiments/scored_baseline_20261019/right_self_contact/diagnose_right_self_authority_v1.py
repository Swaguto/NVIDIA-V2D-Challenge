"""NEW right worst-pair differential and source/bound authority diagnosis, no mutation."""
from right_self_context_v1 import *
import datetime
start=time.monotonic();O=S/'e037-right-self-authority-v1';cached=S/'e037-left-thumb-independent-validation-v1/fresh-candidate-self-contacts.json';saved=json.loads(cached.read_text());bad=[]
for row in saved['frames']:
 contacts=[c for c in row['contacts'] if c['hand']=='right' and c['signed_gap_m']<-.001]
 if contacts:bad.append(dict(frame=row['reference_frame'],contacts=contacts))
rows=[]
for frame in [35,116,119,121]:
 angles=q[frame,1,7:];v=evaluate(frame,angles);lo,hi=bounds(frame);controls=[]
 for k in np.flatnonzero(v['gaps']<0):
  central=[]
  for h in [1e-5,5e-6]:
   diffs=[]
   for j in range(7):
    d=np.eye(7)[j]*h;diffs.append((evaluate(frame,angles+d)['gaps'][k]-evaluate(frame,angles-d)['gaps'][k])/(2*h))
   central.append(diffs)
  controls.append(dict(geom_ids=list(pairs[k]),bodies=[bn[int(model.geom_bodyid[g])] for g in pairs[k]],gap_m=float(v['gaps'][k]),gap_jacobian_m_per_rad=v['gap_jac'][k].tolist(),finite_differences=central,maximum_derivative_error=float(np.max(abs(np.asarray(central)-v['gap_jac'][k])))))
 rows.append(dict(frame=frame,direct=bool(flags['hand_directly_observed'][frame,1]),metric_flag=bool(flags['hand_metric_valid'][frame,1]),valid=bool(flags['hand_valid'][frame,1]),right_angles_rad=angles.tolist(),component_and_URDF_lower_authority_rad=(lo-angles).tolist(),component_and_URDF_upper_authority_rad=(hi-angles).tolist(),source_image_errors_px=v['errors'].tolist(),source_image_error_jacobians_px_per_rad=v['error_jac'].tolist(),native_active_pair_derivative_controls=controls))
report=dict(schema='v2d.right-self-contact-authority.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),context_sha256=sha(S/'compute/right_self_context_v1.py'),new_start_freeze=str(F),new_start_manifest_sha256=sha(F/'manifest.json'),cached_audit_sha256=sha(cached),right_bad_frames=bad,worst_frame_controls=rows,left_verified_component_and_all_commands_not_modified=True,root_source_human_authority_and_model_not_changed=True,robot_fingers_derived_not_measured=True,no_completed_jobs_repeated=True,no_native_compile_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(bad_frames=[r['frame'] for r in bad],controls=rows,wall_seconds=report['wall_seconds']),indent=2))

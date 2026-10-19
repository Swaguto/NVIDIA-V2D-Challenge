"""NEW right multi-pair gap/source bracket construction; preserve verified left."""
from right_self_context_v1 import *
import datetime,xml.etree.ElementTree as ET
from scipy.optimize import brentq
start=time.monotonic();O=S/'e037-right-thumb-bracketed-v1';returned=q.copy();rows=[]
for frame in range(len(q)):
 if time.monotonic()-start>215:raise TimeoutError('right constructor wall guard')
 base=q[frame,1,7:];lo,hi=bounds(frame);lower=lo-base;upper=hi-base;before=evaluate(frame,base)
 chosen=np.zeros(7);success=True;message='unchanged, already at self buffer';compensation=dict(required=False)
 cache={}
 def ev(x):
  key=tuple(x)
  if key not in cache:cache[key]=evaluate(frame,base+x)
  return cache[key]
 def gap_root(delta=0.,axis=4):
  def point(v):x=np.zeros(7);x[axis]=delta;x[5]=v;return x
  def fn(v):return float(ev(point(v))['gaps'].min()+.0005)
  a,b=fn(0.),fn(float(upper[5]))
  if a>=0:return point(0.),dict(converged=True,iterations=0,gap_bracket_m=[a,b])
  if b<0:raise ValueError('no positive thumb1 clearance bracket in original bounds')
  v,result=brentq(fn,0.,float(upper[5]),xtol=1e-12,rtol=1e-14,maxiter=100,full_output=True)
  if not result.converged:raise ValueError('gap root failed')
  return point(v),dict(converged=True,iterations=int(result.iterations),gap_bracket_m=[a,b])
 details={}
 try:
  if before['gaps'].min()<-.0005:
   chosen,details=gap_root();after=ev(chosen);cap=before['errors'][1]-min(.001,.01*before['errors'][1]);residual=float(after['errors'][1]-cap)
   if residual>0:
    active=int(after['gaps'].argmin());g=after['gap_jac'][active];options=[]
    for axis in [4,6]:
     derivative=float(after['error_jac'][1,axis]-after['error_jac'][1,5]*g[axis]/g[5]) if abs(g[5])>1e-12 else float('nan')
     if not np.isfinite(derivative) or abs(derivative)<1e-8:continue
     endpoint=-1.5*residual/derivative
     if lower[axis]<=endpoint<=upper[axis]:options.append((abs(endpoint),axis,derivative,endpoint))
    if not options:raise ValueError('no source-compensation direction inside original bounds')
    _,axis,derivative,endpoint=min(options);ex,_=gap_root(endpoint,axis);endvalue=float(ev(ex)['errors'][1]-cap)
    compensation=dict(required=True,axis=int(axis),implicit_source_derivative_px_per_rad=derivative,endpoint_rad=endpoint,predeclared_bracket_factor=1.5,source_residual_bracket_px=[residual,endvalue])
    if endpoint==0 or endvalue>0:raise ValueError('single directed source bracket failed')
    def source_fn(v):return float(ev(gap_root(v,axis)[0])['errors'][1]-cap)
    root,result=brentq(source_fn,min(0.,endpoint),max(0.,endpoint),xtol=1e-12,rtol=1e-14,maxiter=100,full_output=True)
    if not result.converged:raise ValueError('source root failed')
    chosen,details=gap_root(root,axis);compensation['source_root_iterations']=int(result.iterations)
   message='native all-right-self gap and active thumb source constraint constructed'
 except ValueError as exc:success=False;message=str(exc)
 after=ev(chosen);returned[frame,1,7:]=base+chosen
 bound=bool((chosen>=lower-1e-8).all() and (chosen<=upper+1e-8).all());source=bool((after['errors']<=before['errors']+1e-6).all());depth=max(0.,float(-after['gaps'].min()))
 rows.append(dict(frame=frame,source_direct=bool(flags['hand_directly_observed'][frame,1]),source_metric_flag=bool(flags['hand_metric_valid'][frame,1]),construction_success=success,message=message,gap_root=details,compensation=compensation,delta_rad=chosen.tolist(),source_before_after_px=[before['errors'].tolist(),after['errors'].tolist()],source_nonregression=source,in_original_bounds=bound,max_self_penetration_m=depth,single_knot_checks_passed=bool(success and source and bound and depth<=.001)))
np.testing.assert_array_equal(returned[:,0],q[:,0]);np.testing.assert_array_equal(returned[:,:,:7],q[:,:,:7]);np.testing.assert_array_equal(returned[:,1,7:11],q[:,1,7:11]);motion=[]
for si,side in enumerate(['left','right']):
 urdf=ET.parse(F/f'robot/urdfs/g1/dex3_{side}.urdf').getroot();lj={j.attrib['name']:j.find('limit').attrib for j in urdf.findall('joint') if j.attrib['type']!='fixed'};names=manifest['joint_names'][side];lo=np.array([float(lj[n]['lower']) for n in names]);hi=np.array([float(lj[n]['upper']) for n in names]);vmax=np.array([float(lj[n]['velocity']) for n in names]);speed=abs(np.diff(returned[:,si,7:],axis=0)/np.diff(ts)[:,None]);delta=returned[:,si,7:]-aligned[:,si,7:]
 motion.append(dict(hand=side,component_rad=float(abs(delta).max()),adjacent_added_rad=float(abs(np.diff(delta,axis=0)).max()),second_added_rad=float(abs(np.diff(delta,n=2,axis=0)).max()),position_pass=bool(((returned[:,si,7:]>=lo-1e-8)&(returned[:,si,7:]<=hi+1e-8)).all()),velocity_pass=bool((speed<=vmax+1e-7).all()),component_and_temporal_pass=bool(abs(delta).max()<=.25+1e-7 and abs(np.diff(delta,axis=0)).max()<=.05+1e-7 and abs(np.diff(delta,n=2,axis=0)).max()<=.05+1e-7),actual_velocity_max_rad_s=speed.max(0).tolist(),actual_velocity_caps_rad_s=vmax.tolist()))
np.savez_compressed(O/'right-thumb-reference-commands.npz',robot_qpos=returned,original_robot_qpos=q,frame_indices=np.arange(len(q)),timestamps_s=ts)
report=dict(schema='v2d.right-thumb-bracketed-source-self.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),context_sha256=sha(S/'compute/right_self_context_v1.py'),input_manifest_sha256=sha(F/'manifest.json'),rows=rows,motion=motion,all_single_knots_passed=all(r['single_knot_checks_passed'] for r in rows),failed_frames=[r['frame'] for r in rows if not r['single_knot_checks_passed']],all_original_motion_caps_passed=all(r['position_pass'] and r['velocity_pass'] and r['component_and_temporal_pass'] for r in motion),all_human_data_source_authority_roots_objects_left_and_nonthumb_unchanged=True,derived_robot_angles_not_measured_human=True,commands_sha256=sha(O/'right-thumb-reference-commands.npz'),whole_reference_or_physical_task_qualified=False,no_native_compile_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['all_single_knots_passed','failed_frames','all_original_motion_caps_passed','motion','wall_seconds']},indent=2))

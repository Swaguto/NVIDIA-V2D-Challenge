"""NEW scalar source-level continuation with BOTH actual 2R joints free.

One source-gradient direction per frame, one bracketed level-set root. No wrist
mode/amplitude changes, optimizer restarts, contact scoring or old graph reruns.
Original whole-clock four-joint motion is checked before any native execution.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-continuous-2R-source-projection-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
import thumb_conic_geometry_v1 as g
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.kinematics import URDFKinematics
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();j=(O/'all-source-level-probes-returns-and-URDF-checks.jsonl').open('w');l1,l2=.0458,.035
def value_jac(angles,H,K,pixel):
    a,b=angles;xy=np.array([l1*np.cos(a)+l2*np.cos(a+b),l1*np.sin(a)+l2*np.sin(a+b)]);Jxy=np.array([[-l1*np.sin(a)-l2*np.sin(a+b),-l2*np.sin(a+b)],[l1*np.cos(a)+l2*np.cos(a+b),l2*np.cos(a+b)]]);C=H@np.r_[xy,1.]
    if C[2]<=.01:return float('inf'),np.zeros(2),C
    uv=np.array([K[0,0]*C[0]/C[2]+K[0,2],K[1,1]*C[1]/C[2]+K[1,2]]);res=uv-pixel;error=float(np.linalg.norm(res));P=np.array([[K[0,0]/C[2],0,-K[0,0]*C[0]/C[2]**2],[0,K[1,1]/C[2],-K[1,1]*C[1]/C[2]**2]]);grad=(res/max(error,1e-15))@P@H[:,:2]@Jxy;return error,grad,C
def project(guide,box,H,K,pixel,cap,label):
    guide=np.array(guide,float);error,grad,C=value_jac(guide,H,K,pixel);target=cap-.02
    if error<=cap+1e-6:return dict(status='original_source_gate_already_passed',angles=guide,source_error=error,probes=[])
    direction=-grad;direction[(guide<=box[:,0]+1e-12)&(direction<0)]=0;direction[(guide>=box[:,1]-1e-12)&(direction>0)]=0;norm=float(np.linalg.norm(direction));probes=[]
    if norm<1e-12:return dict(status='unresolved_zero_bounded_source_gradient',angles=None,source_error=error,probes=probes)
    direction/=norm;maxstep=float('inf')
    for k in range(2):
        if direction[k]>0:maxstep=min(maxstep,(box[k,1]-guide[k])/direction[k])
        elif direction[k]<0:maxstep=min(maxstep,(box[k,0]-guide[k])/direction[k])
    if maxstep<=0:return dict(status='unresolved_no_bounded_step',angles=None,source_error=error,probes=probes)
    def evaluate(step):
        angles=guide+step*direction;val,_,C=value_jac(angles,H,K,pixel);row=dict(label=label,step_rad=step,angles_rad=angles,camera_point=C,source_error_px=val,constructive_source_target_px=target);g.journal(j,dict(kind='new_source_level_scalar_probe',record=row));probes.append(row);return val-target
    high=min(maxstep,max(1e-8,(error-target)/norm*1.25));left=0.;right=None
    for _ in range(17):
        residual=evaluate(high)
        if np.isfinite(residual) and residual<=0:right=high;break
        if high>=maxstep:break
        high=min(maxstep,2*high)
    if right is None:return dict(status='unresolved_single_direction_bracket',angles=None,source_error=error,direction=direction,maxstep=maxstep,probes=probes)
    root=brentq(evaluate,left,right,xtol=1e-12,rtol=1e-14,maxiter=100);angles=guide+root*direction;error,_,C=value_jac(angles,H,K,pixel);out=dict(status='constructive_two_joint_source_level_root',angles=angles,source_error=error,direction=direction,root_step_rad=root,maxstep=maxstep,probes=probes);g.journal(j,dict(kind='source_level_root_return_before_checks',label=label,record=out));return out
controls=[];K=np.diag([100.,100.,1.]);H=np.eye(3);guide=np.array([.3,.6]);desired=np.array([.2,.5]);a,b=desired;pixel=100*np.array([l1*np.cos(a)+l2*np.cos(a+b),l1*np.sin(a)+l2*np.sin(a+b)]);box=np.array([[0.,.5],[0.,1.]])
answer=project(guide,box,H,K,pixel,.3,'synthetic_two_joint_level');passed=answer['angles'] is not None and answer['source_error']<=.3+1e-6;controls.append(dict(name='two_frequency_source_level_root',passed=passed));assert passed
answer=project(desired,box,H,K,pixel,.3,'synthetic_already_inside');passed=np.array_equal(answer['angles'],desired);controls.append(dict(name='valid_source_point_is_not_moved',passed=passed));assert passed
error,gradient,_=value_jac(guide,H,K,pixel);finite=[]
for k in range(2):
    d=np.eye(2)[k]*1e-6;finite.append((value_jac(guide+d,H,K,pixel)[0]-value_jac(guide-d,H,K,pixel)[0])/2e-6)
passed=abs(gradient-np.array(finite)).max()<1e-7;controls.append(dict(name='two_joint_source_gradient_independent_difference',passed=bool(passed)));assert passed
for row in controls:g.journal(j,dict(kind='new_continuous_2R_source_analytic_contract',record=row))
rootpath=S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz'
with np.load(rootpath) as z:q=z['robot_qpos'].copy();weights=z['weights'].copy()
source_rows={v['frame']:v for v in json.loads((S/'e037-thumb-ray-independent-validation-v1/report.json').read_text())['source_rows'] if v['hand']=='right'};kin=URDFKinematics(c.F/'robot/urdfs/g1/dex3_right.urdf');records=[];unresolved=[];closuremax=0.;sourcebad=[]
for frame in np.flatnonzero(weights>1e-14):
    frame=int(frame);cmd=q[frame,1];R=Rotation.from_quat(cmd[3:7]).as_matrix();camera=c.cameras[frame];A=camera[:3,:3].T@c.stage.T;offset=camera[:3,:3].T@camera[:3,3];lo,hi=c.bounds(frame)
    for digit,ji,si,sign in [('index',0,8,1),('middle',2,12,-1)]:
        pixel=c.uv((c.human[frame,1,[si]]-camera[:3,3])@camera[:3,:3])[0];cap=source_rows[frame]['source_before_after_px'][0][2 if digit=='index' else 3];H=np.c_[A@R@np.eye(3)[:,:2],A@(cmd[:3]+R@np.array([.0777,-.0016,sign*.0285]))-offset];box=np.c_[lo[ji:ji+2],hi[ji:ji+2]];result=project(c.q[frame,1,7+ji:9+ji],box,H,c.K,pixel,cap,dict(frame=frame,digit=digit));entry=dict(frame=frame,digit=digit,result=result,original_source_cap_px=cap)
        if result['angles'] is None:unresolved.append(dict(frame=frame,digit=digit,status=result['status']));g.journal(j,dict(kind='unresolved_new_continuous_source_branch',record=entry));records.append(entry);continue
        cmd[7+ji:9+ji]=result['angles'];g.journal(j,dict(kind='BEFORE_independent_continuous_source_URDF_point',frame=frame,digit=digit,command=cmd.copy()));poses=kin.poses(cmd[:7],dict(zip(c.manifest['joint_names']['right'],cmd[7:],strict=True)));world=poses['right_'+digit+'_tip'][:3,3];C=(world@c.stage-camera[:3,3])@camera[:3,:3];error=float(np.linalg.norm(c.uv(C[None])[0]-pixel));_,_,predicted=value_jac(result['angles'],H,c.K,pixel);closure=float(abs(C-predicted).max());position=bool((result['angles']>=box[:,0]-1e-8).all() and (result['angles']<=box[:,1]+1e-8).all());passed=bool(error<=cap+1e-6 and position and closure<1e-8);entry.update(command=cmd.copy(),actual_URDF_source_error_px=error,actual_URDF_camera_closure_m=closure,source_position_passed=passed);g.journal(j,dict(kind='AFTER_independent_continuous_source_URDF_point',record=entry));records.append(entry);closuremax=max(closuremax,closure)
        if not passed:sourcebad.append(dict(frame=frame,digit=digit))
delta=q[:,1,7:11]-c.aligned[:,1,7:11];speed=abs(np.diff(q[:,1,7:11],axis=0))/np.diff(c.ts)[:,None];motion=dict(component_rad=float(abs(delta).max()),added_adjacent_rad=float(abs(np.diff(delta,axis=0)).max()),added_second_rad=float(abs(np.diff(delta,n=2,axis=0)).max()),maximum_URDF_velocity_rad_s=float(speed.max()));bad=[]
for key,cap in [('component_rad',.25),('added_adjacent_rad',.05),('added_second_rad',.05),('maximum_URDF_velocity_rad_s',12.)]:
    if motion[key]>cap+1e-7:bad.append(key)
g.journal(j,dict(kind='full_clock_original_four_joint_motion_return',motion=motion,bad_keys=bad));j.close();np.savez_compressed(O/'UNQUALIFIED-continuous-four-joint-source-proposal.npz',robot_qpos=q);(O/'all-new-continuous-source-branches.json').write_text(json.dumps(g.plain(records),allow_nan=False,indent=2)+'\n');out=dict(schema='v2d.continuous-four-joint-source-projection.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),input_manifest_sha256=sha(c.F/'manifest.json'),root_mode_unchanged_sha256=sha(rootpath),new_analytic_controls=controls,independent_URDF_source_points=sum('actual_URDF_source_error_px'in v for v in records),unresolved_branches=unresolved,failed_actual_source_points=sourcebad,maximum_URDF_camera_closure_m=closuremax,original_four_joint_motion=motion,original_motion_bad_keys=bad,four_joint_source_and_original_motion_component_passed=not unresolved and not sourcebad and not bad,thumb_source_native_self_swept_motor_grasp_and_whole_reference_unqualified=True,current_verified_reference_and_source_authority_unchanged=True,no_native_position_or_physics=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_task_qualified=False,all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))

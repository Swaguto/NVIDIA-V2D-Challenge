"""NEW analytic camera-cone dual/rotation controls, no episode/native work."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-source-cone-dual-contract-v1';sys.path.insert(0,str(S/'compute'));start=time.monotonic()
import numpy as np
from scipy.spatial.transform import Rotation
import source_cone_dual_v1 as d
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();j=(O/'all-analytic-control-records.jsonl').open('w');checks=[]
def save(name,passed,**data):
 row=dict(name=name,passed=bool(passed),**data);g.journal(j,row);checks.append(row);assert passed,name
K=np.eye(3);p=np.zeros(2);dual=d.tangent_dual(K,p,1.,np.array([1.,0.,1.]));save('canonical_lorentz_tangent',np.allclose(dual['camera_normal'],np.array([1.,0.,-1.])/np.sqrt(2),atol=1e-15),dual=dual)
K=np.array([[731.,0.,645.],[0.,697.,381.],[0.,0.,1.]]);p=np.array([711.,304.]);r=23.7;theta=.73;u=np.array([np.cos(theta),np.sin(theta)]);C=np.r_[(p+r*u-K[:2,2])/np.diag(K)[:2],1.]*.37;dual=d.tangent_dual(K,p,r,C);save('anisotropic_off_axis_tangent',abs(dual['boundary_residual'])<1e-14,dual=dual)
grid=[]
for depth in [.011,.17,1.,3.]:
 for frac in [0.,.2,1.]:
  for a in np.linspace(-np.pi,np.pi,17):
   pixel=p+frac*r*np.array([np.cos(a),np.sin(a)]);point=np.r_[(pixel-K[:2,2])/np.diag(K)[:2],1.]*depth;value=float(dual['camera_normal']@point);grid.append(dict(camera_point=point,plane_value_m=value))
save('all_positive_depth_source_cone_points_inside_dual',max(v['plane_value_m'] for v in grid)<2e-14,points=grid)
outside=np.r_[(p+2*r*u-K[:2,2])/np.diag(K)[:2],1.]*.37;save('tangent_outward_point_excluded',dual['camera_normal']@outside>0,camera_point=outside,plane_value_m=dual['camera_normal']@outside)
H=np.diag([1.,1.,.5]);lift=d.lift_ellipse_support(np.eye(3),np.zeros(2),.1,H,np.zeros(2),np.eye(2)*.05,np.array([.6,.8]));save('ellipse_support_lifts_to_global_cone_tangent',lift['pullback_direction_residual']<1e-14 and abs(lift['pullback_constant_residual'])<1e-14,lift=lift)
errors=[]
for args in [(np.eye(3),np.zeros(2),-1.,[1.,0.,1.]),(np.eye(3),np.zeros(2),1.,[0.,0.,1.]),(np.eye(3),np.zeros(2),1.,[1.,0.,-1.])]:
 try:d.tangent_dual(*args);errors.append(False)
 except ValueError:errors.append(True)
save('invalid_radius_nonboundary_and_negative_depth_fail_closed',all(errors),failures_caught=errors)
rng=np.random.default_rng(20261019);records=[]
for i in range(256):
 v=rng.normal(size=3);n=rng.normal(size=3);axis=rng.normal(size=3);axis/=np.linalg.norm(axis);a=float(rng.uniform(0,np.pi));diff=float(abs(n@(Rotation.from_rotvec(axis*a).apply(v)-v)));bound=2*np.linalg.norm(n)*np.linalg.norm(v)*np.sin(a/2);records.append(dict(v=v,n=n,axis=axis,angle_rad=a,actual_support_change=diff,upper_bound=bound,passed=diff<=bound+1e-13))
save('rigid_rotation_displacement_support_bound',all(v['passed'] for v in records),records=records)
answer=d.necessary_rotation_angle(1.,1.,1.,0.);save('necessary_angle_has_known_sixty_degree_case',abs(answer['angle_rad']-np.pi/3)<1e-15,result=answer)
j.close();out=dict(schema='v2d.source-cone-dual-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/source_cone_dual_v1.py'),analytic_controls=len(checks),all_controls_passed=all(v['passed'] for v in checks),no_episode_source_native_or_old_controls_repeated=True,no_physics_GPU_PPO=True,all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(out))

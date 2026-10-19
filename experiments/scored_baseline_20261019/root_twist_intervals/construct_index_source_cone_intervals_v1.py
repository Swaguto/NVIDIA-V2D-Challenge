"""NEW actual planar-index/source-cone intervals at saved114 root hypotheses."""
from pathlib import Path
import datetime,hashlib,itertools,json,math,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');O=S/'e037-index-source-cone-intervals-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();sys.path.insert(0,str(S/'compute'))
import source_orientation_context_v1 as c
np=c.np
from scipy.spatial.transform import Rotation
def intervals(C0,Cc,Cs,K,pixel,cap,lo,hi):
 assert -np.pi<lo<=hi<np.pi
 B=np.array([[K[0,0],0,K[0,2]-pixel[0]],[0,K[1,1],K[1,2]-pixel[1]]]);M=B.T@B-cap*cap*np.diag([0.,0.,1.]);f0=float(C0@M@C0+.5*(Cc@M@Cc+Cs@M@Cs));fc=float(2*C0@M@Cc);fs=float(2*C0@M@Cs);fc2=float(.5*(Cc@M@Cc-Cs@M@Cs));fs2=float(Cc@M@Cs);polynomial=np.array([f0+fc+fc2,2*fs+4*fs2,2*f0-6*fc2,2*fs-4*fs2,f0-fc+fc2]);scale=max(float(abs(polynomial).max()),1.);trim=polynomial.copy()
 while len(trim)>1 and abs(trim[-1])<1e-14*scale:trim=trim[:-1]
 roots=np.polynomial.polynomial.polyroots(trim) if len(trim)>1 else np.array([],complex);angles=[]
 for root in roots:
  if abs(root.imag)<1e-7:
   a=float(2*np.arctan(root.real))
   if lo<a<hi and not any(abs(a-v)<1e-9 for v in angles):angles.append(a)
 breaks=sorted([float(lo),float(hi),*angles]);records=[]
 def value(a):return f0+fc*np.cos(a)+fs*np.sin(a)+fc2*np.cos(2*a)+fs2*np.sin(2*a)
 for left,right in zip(breaks,breaks[1:]):
  middle=(left+right)/2;v=float(value(middle));kind='inside_source_cone' if v<-1e-10*scale else ('outside_source_cone' if v>1e-10*scale else 'unresolved_numeric_boundary');records.append(dict(interval=[left,right],classification=kind,midpoint_inequality=v))
 return dict(trigonometric_coefficients=[f0,fc,fs,fc2,fs2],tan_half_angle_polynomial=polynomial.tolist(),all_complex_roots=[[float(x.real),float(x.imag)] for x in roots],intervals=records,numerical_roots_not_formal_global_proof=True)
K0=np.diag([100.,100.,1.]);C0=np.array([0.,0.,1.]);Cc=np.array([.1,0.,0.]);Cs=np.array([0.,.1,0.]);controls=[]
inside=intervals(C0,Cc,Cs,K0,np.zeros(2),20.,-.2,.3);assert all(x['classification']=='inside_source_cone' for x in inside['intervals']);controls.append('full_circle_inside_source_disk')
outside=intervals(C0,Cc,Cs,K0,np.zeros(2),5.,-.2,.3);assert all(x['classification']=='outside_source_cone' for x in outside['intervals']);controls.append('circle_misses_source_disk')
cross=intervals(np.array([.1,0,1.]),Cc,Cs,K0,np.zeros(2),10.,-2.5,2.5);accepted=[r['interval'] for r in cross['intervals'] if r['classification']=='inside_source_cone'];np.testing.assert_allclose(accepted,[[-2.5,-2*np.pi/3],[2*np.pi/3,2.5]],atol=1e-10);controls.append('analytic_two_arc_intersection')
for a in np.linspace(-2.5,2.5,53):
 C=np.array([.1,0,1.])+Cc*np.cos(a)+Cs*np.sin(a);direct=(100*C[0])**2+(100*C[1])**2-100*C[2]**2;coef=cross['trigonometric_coefficients'];trig=coef[0]+coef[1]*np.cos(a)+coef[2]*np.sin(a)+coef[3]*np.cos(2*a)+coef[4]*np.sin(2*a);x=np.tan(a/2);rational=np.polynomial.polynomial.polyval(x,cross['tan_half_angle_polynomial'])/(1+x*x)**2;assert abs(direct-trig)<1e-10 and abs(trig-rational)<1e-10
controls.append('independent_trigonometric_polynomial_projection_closure')
def planar(point,lo,hi):
 x,y=point;co=(x*x+y*y-.0458**2-.035**2)/(2*.0458*.035)
 if abs(co)>1+1e-12:return []
 result=[]
 for b in [math.acos(np.clip(co,-1,1)),-math.acos(np.clip(co,-1,1))]:
  a=math.atan2(y,x)-math.atan2(.035*math.sin(b),.0458+.035*math.cos(b))
  for aa,bb in itertools.product([a-2*np.pi,a,a+2*np.pi],[b-2*np.pi,b,b+2*np.pi]):
   p=np.array([aa,bb])
   if (p>=lo-1e-9).all() and (p<=hi+1e-9).all() and not any(abs(p-z).max()<1e-10 for z in result):result.append(p)
 return result
P=S/'e037-root-twist-cone-boundary-v4';report=json.loads((P/'report.json').read_text());assert report['accepted_finite_witnesses']==0 and sha(P/'all-proposals.jsonl')==report['all_proposals_sha256'];roots=[json.loads(line) for line in (P/'all-proposals.jsonl').read_text().splitlines()];roots=[r for r in roots if r.get('rejection')=='actual bounded planar finger inverse'];assert len(roots)==38;case=report['cases'][0];frame=114;original=c.q.copy();base=original[frame,1].copy();preferred=np.array(case['preferred_command']);before=np.array(case['source_image_caps_px']);lo,hi=c.bounds(frame);camera=c.cameras[frame];A=camera[:3,:3].T@c.stage.T;offset=camera[:3,:3].T@camera[:3,3];origin=A.T@offset;observed=c.uv((c.human[frame,1,[4,8,12]]-camera[:3,3])@camera[:3,:3]);caps=before[1:]-np.minimum(.0001,.01*before[1:]);all_intervals=[];accepted=[];seen=set();attempts=0
with (O/'all-new-states-and-rejections.jsonl').open('w') as journal:
 for parent_index,parent in enumerate(roots):
  if time.monotonic()-start>210:raise TimeoutError('Index source-cone interval cap')
  rootR=Rotation.from_quat(parent['proposed_root_quaternion_xyzw']).as_matrix();z=-.0285;desired=preferred[9:11];a,b=desired;point=np.array([.0777+.0458*np.cos(a)+.035*np.cos(a+b),-.0016+.0458*np.sin(a)+.035*np.sin(a+b),z]);C=A@(base[:3]+rootR@point)-offset;pixel=c.uv(C[None])[0];difference=pixel-observed[2];length=np.linalg.norm(difference)
  if length>caps[2]:pixel=observed[2]+difference*(caps[2]/length)
  ray=np.r_[(pixel-[c.K[0,2],c.K[1,2]])/[c.K[0,0],c.K[1,1]],1.];local_origin=rootR.T@(origin-base[:3]);local_ray=rootR.T@(A.T@ray);depth=(z-local_origin[2])/local_ray[2];local=local_origin+depth*local_ray;middle=planar(local[:2]-[.0777,-.0016],lo[2:4],hi[2:4]) if depth>.01 else []
  if not middle:journal.write(json.dumps(dict(parent_root_index=parent_index,passed=False,rejection='middle inverse remains empty'))+'\n');continue
  levels=sorted(set(float(np.clip(v,lo[1],hi[1])) for v in [preferred[8],preferred[8]-.05,preferred[8]+.05,lo[1],hi[1]]))
  for distal in levels:
   vx=.0458+.035*np.cos(distal);vy=.035*np.sin(distal);center=A@(base[:3]+rootR@np.array([.0777,-.0016,.0285]))-offset;cosine=A@rootR@np.array([vx,vy,0.]);sine=A@rootR@np.array([-vy,vx,0.]);domain=intervals(center,cosine,sine,c.K,observed[1],caps[1],lo[0],hi[0]);all_intervals.append(dict(parent_root_index=parent_index,root_quaternion_xyzw=parent['proposed_root_quaternion_xyzw'],distal_joint_rad=distal,original_proximal_bounds_rad=[float(lo[0]),float(hi[0])],camera_point_constant=center.tolist(),camera_point_cosine=cosine.tolist(),camera_point_sine=sine.tolist(),domain=domain))
   for box in domain['intervals']:
    if box['classification']=='outside_source_cone':continue
    left,right=box['interval'];values=sorted(set([float(np.clip(preferred[7],left,right)),(left+right)/2]))
    for proximal,mid in itertools.product(values,middle):
     command=preferred.copy();command[:3]=base[:3];command[3:7]=parent['proposed_root_quaternion_xyzw'];command[7:9]=[proximal,distal];command[9:11]=mid;key=tuple(np.round(np.r_[command[3:7],command[7:]],11));entry=dict(parent_root_index=parent_index,command=command.tolist(),source_cone_interval=box)
     if key in seen:entry.update(passed=False,rejection='duplicate new point');journal.write(json.dumps(entry)+'\n');continue
     seen.add(key);c.q=original.copy();c.q[frame,1]=command;v=c.evaluate(frame,command[7:]);attempts+=1;self_depth=max(0.,float(-v['gaps'].min()));target=[]
     for ct in c.data.contact[:c.data.ncon]:
      a,b=int(c.model.geom_bodyid[ct.geom1]),int(c.model.geom_bodyid[ct.geom2]);ra=c.bn[a].startswith('dex3_right_');rb=c.bn[b].startswith('dex3_right_')
      if ra!=rb and not (c.bn[b] if ra else c.bn[a]).startswith('dex3_'):target.append(dict(bodies=[c.bn[a],c.bn[b]],signed_gap_m=float(ct.dist)))
     target_depth=max([max(0.,-r['signed_gap_m']) for r in target],default=0.);source_ok=bool((v['errors']<=before+1e-6).all());position_ok=bool((command[7:]>=lo-1e-8).all() and (command[7:]<=hi+1e-8).all());passed=bool(source_ok and position_ok and max(self_depth,target_depth)<=.001);entry.update(source_before_after_px=[before.tolist(),v['errors'].tolist()],source_nonregression=source_ok,original_actual_joint_bounds_passed=position_ok,native_self_penetration_m=self_depth,native_self_contacts=v['self_contacts'],native_external_contacts=target,native_external_penetration_m=target_depth,passed=passed,rejection=None if passed else 'original source or actual URDF/native gate');journal.write(json.dumps(entry)+'\n')
     if passed:accepted.append(entry)
c.q=original;ip=O/'all-index-source-cone-intervals.json';ip.write_text(json.dumps(all_intervals,indent=2)+'\n');ap=O/'accepted-finite-witnesses.json';ap.write_text(json.dumps(accepted,indent=2)+'\n');out=dict(schema='v2d.index-source-cone-intervals.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),input_manifest_sha256=sha(c.F/'manifest.json'),native_model_sha256=sha(c.MP),saved_parent_report_sha256=sha(P/'report.json'),saved_parent_proposals_sha256=sha(P/'all-proposals.jsonl'),analytical_controls_passed=controls,frame=114,parent_roots=38,index_distal_levels_per_parent=5,all_intervals_sha256=sha(ip),all_new_states_sha256=sha(O/'all-new-states-and-rejections.jsonl'),accepted_witnesses_sha256=sha(ap),new_native_states=attempts,accepted_finite_witnesses=len(accepted),actual_original_image_inequality_not_nearest_pixel_heuristic=True,numerical_finger_intervals_and_distal_samples_not_global_reachability=True,all_rejections_roots_intervals_and_new_states_saved=True,source_flags_original_caps_and_current_reference_unchanged=True,source_and_native_old_jobs_or_generic_optimizer_not_repeated=True,no_native_compile_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_physical_task_qualified=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

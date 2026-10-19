"""Independent continuous support-plane checks for saved conic exclusions.

For a strictly positive distal branch away from pi there is no interior
stationary point of a nonzero linear support objective. Its exact numerical
extrema occur on the four circular joint-boundary arcs. No quartic is solved.
"""
from pathlib import Path
import datetime,hashlib,json,math,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-temporal-index-separators-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import thumb_conic_geometry_v1 as g
import source_orientation_context_v1 as c
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.kinematics import URDFKinematics
np=c.np;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();journal=(O/'all-controls-separators-URDF-records.jsonl').open('w');controls=[]
def point(a,b):return np.array([.0458*np.cos(a)+.035*np.cos(a+b),.0458*np.sin(a)+.035*np.sin(a+b)])
def support(n,ab,bb):
 n=np.asarray(n,dtype=float);phi=float(np.arctan2(n[1],n[0]));a0,a1=ab;b0,b1=bb;assert 0<b0<=b1<np.pi and np.linalg.norm(n)>0;candidates=[]
 def add(a,b,kind):candidates.append(dict(a=float(a),b=float(b),kind=kind,point=point(a,b),projection=float(n@point(a,b))))
 for a in [a0,a1]:
  for b in [b0,b1]:add(a,b,'corner')
 for b in [b0,b1]:
  shift=float(np.arctan2(.035*np.sin(b),.0458+.035*np.cos(b)))
  for k in range(-4,5):
   a=phi-shift+k*np.pi
   if a0<=a<=a1:add(a,b,'distal_boundary_stationary')
 for a in [a0,a1]:
  for k in range(-4,5):
   b=phi-a+k*np.pi
   if b0<=b<=b1:add(a,b,'proximal_boundary_stationary')
 return min(candidates,key=lambda v:v['projection']),candidates
def check(name,passed,**extra):
 record=dict(kind='new_support_extremum_control',name=name,passed=bool(passed),**extra);g.journal(journal,record);controls.append(record);assert passed,name
v,allv=support([1.,0.],[.1,.2],[1.,1.1]);check('monotone_x_exact_far_corner',abs(v['projection']-point(.2,1.1)[0])<1e-14)
v,allv=support([0.,1.],[.1,.2],[1.,1.1]);check('monotone_y_exact_near_corner',abs(v['projection']-point(.1,1.)[1])<1e-14)
b=1.05;shift=np.arctan2(.035*np.sin(b),.0458+.035*np.cos(b));phi=.07+shift-np.pi;radius=np.linalg.norm(point(0.,b));v,allv=support([np.cos(phi),np.sin(phi)],[-.3,.3],[b,b]);check('noncorner_boundary_stationary_minimum',abs(v['projection']+radius)<1e-14 and abs(v['a']-.07)<1e-12,returned=v)
P=S/'e037-temporal-index-conic-domains-v1';rep=json.loads((P/'report.json').read_text());path=P/'all-actual-index-conic-domains.json';assert sha(path)==rep['all_outputs_sha256'][path.name];domains=json.loads(path.read_text());assert len(domains)==173 and all(d['classification']=='empty_numerical_positive_branch' for d in domains);first=json.loads((P/'all-controls-domains-and-witnesses.jsonl').read_text().splitlines()[4]);assert first['kind']=='conservative_index_source_outer_conic';Q=np.array(first['Q']);ellipse=first['ellipse'];center=np.array(ellipse['center']);factor=np.array(ellipse['factor']);A=Q[:2,:2];selected=json.loads((S/'e037-temporal-box-source-cones-v1/selected-source-temporal-box.json').read_text());base=np.array(selected['root_control']['command']);frame=114;rootR=Rotation.from_quat(base[3:7]).as_matrix();kin=URDFKinematics(c.F/'robot/urdfs/g1/dex3_right.urdf');camera=c.cameras[frame];human=(c.human[frame,1,[8]]-camera[:3,3])@camera[:3,:3];observed=c.uv(human)[0];rng=np.random.default_rng(72104);records=[];all_verified=True;minimum_margin=math.inf;max_model_closure=0.;native_binding_guard_m=1e-8
for domain in domains:
 box=np.array(domain['necessary_outer_index_box_rad']);directions=[]
 for a in [box[0,0],box[0].mean(),box[0,1]]:
  for b in [box[1,0],box[1].mean(),box[1,1]]:
   n=A@(point(a,b)-center)
   if np.linalg.norm(n)>1e-15:n/=np.linalg.norm(n);directions.append(n)
 results=[]
 for n in directions:
  minimum,candidates=support(n,box[0],box[1]);ellipse_support=float(n@center+np.linalg.norm(factor.T@n));margin=minimum['projection']-ellipse_support-native_binding_guard_m;results.append(dict(direction=n,workspace_minimum=minimum,all_boundary_critical_points=candidates,ellipse_maximum_support=ellipse_support,metric_binding_outward_guard_m=native_binding_guard_m,separation_margin_m=margin))
 best=max(results,key=lambda r:r['separation_margin_m']);separated=best['separation_margin_m']>1e-12;minimum_margin=min(minimum_margin,best['separation_margin_m']);all_verified=all_verified and separated;sample_records=[]
 for sample in range(8):
  angles=rng.uniform(box[:,0],box[:,1]);command=base.copy();command[7:9]=angles;poses=kin.poses(command[:7],dict(zip(c.manifest['joint_names']['right'],command[7:],strict=True)));native_point=poses['right_index_tip'][:3,3];local=rootR.T@(native_point-command[:3])-np.array([.0777,-.0016,.0285]);closure=float(abs(local[:2]-point(*angles)).max());max_model_closure=max(max_model_closure,closure);C=(native_point@c.stage-camera[:3,3])@camera[:3,:3];pixel=c.uv(C[None])[0];projection=float(best['direction']@local[:2]);passed=bool(closure<native_binding_guard_m and projection>=best['workspace_minimum']['projection']-native_binding_guard_m);sample=dict(command=command,index_local_point_from_URDF=local,model_closure_m=closure,URDF_source_image_error_px=float(np.linalg.norm(pixel-observed)),projection=projection,passed=passed);sample_records.append(sample);assert passed
 record=dict(kind='independent_continuous_index_separator',correlated_box_index=domain['correlated_box_index'],neighbor113_index=domain['neighbor113_index'],neighbor115_index=domain['neighbor115_index'],outer_index_box_rad=box,plane=best,other_tested_directions=results,independent_URDF_samples=sample_records,numerically_separated=separated,conditional_root_and_finite_neighbors_only=True);records.append(record);g.journal(journal,record)
journal.close();(O/'all-independent-separator-certificates.json').write_text(json.dumps(g.plain(records),allow_nan=False,indent=2)+'\n');out=dict(schema='v2d.temporal-index-separators.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),input_manifest_sha256=sha(c.F/'manifest.json'),saved_domain_records_sha256=sha(path),analytic_controls=controls,independent_domains=len(records),all_domains_numerically_separated=all_verified,minimum_recorded_plane_separation_m=minimum_margin,independent_actual_URDF_samples=8*len(records),maximum_URDF_planar_model_closure_m=max_model_closure,conservative_binding_guard_m=native_binding_guard_m,quartics_not_resolved_or_source_fits_repeated=True,numerical_support_planes_not_formal_directed_rounding_or_global_robot_proof=True,conditional_root_and_finite_neighbor_set_only=True,current_verified_reference_and_source_authority_unchanged=True,no_native_or_physics_steps=True,new_GPU_jobs=0,new_PPO_steps=0,whole_reference_or_task_qualified=False,all_outputs_sha256={f.name:sha(f) for f in O.iterdir() if f.is_file() and f.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out),indent=2))

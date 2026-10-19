"""NEW object-local clipped native-surface patches and contact-mode proposals.

Source grip region is inferred, not an observed contact label. Native geometry,
friction, condim, mass, inertia and commands are not modified.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/dex3_training'))
import mujoco,numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from v2d_dex3_flashchord.contact_patches import ObjectContactPatch,RobotContactMode
from v2d_dex3_flashchord.contact_load_feasibility import ContactGeometry,outer_contact_load_feasibility
from v2d_dex3_flashchord.kinematics import URDFKinematics
O=S/'e037-object-local-contact-patches-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();plan_path=S/'compute/object-local-contact-patches-CPU-plan-v1.json';plan=json.loads(plan_path.read_text());P=S/'e037-motion-constrained-wrist-native-v1';f=P/'frozen';manifest=json.loads((f/'manifest.json').read_text())
for name,h in manifest['files'].items():assert sha(f/name)==h
mp=P/'native-model.mjb';model_hash=sha(mp);model=mujoco.MjModel.from_binary_path(str(mp));names=[model.body(i).name for i in range(model.nbody)];bid=next(i for i,n in enumerate(names) if n.endswith('wooden_spoon_object'))
with np.load(f/'targets.npz') as z:obj=z['object_poses'][50,0].copy();root=z['robot_qpos'][50,1].copy();time_s=float(z['timestamps_s'][50])
objR=Rotation.from_quat(obj[3:]).as_matrix();T=np.eye(4);T[:3,:3]=objR;T[:3,3]=obj[:3]
site_path=S/'e037-source-patch-wrench-sites-v1/all-sampled-potential-sites.json';source_sites=json.loads(site_path.read_text());world=np.array([s['point'] for label in ['multiview_fit_unqualified','hard_skin_fit_failed'] for pool in source_sites[label].values() for s in pool]);centers=np.einsum('ni,ij->nj',world-obj[:3],objR,optimize=False);lo=centers.min(0)-plan['source_region_padding_component_m'];hi=centers.max(0)+plan['source_region_padding_component_m']
parts=[];gids=np.flatnonzero(model.geom_bodyid==bid)
for g in gids:
 assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and model.geom_condim[g]==3
 mid=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);raw=np.asarray(model.mesh_vert[a:a+n],float);G=Rotation.from_quat(model.geom_quat[g][[1,2,3,0]]).as_matrix();v=np.einsum('ni,ji->nj',raw,G,optimize=False)+model.geom_pos[g];h=ConvexHull(v);parts.append(dict(vertices=v,triangles=v[h.simplices],normals=h.equations[:,:3],planes=h.equations))
def clip(poly,n,b,inside=True):
 if len(poly)<3:return np.empty((0,3))
 d=np.einsum('ni,i->n',poly,n,optimize=False)+b
 if not inside:d=-d
 output=[]
 for k in range(len(poly)):
  a=poly[k-1];c=poly[k];da=d[k-1];dc=d[k];ia=da<=0;ic=dc<=0
  if ia!=ic:output.append(a+(c-a)*(da/(da-dc)))
  if ic:output.append(c)
 return np.asarray(output).reshape(-1,3)
def subtract_interior(poly,planes):
 # Each emitted convex piece lies outside at least one halfspace of this hull.
 remaining=poly;outside=[]
 for eq in planes:
  d=np.einsum('ni,i->n',remaining,eq[:3],optimize=False)+eq[3]+1e-9
  if (d>=0).all():outside.append(remaining);return outside
  if (d<=0).all():continue
  exterior=clip(remaining,eq[:3],eq[3]+1e-9,inside=False)
  if len(exterior)>=3:outside.append(exterior)
  remaining=clip(remaining,eq[:3],eq[3]+1e-9)
  if len(remaining)<3:return outside
 return outside
banks={};original_faces=0;clipped_faces=0
for pi,part in enumerate(parts):
 for triangle,normal in zip(part['triangles'],part['normals'],strict=True):
  original_faces+=1;poly=triangle.copy()
  for axis in range(3):
   e=np.eye(3)[axis];poly=clip(poly,e,-hi[axis]);poly=clip(poly,-e,lo[axis])
   if len(poly)<3:break
  if len(poly)<3:continue
  polygons=[poly]
  for other_index,other in enumerate(parts):
   if other_index==pi:continue
   polygons=[piece for poly in polygons for piece in subtract_interior(poly,other['planes'])]
   if not polygons:break
  axis=int(np.argmax(abs(normal)));key=('xyz'[axis]+'_'+('plus' if normal[axis]>=0 else 'minus'));bank=banks.setdefault(key,dict(triangles=[],normals=[],part_ids=[]))
  for poly in polygons:
   for j in range(1,len(poly)-1):
    t=np.stack([poly[0],poly[j],poly[j+1]])
    if np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))<1e-14:continue
    assert (t>=lo-1e-10).all() and (t<=hi+1e-10).all()
    for oi,other in enumerate(parts):
     if oi==pi:continue
     # Existence of one plane separating every triangle vertex proves the entire
     # convex triangle stays outside the other hull, up to numeric tolerance.
     signed=np.einsum('ni,ki->nk',t,other['planes'][:,:3],optimize=False)+other['planes'][:,3];assert signed.min(0).max()>=-2e-9
    bank['triangles'].append(t);bank['normals'].append(normal);bank['part_ids'].append(pi);clipped_faces+=1
patches={};arrays={};samples={}
for key,b in sorted(banks.items()):
 if not b['triangles']:continue
 tri=np.asarray(b['triangles']);normal=np.asarray(b['normals']);patch=ObjectContactPatch(key,'wooden_spoon',model_hash,tri,normal);patches[key]=patch;arrays[key+'_triangles']=tri;arrays[key+'_normals']=normal;arrays[key+'_part_ids']=np.asarray(b['part_ids'])
 c=tri.mean(1);chosen=[int(np.argmin(np.linalg.norm(c-centers.mean(0),axis=1)))]
 while len(chosen)<min(plan['potential_samples_per_patch'],len(c)):
  dist=np.linalg.norm(c[:,None]-c[chosen][None],axis=2).min(1);dist[chosen]=-1;k=int(dist.argmax())
  if dist[k]<1e-6:break
  chosen.append(k)
 samples[key]=dict(points_local=c[chosen],normals_local=normal[chosen])
np.savez_compressed(O/'object-local-patches.npz',**arrays,object_to_world=T,source_region_min_local=lo,source_region_max_local=hi)
meta_path=S/'compute/palm-wrench-library-compatibility-inputs-v1.json';meta=json.loads(meta_path.read_text());loads_report=json.loads(Path(meta['source_report']).read_text());loads={k:loads_report['rows'][0][field] for k,field in [('gravity','gravity_load_wrench_world'),('estimated_motion','estimated_reference_motion_load_wrench_world')]};friction=np.asarray(next(c for row in loads_report['rows'] for c in row['contacts'])['friction']);com=np.asarray(meta['object_COM_native_world_m'])
def frame(n):
 n=-n;t=np.cross(n,np.eye(3)[np.argmin(abs(n))]);t/=np.linalg.norm(t);return np.stack([n,t,np.cross(n,t)])
pairs=[]
for axis in 'xyz':
 a,b=axis+'_plus',axis+'_minus'
 if a not in patches or b not in patches:continue
 contacts=[]
 for key in [a,b]:
  for p,n in zip(samples[key]['points_local'],samples[key]['normals_local'],strict=True):
   point=np.einsum('ij,j->i',objR,p,optimize=False)+obj[:3];normal=np.einsum('ij,j->i',objR,n,optimize=False);contacts.append(ContactGeometry(point,frame(normal),3,friction))
 results={k:outer_contact_load_feasibility(contacts,com,wrench) for k,wrench in loads.items()};feasible=all(r['feasible_in_outer_model'] for r in results.values());pairs.append(dict(patches=[a,b],potential_contact_count=len(contacts),load_results=results,both_loads_outer_feasible=feasible,worst_total_normal_force_N=max(r['minimum_total_normal_force_N'] for r in results.values()) if feasible else None,contacts_are_potential_not_active=True))
kin=URDFKinematics(f/'robot/urdfs/g1/dex3_right.urdf');poses=kin.poses(root[:7],dict(zip(manifest['joint_names']['right'],root[7:],strict=True)));mode_rows=[]
for label,digits in [('thumb_index',['thumb','index']),('thumb_middle',['thumb','middle']),('three_digit',['thumb','index','middle'])]:
 options=[]
 for row in pairs:
  if not row['both_loads_outer_feasible']:continue
  for thumb,other in [row['patches'],row['patches'][::-1]]:
   assignment=tuple((digit,thumb if digit=='thumb' else other) for digit in digits);cost=0
   for digit,key in assignment:cost+=float(patches[key].closest_world(poses[f'right_{digit}_tip'][:3,3][None],T).distances_m[0]**2)
   options.append((row['worst_total_normal_force_N'],cost,assignment))
 selected=min(options) if options else None
 if selected:
  mode=RobotContactMode(label,'right','wooden_spoon',model_hash,selected[2]);mode.validate_patch_bindings(patches);bindings=list(mode.digit_patches)
 else:bindings=[]
 mode_rows.append(dict(mode_id=label,required_digits=digits,digit_patches=bindings,status='hypothesis_ready_for_new_geometry_solve' if selected else 'rejected_no_potential_wrench_pair',potential_worst_total_normal_force_N=selected[0] if selected else None,initial_geometric_squared_error_m2=selected[1] if selected else None,mode_is_not_source_contact_truth=True))
report=dict(schema='v2d.object-local-contact-patch-bank.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),input_sha256={str(p):sha(p) for p in [mp,f/'manifest.json',f/'targets.npz',site_path,meta_path]},patch_file_sha256=sha(O/'object-local-patches.npz'),geometry_sha256=model_hash,object_id='wooden_spoon',frame=50,timestamp_s=time_s,source_window_is_inferred_union_not_observed_contact=True,source_region_min_local_m=lo.tolist(),source_region_max_local_m=hi.tolist(),native_geometry_friction_condim_mass_inertia_not_changed=True,original_native_convex_faces=original_faces,external_clipped_patch_faces=clipped_faces,patches={key:dict(triangles=len(p.triangles_local_m),potential_samples=len(samples[key]['points_local'])) for key,p in patches.items()},potential_pair_screen=pairs,contact_mode_hypotheses=mode_rows,source_flags_and_31of32_heldout_failure_unchanged=True,source_pose_winner_not_selected=True,root_anchor='unchanged current wristv3, not rejected source/MCP centers',required_root_translation_component_bound_m=.008,required_root_rotation_component_bound_rad=.08,required_full_native_reference_contact_gate_m=.001,source_original_motion_swept_native_motor_grasp_gates_required=True,physics_stepped=False,new_GPU_work=False,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['patches','potential_pair_screen','contact_mode_hypotheses','wall_seconds']},indent=2),flush=True)

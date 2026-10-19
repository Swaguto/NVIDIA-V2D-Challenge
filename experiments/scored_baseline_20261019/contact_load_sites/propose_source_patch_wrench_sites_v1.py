"""NEW source-conditioned site proposals, screened for the required object wrench.

No inference, source fit, native forward/compile, pose optimization or command
mutation. Finite potential sites are not native contacts or source contact truth.
"""
from pathlib import Path
import datetime,hashlib,itertools,json,sys,time,xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0,str(R/'scripts/dex3_training'))
import mujoco,numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from trimesh.triangles import closest_point
from v2d_dex3_flashchord.contact_load_feasibility import ContactGeometry,outer_contact_load_feasibility
O=S/'e037-source-patch-wrench-sites-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan_path=S/'compute/source-patch-wrench-sites-CPU-plan-v1.json';plan=json.loads(plan_path.read_text())
P=S/'e037-motion-constrained-wrist-native-v1';frozen=P/'frozen'
manifest=json.loads((frozen/'manifest.json').read_text())
for name,h in manifest['files'].items():assert sha(frozen/name)==h,name
meta_path=S/'compute/palm-wrench-library-compatibility-inputs-v1.json';meta=json.loads(meta_path.read_text())
load_path=Path(meta['source_report']);assert sha(load_path)==meta['source_report_sha256'];load_report=json.loads(load_path.read_text())
model_path=P/'native-model.mjb';assert sha(model_path)==meta['native_model_sha256']
# Read geometry only. Deliberately do not construct MjData or call any forward routine.
model=mujoco.MjModel.from_binary_path(str(model_path));names=[model.body(i).name for i in range(model.nbody)]
bid=next(i for i,n in enumerate(names) if n=='wooden_spoon_object' or n.endswith('/wooden_spoon_object'))
with np.load(frozen/'targets.npz') as z:pose=z['object_poses'][50,0].copy();old_command=z['robot_qpos'][50,1].copy()
obj_R=Rotation.from_quat(pose[3:]).as_matrix()
np.testing.assert_allclose(obj_R@model.body_ipos[bid]+pose[:3],meta['object_COM_native_world_m'],atol=1e-10,rtol=0)
triangles=[];normals=[];planes=[];part_ids=[];object_geoms=np.flatnonzero(model.geom_bodyid==bid)
def body_mesh_vertices(g):
    assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
    mid=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid])
    return np.asarray(model.mesh_vert[a:a+n],float)@Rotation.from_quat(model.geom_quat[g][[1,2,3,0]]).as_matrix().T+model.geom_pos[g]
for part,g in enumerate(object_geoms):
    world=body_mesh_vertices(g)@obj_R.T+pose[:3];hull=ConvexHull(world)
    triangles.extend(world[hull.simplices]);normals.extend(hull.equations[:,:3]);planes.append(hull.equations);part_ids.extend([part]*len(hull.simplices))
triangles=np.asarray(triangles);normals=np.asarray(normals)
# Use the exact saved native pair-friction hypothesis, not an invented budget.
recorded=[c for row in load_report['rows'] for c in row['contacts']]
assert recorded and all(c['dim']==3 for c in recorded)
friction=np.asarray(recorded[0]['friction'])
assert all(np.array_equal(c['friction'],friction) for c in recorded)
assert all(model.geom_condim[g]==3 for g in object_geoms)
source_report_path=S/'e037-source-wrist-convention-audit-v1/report.json';source_report=json.loads(source_report_path.read_text())
source_geometry_paths=[S/'e037-native-RGB-hand-source-audit-v1/unfitted-source-geometry.npz',S/'e037-native-RGB-multiview-source-fit-v1/source-hand-candidate.npz',S/'e037-native-RGB-hard-skin-coverage-fit-v2/source-hand-candidate.npz']
orientation_path=S/'e037-source-wrist-convention-audit-v1/saved-palm-orientation-diagnostics.npz'
with np.load(orientation_path) as z:native_from_C=z['native_from_C'].copy();source_rotations=z['source_robot_palm_rotations_native'].copy()
with np.load(frozen/'aligned-source-reference.npz') as z:camera=z['camera_c2w'][50].copy();flags={k:bool(z[k][50,1]) for k in ['hand_valid','hand_directly_observed','hand_metric_valid']}
assert not any(flags.values())
stage=np.asarray(json.loads((frozen/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal']);native_origin=stage@camera[:3,3]
bundle=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(bundle) as z:old_camera=z['camera_c2w'][50].copy()
# Necessary morphology bounds, not a reachability or joint/motor certificate.
urdf=frozen/'robot/urdfs/g1/dex3_right.urdf';root=ET.parse(urdf).getroot();joints=root.findall('joint')
limits={j.attrib['name']:{k:float(v) for k,v in j.find('limit').attrib.items()} for j in joints if j.attrib['type']!='fixed'}
by_child={j.find('child').attrib['link']:j for j in joints};workspace={}
for digit,last in [('thumb',2),('index',1),('middle',1)]:
    distal=f'right_hand_{digit}_{last}_link';chain=[];link=distal
    while link!='right_hand_palm_link':
        j=by_child[link];origin=j.find('origin');xyz=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3);chain.append(xyz);link=j.find('parent').attrib['link']
    chain=chain[::-1];body=next(i for i,n in enumerate(names) if n.endswith(distal));gs=np.flatnonzero(model.geom_bodyid==body)
    assert len(gs) and all(model.geom_condim[g]==3 for g in gs)
    radius=max(float(np.linalg.norm(body_mesh_vertices(g),axis=1).max()) for g in gs)
    workspace[digit]=dict(first_origin=chain[0],chain_radius=sum(np.linalg.norm(x) for x in chain[1:])+radius,distal_radius=radius)

def frame_from_normal(n):
    n=-np.asarray(n);helper=np.eye(3)[np.argmin(abs(n))];t=np.cross(n,helper);t/=np.linalg.norm(t);return np.stack([n,t,np.cross(n,t)])
def samples(source):
    points=closest_point(triangles,np.broadcast_to(source,(len(triangles),3)))
    external=np.ones(len(points),bool)
    for eq in planes:external &= ~((points@eq[:,:3].T+eq[:,3]).max(1)<-1e-7)
    dist=np.linalg.norm(points-source,axis=1);dist[~external]=np.inf;nearest=int(np.argmin(dist));assert np.isfinite(dist[nearest])
    center=points[nearest];pool=np.flatnonzero(external&(np.linalg.norm(points-center,axis=1)<=plan['source_projected_patch_radius_m']))
    # Deterministic diversity in position and normal within the predeclared patch.
    chosen=[nearest]
    while len(chosen)<min(plan['maximum_sites_per_digit'],len(pool)):
        position=np.linalg.norm(points[pool,None]-points[chosen][None],axis=2)/plan['source_projected_patch_radius_m']
        angle=1-normals[pool]@normals[chosen].T;score=(position+angle).min(1)
        score[np.isin(pool,chosen)]=-np.inf;k=int(pool[np.argmax(score)])
        if score.max()<1e-6:break
        chosen.append(k)
    return [dict(point=points[k],normal=normals[k],triangle=int(k),source_gap_m=float(dist[k]),patch_offset_m=float(np.linalg.norm(points[k]-center))) for k in chosen],dict(nearest_source_surface_gap_m=float(dist[nearest]),patch_triangles=len(pool),sampled_sites=len(chosen),projection_center_world_m=center.tolist())

loads={k:np.asarray(load_report['rows'][0][v]) for k,v in [('gravity','gravity_load_wrench_world'),('estimated_motion','estimated_reference_motion_load_wrench_world')]}
rows=[];all_site_sets={};families=['raw_RGB','multiview_fit_unqualified','hard_skin_fit_failed'];digits=['thumb','index','middle']
for i,label in enumerate(families):
    path=source_geometry_paths[i]
    with np.load(path) as z:
        hand_C=z['joints_camera_C'].copy() if i==0 else (z['joints_world'][0]-old_camera[:3,3])@old_camera[:3,:3]
    human=hand_C@native_from_C.T+native_origin;root_center=old_command[:7].copy();root_center[:3]+=source_report['orientation_rows'][i]['estimated_human_wrist_minus_robot_root_m'];root_center[3:7]=Rotation.from_matrix(source_rotations[i]).as_quat()
    np.testing.assert_allclose(root_center[:3],human[0],atol=1e-10,rtol=0)
    sites={};patch={}
    for digit,joint in zip(digits,[4,8,12],strict=True):sites[digit],patch[digit]=samples(human[joint])
    evaluated=0;opposed=0;morphology_passed=0;gravity_passed=0;both_passed=[];statuses={};pairs=[]
    for triple in itertools.product(*(sites[d] for d in digits)):
        evaluated+=1
        if time.monotonic()-start>220:raise TimeoutError('New finite contact-site proposal wall cap')
        thumb_dot=min(float(np.dot(triple[0]['normal'],s['normal'])) for s in triple[1:])
        if thumb_dot>plan['required_thumb_opposition_dot_max']:continue
        opposed+=1;lower=[]
        for digit,site in zip(digits,triple,strict=True):
            w=workspace[digit];origin=source_rotations[i]@w['first_origin']+root_center[:3]
            extra=np.sqrt(3)*plan['root_translation_component_bound_m']+2*np.sin(np.sqrt(3)*plan['root_rotation_component_bound_rad']/2)*np.linalg.norm(w['first_origin'])
            lower.append(max(0,float(np.linalg.norm(site['point']-origin)-w['chain_radius']-extra)))
        if max(lower)>plan['necessary_surface_proximity_m']:continue
        morphology_passed+=1
        contacts=[ContactGeometry(s['point'],frame_from_normal(s['normal']),3,friction) for s in triple]
        fits={k:outer_contact_load_feasibility(contacts,meta['object_COM_native_world_m'],load) for k,load in loads.items()}
        gravity_passed+=int(fits['gravity']['feasible_in_outer_model'])
        if not all(f['feasible_in_outer_model'] for f in fits.values()):continue
        score=sum(s['source_gap_m']**2 for s in triple)
        both_passed.append((score,triple,lower,fits,thumb_dot))
    both_passed.sort(key=lambda x:x[0]);best=None
    if both_passed:
        score,triple,lower,fits,dot=both_passed[0]
        best=dict(inferred_wrist_pose_seed_xyzw=root_center.tolist(),original_fingers_rad_not_solved=old_command[7:].tolist(),source_gap_squared_sum_m2=score,thumb_opposition_dot=dot,necessary_surface_distance_lower_bounds_m=lower,sites={d:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in s.items()} for d,s in zip(digits,triple,strict=True)},outer_model_load_results=fits,source_pose_qualified=False,contacts_not_realized_by_robot=True,IK_or_native_or_motor_or_source_fit_not_validated=True)
    rows.append(dict(source_label=label,source_geometry_sha256=sha(path),source_flags=flags,patches=patch,total_site_triples=evaluated,opposed_triples=opposed,necessary_morphology_passed=morphology_passed,gravity_outer_model_passed=gravity_passed,both_loads_outer_model_passed=len(both_passed),proposed_site_seed=best))
    all_site_sets[label]={d:[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in s.items()} for s in sites[d]] for d in digits}
report=dict(schema='v2d.source-conditioned-wrench-site-seeds.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),library_sha256=sha(R/'scripts/dex3_training/v2d_dex3_flashchord/contact_load_feasibility.py'),inputs_sha256={str(p):sha(p) for p in [model_path,frozen/'manifest.json',frozen/'targets.npz',frozen/'aligned-source-reference.npz',orientation_path,source_report_path,meta_path,load_path,urdf,bundle]+source_geometry_paths},frame=50,rows=rows,native_outer_box_friction_hypothesis=friction.tolist(),new_sites_are_potential_not_native_or_measured_contacts=True,source_hand_and_heldout_failures_unchanged=True,source_patch_radius_is_exploratory_not_measured_uncertainty=True,source_pose_winner_not_selected=True,morphology_is_only_necessary_triangle_inequality_bound=True,actual_URDF_limits=limits,required_full_native_reference_contact_gate_m=.001,required_full_source_motion_swept_motor_and_grasp_checks_before_GPU_or_PPO=True,source_arrays_commands_or_default_behavior_not_modified=True,no_native_compile_or_forward_or_physics_steps=True,no_new_source_or_old_optimizer_or_completed_audit_jobs=True,new_GPU_work=False,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'all-sampled-potential-sites.json').write_text(json.dumps(all_site_sets,indent=2)+'\n')
for name,h in manifest['files'].items():assert sha(frozen/name)==h
print(json.dumps({k:report[k] for k in ['schema','created_utc','rows','wall_seconds']},indent=2),flush=True)

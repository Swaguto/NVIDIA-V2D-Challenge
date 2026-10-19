"""One-knot source-oriented Dex3 hypothesis, not a grasp/trajectory certificate.

Use a frozen unfitted inferred RGB orientation and fixed reconstructed object. Search
actual seven-joint URDF ranges for distal geometry near the object's native
hulls, penalizing native collisions. No source landmark/depth enters the fit.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time,xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(R/'scripts/dex3_training'),str(R/'scripts/track3')]
import mujoco,numpy as np
from trimesh.triangles import closest_point as triangle_closest_point
from scipy.optimize import differential_evolution,minimize
from scipy.spatial import cKDTree,ConvexHull
from scipy.spatial.transform import Rotation
from audit_hand_reference_contacts import canonical_id
from v2d_dex3_flashchord.kinematics import URDFKinematics
P=S/'e037-motion-constrained-wrist-native-v1';O=S/'e037-source-oriented-opposed-grasp-v1';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();start=time.monotonic()
m=json.loads((P/'frozen/manifest.json').read_text())
for name,h in m['files'].items():assert sha(P/'frozen'/name)==h
with np.load(P/'frozen/targets.npz') as z:targets={k:z[k].copy() for k in z.files}
model=mujoco.MjModel.from_binary_path(str(P/'native-model.mjb'));data=mujoco.MjData(model)
bodies=[model.body(i).name for i in range(model.nbody)];joints=[model.joint(i).name for i in range(model.njnt)]
object_id=canonical_id(bodies,'wooden_spoon_object');object_qadr=int(model.jnt_qposadr[model.body_jntadr[object_id]])
frame=50;root=ET.parse(P/'frozen/robot/urdfs/g1/dex3_right.urdf').getroot()
limits={j.attrib['name']:j.find('limit').attrib for j in root.findall('joint') if j.attrib['type']!='fixed'}
bounds=[(float(limits[n]['lower']),float(limits[n]['upper'])) for n in m['joint_names']['right']]
velocity_limits=[float(limits[n]['velocity']) for n in m['joint_names']['right']]
binding={}
for si,side in enumerate(['left','right']):
    w=[int(model.jnt_qposadr[canonical_id(joints,f'{side}_wrist_{a}')]) for a in ['x','y','z','rotvec']]
    f=[int(model.jnt_qposadr[canonical_id(joints,n)]) for n in m['joint_names'][side]]
    binding[side]=(w,f)
data.qpos[:]=model.qpos0;pose=targets['object_poses'][frame,0]
data.qpos[object_qadr:object_qadr+3]=pose[:3];data.qpos[object_qadr+3:object_qadr+7]=pose[[6,3,4,5]]
for si,side in enumerate(['left','right']):
    w,f=binding[side];q=targets['robot_qpos'][frame,si]
    data.qpos[w[:3]]=q[:3];data.qpos[w[3]:w[3]+4]=q[[6,3,4,5]];data.qpos[f]=q[7:]
base_qpos=data.qpos.copy();right_fingers=binding['right'][1]
object_geoms=list(np.flatnonzero(model.geom_bodyid==object_id))
hand_bodies={i for i,n in enumerate(bodies) if n.startswith('dex3_right_')}
groups={digit:[g for g,b in enumerate(model.geom_bodyid) if bodies[b].endswith(f'right_hand_{digit}_{last}_link')] for digit,last in [('thumb',2),('index',1),('middle',1)]}
assert all(groups.values())
local={};faces={};local_normals={}
for g in set(object_geoms+[g for gs in groups.values() for g in gs]):
    mid=int(model.geom_dataid[g]);assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
    a=int(model.mesh_vertadr[mid]);count=int(model.mesh_vertnum[mid]);verts=np.asarray(model.mesh_vert[a:a+count],float)
    hull=ConvexHull(verts);local[g]=verts[hull.vertices]
    remap=np.full(len(verts),-1);remap[hull.vertices]=np.arange(len(hull.vertices));faces[g]=remap[hull.simplices];local_normals[g]=hull.equations[:,:3];assert (faces[g]>=0).all()
def vertices(g):return np.einsum('ni,ji->nj',local[g],data.geom_xmat[g].reshape(3,3),optimize=False)+data.geom_xpos[g]
mujoco.mj_fwdPosition(model,data)
object_vertices=np.concatenate([vertices(g) for g in object_geoms]);tree=cKDTree(object_vertices)
object_faces=[];offset=0
for g in object_geoms:
    object_faces.append(faces[g]+offset);offset+=len(local[g])
object_faces=np.concatenate(object_faces);object_triangles=object_vertices[object_faces]
object_incident=[np.flatnonzero(np.any(object_faces==i,axis=1)) for i in range(len(object_vertices))]
def surface_upper(g):
    a=vertices(g);distance,nearest=tree.query(a);chosen=np.argsort(distance)[:8]
    tids=np.concatenate([object_incident[int(nearest[i])] for i in chosen]);point_ids=np.concatenate([np.full(len(object_incident[int(nearest[i])]),i) for i in chosen])
    points=a[point_ids];closest=triangle_closest_point(object_triangles[tids],points)
    best=float(np.linalg.norm(points-closest,axis=1).min())
    # Symmetric witnesses on the same retained nearest-vertex neighborhoods.
    selected_faces=[];query=[]
    for i in chosen:
        incident=np.flatnonzero(np.any(faces[g]==i,axis=1));selected_faces.extend(incident);query.extend([object_vertices[nearest[i]]]*len(incident))
    if selected_faces:
        query=np.asarray(query);closest=triangle_closest_point(a[faces[g][selected_faces]],query)
        best=min(best,float(np.linalg.norm(query-closest,axis=1).min()))
    return best
# A witness is an upper bound, not an exact closest-distance or contact proof.

# Freeze geometric sites before this fit. No image/source point is a fit target.
witness_path=S/'e037-coupled-grasp-witnesses-v1/report.json';witness=json.loads(witness_path.read_text())
by_digit={r['digit']:r for r in witness['witnesses']}
normal_parts=[];part_ids=[];planes={}
for g in object_geoms:
    v=vertices(g);h=ConvexHull(v);planes[g]=h.equations
    normal_parts.append(np.einsum('ni,ji->nj',local_normals[g],data.geom_xmat[g].reshape(3,3),optimize=False));part_ids.extend([g]*len(faces[g]))
object_normals=np.concatenate(normal_parts);centroids=object_triangles.mean(1)
index_point=np.asarray(by_digit['index']['object_point_native_world_m']);index_normal=np.asarray(by_digit['index']['object_outward_normal_world'])
distance=np.linalg.norm(centroids-index_point,axis=1)
eligible=np.flatnonzero((np.einsum('ni,i->n',object_normals,index_normal,optimize=False)<=-.7)&(distance<=.04))
eligible=eligible[np.argsort(distance[eligible])];opposed=None
for i in eligible:
    point=centroids[i];inside=False
    for g,eq in planes.items():
        if (np.einsum('ni,i->n',eq[:,:3],point,optimize=False)+eq[:,3]).max() < -1e-7:inside=True;break
    if not inside:opposed=int(i);break
assert opposed is not None,'No external opposed site within declared40mm neighborhood'
sites={digit:dict(point=np.asarray(by_digit[digit]['object_point_native_world_m']),normal=np.asarray(by_digit[digit]['object_outward_normal_world'])) for digit in ['index','middle']}
sites['thumb']=dict(point=centroids[opposed].copy(),normal=object_normals[opposed].copy())
def site_distance(g,digit):
    normal=np.einsum('ni,ji->nj',local_normals[g],data.geom_xmat[g].reshape(3,3),optimize=False)
    selected=np.flatnonzero(np.einsum('ni,i->n',normal,sites[digit]['normal'],optimize=False)<=-.5)
    if not len(selected):return .1
    triangles=vertices(g)[faces[g][selected]];query=np.broadcast_to(sites[digit]['point'],(len(selected),3))
    closest=triangle_closest_point(triangles,query)
    return float(np.linalg.norm(closest-query,axis=1).min())
calls=0;best=None
original_control_root=targets['robot_qpos'][frame,1,:7].copy()
original_root=original_control_root.copy()
source_orientation_path=S/'e037-source-wrist-convention-audit-v1/saved-palm-orientation-diagnostics.npz'
with np.load(source_orientation_path) as z:source_orientation=z['source_robot_palm_rotations_native'][0].copy()
# Frozen primary unfitted RGB prior; not selected by physical/image outcomes.
original_root[3:7]=Rotation.from_matrix(source_orientation).as_quat()
right_wrist=binding['right'][0]
def root_command(x):
    q=original_root.copy();q[:3]+=.008*x[:3]
    q[3:7]=(Rotation.from_rotvec(.08*x[3:6])*Rotation.from_quat(original_root[3:7])).as_quat()
    return q
def measure(x):
    data.qpos[:]=base_qpos;q=root_command(x)
    data.qpos[right_wrist[:3]]=q[:3];adr=right_wrist[3];data.qpos[adr:adr+4]=q[[6,3,4,5]]
    data.qpos[right_fingers]=x[6:];mujoco.mj_fwdPosition(model,data)
    gaps={digit:float(min(site_distance(g,digit) for g in gs)) for digit,gs in groups.items()}
    contacts=[]
    for c in data.contact[:data.ncon]:
        a,b=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
        if a in hand_bodies or b in hand_bodies:
            contacts.append(dict(bodies=[bodies[a],bodies[b]],signed_gap_m=float(c.dist),normal_geom1_to_geom2=np.asarray(c.frame[:3]).tolist()))
    return gaps,contacts
original=targets['robot_qpos'][frame,1,7:].copy();initial_gaps,initial_contacts=measure(np.r_[np.zeros(6),original])
def objective(x):
    global calls,best
    calls+=1
    if time.monotonic()-start>220:raise TimeoutError('Bounded one-knot finger probe')
    gaps,contacts=measure(x)
    cost=sum((g/.01)**2 for g in gaps.values())
    cost+=100*sum((max(0,-.00025-c['signed_gap_m'])/.001)**2 for c in contacts)
    cost+=.0001*float(np.sum((x[6:]-original)**2))+.0001*float(np.sum(x[:6]**2))
    if best is None or cost<best[0]:best=(cost,np.asarray(x).copy())
    return cost
with np.load(S/'e037-coupled-wrist-finger-reach-v1/single-knot-finger-command.npz') as z:warm_command=z['right_command'].copy()
warm=np.r_[(warm_command[:3]-original_root[:3])/.008,(Rotation.from_quat(warm_command[3:7])*Rotation.from_quat(original_root[3:7]).inv()).as_rotvec()/.08,warm_command[7:]]
fit_history=[]
coupled_bounds=[(-1.,1.)]*6+bounds
for seed in [warm,np.r_[np.zeros(6),original]]:
    polish=minimize(objective,seed,method='Powell',bounds=coupled_bounds,options=dict(maxiter=100,maxfev=4000,xtol=1e-6,ftol=1e-9))
    fit_history.append(dict(success=bool(polish.success),message=str(polish.message),objective=float(polish.fun),evaluations=int(polish.nfev)))
x=best[1];angles=x[6:];gaps,contacts=measure(x);max_depth=max([max(0,-c['signed_gap_m']) for c in contacts]+[0])
command=targets['robot_qpos'][frame,1].copy();command[:7]=root_command(x);command[7:]=angles
# Held-out source image is evaluated after the geometry-only search.
with np.load(P/'frozen/aligned-source-reference.npz') as z:camera=z['camera_c2w'][frame].copy()
stage=np.asarray(json.loads((P/'frozen/command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz') as z:K=z['intrinsics'].copy()
image_path=S/'e037-watcloud-mask-refinement-handoff-v1/image-observations/right/000050.json';image=json.loads(image_path.read_text());keypoints=np.asarray(image['keypoints']);ids=np.array([4,8,12]);selected=keypoints[ids,2]>0
assert not image['metric_valid'] and not image['directly_observed']
kin=URDFKinematics(P/'frozen/robot/urdfs/g1/dex3_right.urdf');image_errors=[]
for q in [targets['robot_qpos'][frame,1],command]:
    poses=kin.poses(q[:7],dict(zip(m['joint_names']['right'],q[7:],strict=True)))
    points=np.stack([poses[f'right_{d}_tip'][:3,3] for d in ['thumb','index','middle']])[selected]
    world=np.einsum('ni,ij->nj',points,stage,optimize=False);C=np.einsum('ni,ij->nj',world-camera[:3,3],camera[:3,:3],optimize=False)
    assert (C[:,2]>.01).all();uv=np.c_[K[0,0]*C[:,0]/C[:,2]+K[0,2],K[1,1]*C[:,1]/C[:,2]+K[1,2]]
    image_errors.append(np.linalg.norm(uv-keypoints[ids[selected],:2],axis=1))
np.savez_compressed(O/'single-knot-finger-command.npz',frame_index=frame,right_command=command,original_right_command=targets['robot_qpos'][frame,1])
report=dict(schema='v2d.source-oriented-opposed-grasp-probe.v1',source_orientation_sha256=sha(source_orientation_path),source_orientation_kind='primary_unfitted_native_RGB_prior; inferred and unqualified',source_orientation_selected_before_fit=True,translation_of_source_model_not_used=True,root_delta_from_old_control_m=(command[:3]-original_control_root[:3]).tolist(),total_root_rotation_delta_from_old_control_rad=(Rotation.from_quat(command[3:7])*Rotation.from_quat(original_control_root[3:7]).inv()).as_rotvec().tolist(),residual_rotation_bounds_are_about_new_explicit_source_prior=True,frozen_site_selection_audit_sha256=sha(witness_path),frozen_object_sites={d:{k:v.tolist() for k,v in site.items()} for d,site in sites.items()},thumb_index_target_normal_dot=float(np.dot(sites['thumb']['normal'],sites['index']['normal'])),native_union_proximity_was_replaced_by_explicit_opposed_site_objective=True,maximum_allowed_root_component_change_m=.008,maximum_allowed_rotation_component_rad=.08,actual_root_delta_world_m=(command[:3]-original_root[:3]).tolist(),actual_rotation_delta_world_rad=(Rotation.from_quat(command[3:7])*Rotation.from_quat(original_root[3:7]).inv()).as_rotvec().tolist(),prior_wrist_motion_certificate_invalidated=True,prior_vertex_pair_control_retained_sha256=sha(S/'e037-fixed-wrist-finger-reach-v1/report.json'),prior_fixed_wrist_surface_control_sha256=sha(S/'e037-fixed-wrist-finger-reach-v2/report.json'),distance_witness='native distal triangle to frozen site, require facing normal dot<=-.5; every negative native contact checked',declared_optimizer_penetration_target_m=.00025,unchanged_native_acceptance_gate_m=.001,fit_history=fit_history,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),input_manifest_sha256=sha(P/'frozen/manifest.json'),native_model_sha256=sha(P/'native-model.mjb'),source_image_sha256=sha(image_path),frame=frame,original_joint_angles_rad=original.tolist(),proposed_joint_angles_rad=angles.tolist(),actual_URDF_joint_bounds=bounds,actual_URDF_joint_velocity_limits_rad_s=velocity_limits,within_URDF_limits=all(lo-1e-9<=x<=hi+1e-9 for x,(lo,hi) in zip(angles,bounds,strict=True)),original_distal_opposed_site_distances_m=initial_gaps,proposed_distal_opposed_site_distances_m=gaps,original_native_contacts=initial_contacts,proposed_native_contacts=contacts,maximum_native_penetration_m=max_depth,single_knot_1mm_penetration_passed=max_depth<=.001,all_distal_distance_upper_bounds_under_2mm=all(v<=.002 for v in gaps.values()),geometry_only_objective=True,source_images_or_depth_not_in_objective=True,source_authority_flags_unchanged=True,wrist_fixed=False,object_fixed=True,powell_success=all(r['success'] for r in fit_history),objective_evaluations=calls,objective=float(best[0]),inferred_landmark_ids=ids[selected].tolist(),before_image_errors_px=image_errors[0].tolist(),after_image_errors_px=image_errors[1].tolist(),source_inferred_image_nonregression=bool(np.all(image_errors[1]<=image_errors[0]+1e-6)),source_metric_truth_certified=False,whole_trajectory_or_velocity_feasibility_assessed=False,opposing_normals_or_force_closure_certified=False,source_reference_qualified=False,training_approved=False,physical_task_accepted=False,new_GPU_work=False,new_PPO_steps=0,wall_seconds=time.monotonic()-start)
for name,h in m['files'].items():assert sha(P/'frozen'/name)==h
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

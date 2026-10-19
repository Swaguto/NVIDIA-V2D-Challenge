"""One-knot inferred Dex3 reachability probe, not a grasp/trajectory certificate.

Hold the motion-constrained wrist and improved source object fixed. Search
actual seven-joint URDF ranges for distal geometry near the object's native
hulls, penalizing native collisions. No source landmark/depth enters the fit.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time,xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(R/'scripts/dex3_training'),str(R/'scripts/track3')]
import mujoco,numpy as np
from scipy.optimize import differential_evolution,minimize
from scipy.spatial import cKDTree,ConvexHull
from audit_hand_reference_contacts import canonical_id
from v2d_dex3_flashchord.kinematics import URDFKinematics
P=S/'e037-motion-constrained-wrist-native-v1';O=S/'e037-fixed-wrist-finger-reach-v1';O.mkdir(exist_ok=True)
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
local={}
for g in set(object_geoms+[g for gs in groups.values() for g in gs]):
    mid=int(model.geom_dataid[g]);assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
    a=int(model.mesh_vertadr[mid]);count=int(model.mesh_vertnum[mid]);verts=np.asarray(model.mesh_vert[a:a+count],float)
    hull=ConvexHull(verts);local[g]=verts[hull.vertices]
def vertices(g):return np.einsum('ni,ji->nj',local[g],data.geom_xmat[g].reshape(3,3),optimize=False)+data.geom_xpos[g]
mujoco.mj_fwdPosition(model,data)
object_vertices=np.concatenate([vertices(g) for g in object_geoms]);tree=cKDTree(object_vertices)
calls=0;best=None
def measure(angles):
    data.qpos[:]=base_qpos;data.qpos[right_fingers]=angles;mujoco.mj_fwdPosition(model,data)
    gaps={digit:float(min(tree.query(vertices(g))[0].min() for g in gs)) for digit,gs in groups.items()}
    contacts=[]
    for c in data.contact[:data.ncon]:
        a,b=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
        if a in hand_bodies or b in hand_bodies:
            contacts.append(dict(bodies=[bodies[a],bodies[b]],signed_gap_m=float(c.dist)))
    return gaps,contacts
original=targets['robot_qpos'][frame,1,7:].copy();initial_gaps,initial_contacts=measure(original)
def objective(angles):
    global calls,best
    calls+=1
    if time.monotonic()-start>220:raise TimeoutError('Bounded one-knot finger probe')
    gaps,contacts=measure(angles)
    cost=sum((max(0,g-.0005)/.01)**2 for g in gaps.values())
    cost+=100*sum((max(0,.0005-c['signed_gap_m'])/.001)**2 for c in contacts)
    cost+=.001*float(np.sum((angles-original)**2))
    if best is None or cost<best[0]:best=(cost,np.asarray(angles).copy())
    return cost
fit=differential_evolution(objective,bounds,seed=37,popsize=8,maxiter=60,tol=1e-7,atol=1e-9,polish=False,workers=1,updating='immediate',x0=original)
polish=minimize(objective,fit.x,method='Powell',bounds=bounds,options=dict(maxiter=100,maxfev=4000,xtol=1e-6,ftol=1e-9))
angles=best[1];gaps,contacts=measure(angles);max_depth=max([max(0,-c['signed_gap_m']) for c in contacts]+[0])
command=targets['robot_qpos'][frame,1].copy();command[7:]=angles
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
report=dict(schema='v2d.fixed-wrist-finger-reachability-probe.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),input_manifest_sha256=sha(P/'frozen/manifest.json'),native_model_sha256=sha(P/'native-model.mjb'),source_image_sha256=sha(image_path),frame=frame,original_joint_angles_rad=original.tolist(),proposed_joint_angles_rad=angles.tolist(),actual_URDF_joint_bounds=bounds,actual_URDF_joint_velocity_limits_rad_s=velocity_limits,within_URDF_limits=all(lo-1e-9<=x<=hi+1e-9 for x,(lo,hi) in zip(angles,bounds,strict=True)),original_distal_vertex_pair_upper_bounds_m=initial_gaps,proposed_distal_vertex_pair_upper_bounds_m=gaps,original_native_contacts=initial_contacts,proposed_native_contacts=contacts,maximum_native_penetration_m=max_depth,single_knot_1mm_penetration_passed=max_depth<=.001,all_distal_distance_upper_bounds_under_2mm=all(v<=.002 for v in gaps.values()),geometry_only_objective=True,source_images_or_depth_not_in_objective=True,source_authority_flags_unchanged=True,wrist_and_objects_fixed=True,differential_evolution_success=bool(fit.success),differential_evolution_message=str(fit.message),powell_success=bool(polish.success),objective_evaluations=calls,objective=float(best[0]),inferred_landmark_ids=ids[selected].tolist(),before_image_errors_px=image_errors[0].tolist(),after_image_errors_px=image_errors[1].tolist(),source_inferred_image_nonregression=bool(np.all(image_errors[1]<=image_errors[0]+1e-6)),source_metric_truth_certified=False,whole_trajectory_or_velocity_feasibility_assessed=False,opposing_normals_or_force_closure_certified=False,source_reference_qualified=False,training_approved=False,physical_task_accepted=False,new_GPU_work=False,new_PPO_steps=0,wall_seconds=time.monotonic()-start)
for name,h in m['files'].items():assert sha(P/'frozen'/name)==h
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

"""Passive comparison of retained inferred wrist poses and actual thumb reach.

The chain-length sphere is a necessary bound for one frozen proposed site,
not a grasp certificate or proof about the coupled wrist search / other sites.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time,xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0,str(R/'scripts/dex3_training'))
import mujoco,numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from audit_hand_reference_contacts import canonical_id
from v2d_dex3_flashchord.kinematics import URDFKinematics
O=S/'e037-source-wrist-position-workspace-v1';O.mkdir(exist_ok=True);P=S/'e037-motion-constrained-wrist-native-v1'
start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((P/'frozen/manifest.json').read_text())
for name,h in m['files'].items():assert sha(P/'frozen'/name)==h
with np.load(P/'frozen/targets.npz') as z:commands=z['robot_qpos'][50].copy();obj=z['object_poses'][50,0].copy()
prior_path=S/'e037-source-wrist-convention-audit-v1/report.json';prior=json.loads(prior_path.read_text())
rotation_path=S/'e037-source-wrist-convention-audit-v1/saved-palm-orientation-diagnostics.npz'
with np.load(rotation_path) as z:rotations=z['source_robot_palm_rotations_native'].copy()
site_path=S/'e037-opposed-grasp-sites-v1/report.json';site=json.loads(site_path.read_text())['frozen_object_sites']['thumb'];point=np.asarray(site['point'])
model_path=P/'native-model.mjb';model=mujoco.MjModel.from_binary_path(str(model_path));data=mujoco.MjData(model)
bodies=[model.body(i).name for i in range(model.nbody)];joints=[model.joint(i).name for i in range(model.njnt)]
object_id=canonical_id(bodies,'wooden_spoon_object');oa=int(model.jnt_qposadr[model.body_jntadr[object_id]])
kin=URDFKinematics(P/'frozen/robot/urdfs/g1/dex3_right.urdf');root=ET.parse(P/'frozen/robot/urdfs/g1/dex3_right.urdf').getroot()
chain=[]
for index in range(3):
    j=next(j for j in root.findall('joint') if j.attrib['name']==f'right_hand_thumb_{index}_joint')
    xyz=np.fromstring(j.find('origin').attrib.get('xyz','0 0 0'),sep=' ');rpy=np.fromstring(j.find('origin').attrib.get('rpy','0 0 0'),sep=' ')
    np.testing.assert_array_equal(rpy,np.zeros(3));chain.append(xyz)
thumb_geom=next(g for g,b in enumerate(model.geom_bodyid) if bodies[b].endswith('right_hand_thumb_2_link'))
def body_vertices(g):
    mid=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);verts=np.asarray(model.mesh_vert[a:a+n],float)
    rotation=Rotation.from_quat(model.geom_quat[g][[1,2,3,0]]).as_matrix()
    return np.einsum('ni,ji->nj',verts,rotation,optimize=False)+model.geom_pos[g]
radius=float(np.linalg.norm(body_vertices(thumb_geom),axis=1).max())
reach=float(np.linalg.norm(chain[1])+np.linalg.norm(chain[2])+radius)
bindings={}
for si,side in enumerate(['left','right']):
    wrists=[int(model.jnt_qposadr[canonical_id(joints,f'{side}_wrist_{a}')]) for a in ['x','y','z','rotvec']]
    fingers=[int(model.jnt_qposadr[canonical_id(joints,n)]) for n in m['joint_names'][side]]
    bindings[side]=(wrists,fingers)
local={}
for g in range(model.ngeom):
    mid=int(model.geom_dataid[g]);assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
    a=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);local[g]=np.asarray(model.mesh_vert[a:a+n],float)
def world_vertices(g):return np.einsum('ni,ji->nj',local[g],data.geom_xmat[g].reshape(3,3),optimize=False)+data.geom_xpos[g]
variants=[('current_gap_reconstruction',commands[1,:7].copy()),('unfitted_RGB_orientation_only',np.r_[commands[1,:3],Rotation.from_matrix(rotations[0]).as_quat()])]
for i,row in enumerate(prior['orientation_rows']):
    variants.append((row['source']+'_full_inferred_pose',np.r_[commands[1,:3]+row['estimated_human_wrist_minus_robot_root_m'],Rotation.from_matrix(rotations[i]).as_quat()]))
rows=[];max_fk=0.
for name,pose in variants:
    data.qpos[:]=model.qpos0;data.qpos[oa:oa+3]=obj[:3];data.qpos[oa+3:oa+7]=obj[[6,3,4,5]]
    for si,side in enumerate(['left','right']):
        command=commands[si].copy()
        if side=='right':command[:7]=pose
        w,f=bindings[side];data.qpos[w[:3]]=command[:3];data.qpos[w[3]:w[3]+4]=command[[6,3,4,5]];data.qpos[f]=command[7:]
    mujoco.mj_fwdPosition(model,data)
    fk=kin.poses(pose,dict(zip(m['joint_names']['right'],commands[1,7:],strict=True)))
    for bid,body in enumerate(bodies):
        if not body.startswith('dex3_right_'):continue
        matches=[n for n in fk if body.endswith('_'+n)];assert len(matches)==1;n=matches[0]
        error=max(float(abs(data.xpos[bid]-fk[n][:3,3]).max()),float(abs(data.xmat[bid].reshape(3,3)-fk[n][:3,:3]).max()));assert error<2e-6;max_fk=max(max_fk,error)
    object_vertices=np.concatenate([world_vertices(g) for g,b in enumerate(model.geom_bodyid) if b==object_id]);tree=cKDTree(object_vertices)
    upper={}
    for digit,last in [('thumb',2),('index',1),('middle',1)]:
        gs=[g for g,b in enumerate(model.geom_bodyid) if bodies[b].endswith(f'right_hand_{digit}_{last}_link')]
        upper[digit]=float(min(tree.query(world_vertices(g))[0].min() for g in gs))
    base=pose[:3]+Rotation.from_quat(pose[3:7]).apply(chain[0]);distance=float(np.linalg.norm(point-base));lower=max(0.,distance-reach)
    # Randomly sampled actual poses must remain INSIDE the analytic sphere.
    rng=np.random.default_rng(37);thumb_names=[f'right_hand_thumb_{i}_joint' for i in range(3)];random_max=0.
    for _ in range(16):
        positions=dict(zip(m['joint_names']['right'],commands[1,7:],strict=True))
        for n in thumb_names:positions[n]=rng.uniform(*kin.limits[n])
        f=kin.poses(pose,positions);v=np.einsum('ni,ji->nj',body_vertices(thumb_geom),f['right_hand_thumb_2_link'][:3,:3],optimize=False)+f['right_hand_thumb_2_link'][:3,3]
        random_max=max(random_max,float(np.linalg.norm(v-base,axis=1).max()))
    assert random_max<=reach+1e-12
    contacts=[]
    for c in data.contact[:data.ncon]:
        a,b=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
        if bodies[a].startswith('dex3_right_') or bodies[b].startswith('dex3_right_'):
            contacts.append(dict(bodies=[bodies[a],bodies[b]],signed_gap_m=float(c.dist)))
    rows.append(dict(variant=name,source_pose_is_unqualified=True,finger_commands_fixed=True,root_shift_from_current_m=float(np.linalg.norm(pose[:3]-commands[1,:3])),root_rotation_from_current_deg=float(np.rad2deg((Rotation.from_quat(pose[3:7])*Rotation.from_quat(commands[1,3:7]).inv()).magnitude())),fixed_site_from_thumb0_distance_m=distance,relaxed_thumb_chain_reach_upper_m=reach,necessary_fixed_root_site_distance_lower_m=lower,fixed_root_distal_site_within_2mm_ruled_out_by_sphere=lower>.002,sphere_containment_random_control_passed=True,maximum_sampled_actual_thumb_reach_m=random_max,distal_vertex_pair_object_distance_upper_m=upper,native_contacts_with_current_fingers=contacts,maximum_penetration_with_current_fingers_m=max([max(0,-c['signed_gap_m']) for c in contacts]+[0])))
report=dict(schema='v2d.source-wrist-position-and-thumb-workspace-audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),inputs_sha256={str(p):sha(p) for p in [P/'frozen/manifest.json',model_path,prior_path,rotation_path,site_path]},thumb_chain_origin_translations_m=[p.tolist() for p in chain],actual_distal_mesh_body_radius_m=radius,analytic_relaxed_chain_reach_upper_m=reach,rows=rows,maximum_native_URDF_FK_error=max_fk,frozen_site_is_an_inferred_proposal=True,bound_ignores_joint_limits_facing_constraints_and_collisions=True,positive_lower_bound_is_necessary_failure_only_for_exact_fixed_root_and_site=True,no_unreachability_claim_for_coupled_root_boxes_or_other_sites=True,zero_lower_bound_does_not_certify_reachability=True,best_source_pose_not_selected=True,source_pose_uncertainty_not_metric_truth=True,source_observation_flags_or_historical_failures_unchanged=True,command_files_not_changed=True,physics_stepped=False,new_GPU_work=False,new_PPO_steps=0,source_reference_qualified=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start)
for name,h in m['files'].items():assert sha(P/'frozen'/name)==h
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

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
from trimesh.triangles import closest_point as triangle_closest_point
from scipy.optimize import differential_evolution,minimize
from scipy.spatial import cKDTree,ConvexHull
from audit_hand_reference_contacts import canonical_id
from v2d_dex3_flashchord.kinematics import URDFKinematics
P=S/'e037-motion-constrained-wrist-native-v1';O=S/'e037-coupled-grasp-witnesses-v1';O.mkdir(exist_ok=True)
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
local={};faces={}
for g in set(object_geoms+[g for gs in groups.values() for g in gs]):
    mid=int(model.geom_dataid[g]);assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
    a=int(model.mesh_vertadr[mid]);count=int(model.mesh_vertnum[mid]);verts=np.asarray(model.mesh_vert[a:a+count],float)
    hull=ConvexHull(verts);local[g]=verts[hull.vertices]
    remap=np.full(len(verts),-1);remap[hull.vertices]=np.arange(len(hull.vertices));faces[g]=remap[hull.simplices];assert (faces[g]>=0).all()
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


with np.load(S/'e037-coupled-wrist-finger-reach-v1/single-knot-finger-command.npz') as z:q=z['right_command'].copy()
w,f=binding['right'];data.qpos[:]=base_qpos;data.qpos[w[:3]]=q[:3];data.qpos[w[3]:w[3]+4]=q[[6,3,4,5]];data.qpos[f]=q[7:];mujoco.mj_fwdPosition(model,data)
# All hull planes use their outward ConvexHull normals, verified with centroids.
planes={};normal_parts=[]
for g in object_geoms:
    v=vertices(g);h=ConvexHull(v);planes[g]=h.equations
    triangles=object_vertices[object_faces[sum(len(faces[k]) for k in object_geoms[:object_geoms.index(g)]):sum(len(faces[k]) for k in object_geoms[:object_geoms.index(g)+1])]]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None]
    centers=triangles.mean(1);reverse=np.einsum('ni,ni->n',normals,centers-v.mean(0),optimize=False)<0;normals[reverse]*=-1
    assert np.all(np.einsum('ni,ni->n',normals,centers-v.mean(0),optimize=False)>0)
    normal_parts.append(normals)
object_normals=np.concatenate(normal_parts)
rows=[]
for digit,gs in groups.items():
    candidates=[]
    for g in gs:
        a=vertices(g);distance,nearest=tree.query(a);chosen=np.argsort(distance)[:8]
        tids=np.concatenate([object_incident[int(nearest[i])] for i in chosen]);point_ids=np.concatenate([np.full(len(object_incident[int(nearest[i])]),i) for i in chosen])
        points=a[point_ids];closest=triangle_closest_point(object_triangles[tids],points);dist=np.linalg.norm(points-closest,axis=1);i=int(np.argmin(dist));tid=int(tids[i]);point=closest[i]
        containing=[]
        for og,eq in planes.items():
            values=np.einsum('ni,i->n',eq[:,:3],point,optimize=False)+eq[:,3]
            if values.max() < -1e-7:containing.append(og)
        candidates.append(dict(digit=digit,native_hand_geom=g,distance_upper_bound_m=float(dist[i]),hand_point_native_world_m=points[i].tolist(),object_point_native_world_m=point.tolist(),object_outward_normal_world=object_normals[tid].tolist(),point_strictly_inside_other_object_hulls=containing,external_union_surface_candidate=not containing))
    rows.append(min(candidates,key=lambda r:r['distance_upper_bound_m']))
normal_dots={a['digit']+'/'+b['digit']:float(np.dot(a['object_outward_normal_world'],b['object_outward_normal_world'])) for i,a in enumerate(rows) for b in rows[i+1:]}
contacts=[]
for c in data.contact[:data.ncon]:
    a,b=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
    if a in hand_bodies or b in hand_bodies:contacts.append(dict(bodies=[bodies[a],bodies[b]],signed_gap_m=float(c.dist)))
report=dict(schema='v2d.coupled-grasp-surface-witness-audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),candidate_sha256=sha(S/'e037-coupled-wrist-finger-reach-v1/single-knot-finger-command.npz'),native_model_sha256=sha(P/'native-model.mjb'),witnesses=rows,object_outward_normal_pair_dot_products=normal_dots,native_contacts=contacts,actual_object_contacts=sum('wooden_spoon_object' in r['bodies'] for r in contacts),all_witnesses_on_external_union_surface=all(r['external_union_surface_candidate'] for r in rows),grasp_or_force_closure_certified=False,friction_or_contact_forces_measured=False,physics_stepped=False,new_GPU_work=False,new_PPO_steps=0,physical_task_accepted=False,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

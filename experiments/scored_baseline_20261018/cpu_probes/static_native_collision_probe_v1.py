"""Validate a local CPU contact-only model against all archived native knots.

Rebuild convex contact surfaces from recorded MuJoCo arrays. Flat free bodies
are assigned exact reference transforms from frozen URDF FK. This never steps
physics and cannot stand in for the articulated solver or its acceptance gate.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

S = Path(__file__).resolve().parents[1]
R = Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0, str(R/'scripts/dex3_training'))
from v2d_dex3_flashchord.kinematics import URDFKinematics

P = S/'scored-cohort-14-results/policy-e037'
OUT = S/'static-native-collision-probe-v1'
OUT.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
inputs = P/'frozen'
manifest = json.loads((inputs/'manifest.json').read_text())
for name, expected in manifest['files'].items():
    assert sha(inputs/name) == expected
with np.load(P/'readback/native-model-readback.npz', allow_pickle=False) as z:
    native = {k:z[k].copy() for k in z.files}
with np.load(P/'readback/native-state-readback.npz', allow_pickle=False) as z:
    # Support is static. No moving hand/object state is taken from the rollout.
    support_pose = z['body_poses'][0,29].copy()
with np.load(inputs/'targets.npz', allow_pickle=False) as z:
    q, objects, clock = z['robot_qpos'].copy(), z['object_poses'].copy(), z['frame_indices'].copy()
assert q.shape == (151,2,14) and np.array_equal(clock,np.arange(151))
assert native['geom_type'].tolist() == [7]*len(native['geom_type'])
labels = json.loads((P/'readback/body-labels.json').read_text())[:30]
hand_report = json.loads((S/'all-scored-hand-audit-results-v1/e037.json').read_text())
object_report = json.loads((P/'reference-contacts.json').read_text())
assert hand_report['input_manifest_sha256'] == object_report['input_manifest_sha256'] == sha(inputs/'manifest.json')
started = time.monotonic()
report = dict(schema='v2d.flat-static-native-contact-validation.v1',episode=37,
              producer_sha256=sha(Path(__file__)), frozen_manifest_sha256=sha(inputs/'manifest.json'),
              native_model_arrays_sha256=sha(P/'readback/native-model-readback.npz'),
              hand_control_report_sha256=sha(S/'all-scored-hand-audit-results-v1/e037.json'),
              object_control_report_sha256=sha(P/'reference-contacts.json'),
              mujoco_version=mujoco.__version__, source_targets_modified=False,
              physics_stepped=False, articulated_dynamics_equivalence_claimed=False,
              candidate_promoted=False, physical_task_accepted=False, rows=[])
save = lambda: (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')


def values(x):
    return ' '.join(f'{float(v):.17g}' for v in np.asarray(x).ravel())


try:
    xml = ET.Element('mujoco',model='recorded-contact-surfaces-static-only')
    ET.SubElement(xml,'compiler',inertiafromgeom='false',fusestatic='false',alignfree='false')
    ET.SubElement(xml,'option',gravity='0 0 0')
    assets = ET.SubElement(xml,'asset')
    world = ET.SubElement(xml,'worldbody')
    used = sorted(set(int(v) for v in native['geom_bodyid']))
    assert used[0] > 0
    for mid, (a,n) in enumerate(zip(native['mesh_vertadr'],native['mesh_vertnum'],strict=True)):
        ET.SubElement(assets,'mesh',name=f'm{mid}',vertex=values(native['mesh_vert'][int(a):int(a+n)]),inertia='convex')
    for bid in used:
        body = ET.SubElement(world,'body',name=f'b{bid}')
        ET.SubElement(body,'inertial',pos='0 0 0',mass='1',diaginertia='.001 .001 .001')
        ET.SubElement(body,'freejoint',name=f'j{bid}')
        for gi in np.flatnonzero(native['geom_bodyid'] == bid):
            hand = labels[bid-1].startswith(('dex3_left/','dex3_right/'))
            attrs = dict(name=f'g{gi}',type='mesh',mesh=f'm{int(native["geom_dataid"][gi])}',
                         pos=values(native['geom_pos'][gi]),quat=values(native['geom_quat'][gi]),
                         contype='1' if hand else '2',conaffinity='2' if hand else '3',
                         condim=str(int(native['geom_condim'][gi])),priority=str(int(native['geom_priority'][gi])),
                         friction=values(native['geom_friction'][gi]),solref=values(native['geom_solref'][gi]),
                         solimp=values(native['geom_solimp'][gi]),margin=str(native['geom_margin'][gi]),gap=str(native['geom_gap'][gi]))
            ET.SubElement(body,'geom',**attrs)
    xml_bytes = ET.tostring(xml)
    (OUT/'static-contact-surfaces.xml').write_bytes(xml_bytes)
    model = mujoco.MjModel.from_xml_string(xml_bytes.decode())
    for name in ['cone','impratio','solver','integrator','iterations','ls_iterations','timestep','enableflags','disableflags']:
        setattr(model.opt,name,native['option_'+name].item())
    data = mujoco.MjData(model)
    # Model compiler may recenter mesh assets. Validate physical vertex clouds
    # in each unchanged body frame before trusting a single contact result.
    maximum_surface_error = 0.
    for gi in range(len(native['geom_type'])):
        a = int(native['mesh_vertadr'][native['geom_dataid'][gi]])
        n = int(native['mesh_vertnum'][native['geom_dataid'][gi]])
        v = native['mesh_vert'][a:a+n]
        original = Rotation.from_quat(native['geom_quat'][gi][[1,2,3,0]]).apply(v)+native['geom_pos'][gi]
        gid = model.geom(f'g{gi}').id
        mid = model.geom_dataid[gid]
        a,n = int(model.mesh_vertadr[mid]),int(model.mesh_vertnum[mid])
        rebuilt = Rotation.from_quat(model.geom_quat[gid][[1,2,3,0]]).apply(model.mesh_vert[a:a+n])+model.geom_pos[gid]
        error = max(cKDTree(original).query(rebuilt)[0].max(),cKDTree(rebuilt).query(original)[0].max())
        maximum_surface_error = max(maximum_surface_error,float(error))
    report['maximum_body_local_surface_error_m'] = maximum_surface_error
    assert maximum_surface_error < 1e-6, 'Recompiled surfaces differ'
    kin = {side:URDFKinematics(inputs/f'robot/urdfs/g1/dex3_{side}.urdf') for side in ['left','right']}
    max_gap_error = 0.
    false_clear = false_penetrating = 0
    for frame in range(len(clock)):
        poses = {}
        for si,side in enumerate(['left','right']):
            fk = kin[side].poses(q[frame,si,:7],dict(zip(manifest['joint_names'][side],q[frame,si,7:],strict=True)))
            for bid in used:
                label = labels[bid-1]
                if label.startswith('dex3_'+side+'/'):
                    poses[bid] = fk[label.split('/',1)[1]]
        for bid in used:
            label = labels[bid-1]
            if label == 'support':
                position = support_pose
            elif label == 'wooden_spoon/object':
                position = objects[frame,0]
            else:
                position = None
            if position is not None:
                T = np.eye(4)
                T[:3,:3] = Rotation.from_quat(position[3:]).as_matrix()
                T[:3,3] = position[:3]
                poses[bid] = T
        assert set(poses) == set(used)
        for bid,T in poses.items():
            adr = int(model.jnt_qposadr[model.joint(f'j{bid}').id])
            data.qpos[adr:adr+3] = T[:3,3]
            data.qpos[adr+3:adr+7] = Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]]
        mujoco.mj_fwdPosition(model,data)
        actual,expected = {},{}
        def pair_key(a,b):
            a,b = labels[a-1],labels[b-1]
            if a.startswith('dex3_'):
                return (a.split('/')[0].split('_')[1],b.split('/')[0])
            if b.startswith('dex3_'):
                return (b.split('/')[0].split('_')[1],a.split('/')[0])
            return tuple(sorted([a.split('/')[0],b.split('/')[0]]))
        for contact in data.contact[:data.ncon]:
            ids = [int(model.geom_bodyid[g]) for g in [contact.geom1,contact.geom2]]
            original_ids = [int(model.body(i).name[1:]) for i in ids]
            key = pair_key(*original_ids)
            actual[key] = max(actual.get(key,0.),max(0.,-float(contact.dist)))
        for c in hand_report['frames'][frame]['contacts']:
            key = (c['hand'],c['target'])
            expected[key] = max(expected.get(key,0.),max(0.,-c['signed_gap_m']))
        for c in object_report['frames'][frame]['contact_pairs']:
            expected[tuple(c['bodies'])] = c['maximum_penetration_m']
        row=[]
        for key in sorted(set(actual)|set(expected)):
            a,b = actual.get(key,0.),expected.get(key,0.)
            max_gap_error = max(max_gap_error,abs(a-b))
            false_clear += int(a <= .001 < b)
            false_penetrating += int(b <= .001 < a)
            row.append(dict(pair=key,local_penetration_m=a,archived_native_penetration_m=b))
        report['rows'].append(dict(frame=frame,pairs=row))
    report.update(maximum_penetration_difference_m=max_gap_error,false_clear_pairs=false_clear,
                  false_penetrating_pairs=false_penetrating,reference_knots=len(clock),
                  static_geometry_validation_passed=bool(max_gap_error < 1e-5 and false_clear == 0 and false_penetrating == 0))
except BaseException as error:
    report['error'] = repr(error)
    raise
finally:
    report['wall_seconds'] = time.monotonic()-started
    save()
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}),flush=True)
